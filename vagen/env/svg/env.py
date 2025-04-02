import os
import re
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from datasets import Dataset
from PIL import Image
from vagen.env.register import register
from vagen.env.base import BaseInterface, BaseEnv, IMAGE_PLACEHOLDER
from vagen.env.utils import preprocess, PreprocessResult, postprocess
from vagen.env.svg.svg_utils import process_and_rasterize_svg
from vagen.env.svg.dino import DINOScoreCalculator
from vagen.env.svg.prompt import (instruction_template, init_observation_template)

class SVGEnv(BaseEnv):
    """
    input example:
      data_source: str
      prompt: list
      extra_info: dict
          env_config: dict
              data_dir: str
              dataset_name: str
              item_idx: int
              svg_code: str
              svg_filename: str
          env_name: str
          interface_config: dict
              format_penalty: int
              format_reward: int
              max_action_penalty: int
              max_action_per_step: int
          seed: int
          split: str
    output: obs, reward, done, info
    """
    def __init__(self, dataset_path: str, device: str = 'cuda'):
        """
        Args:
            dataset_path: 'data/svg/train(test).parquet'
            device: for dino reward model
        """
        #@TODO avoid double loading (one from here and one from trainer) check!
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        # load dataset
        dataset_path =  os.path.join(dataset_path, 'train.parquet')
        self.dataset = Dataset.from_parquet(dataset_path)
        self.device = device
        # init reward model
        self.reward_model = DINOScoreCalculator(device=self.device)
        self.done = False
        #@TODO Do we really need this?
        self.first_round = True
        self.infos = {}

        self.current_sample = None
        self.img_id = None
        self.gt_svg_code = None
        self.gt_image = None
        self.gen_svg_code = ""

    def _reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        #@TODO choose starting data by seed
        #@TODO check text template
        index = 0 if seed is None else seed % len(self.dataset)
        self.current_sample = self.dataset[index]
        self.gt_svg_code = self.current_sample['extra_info']['env_config'].get('svg_code', '')
        self.img_id = self.current_sample['extra_info']['env_config'].get('svg_filename', '')
        if not self.gt_svg_code:
            raise ValueError("Ground truth SVG code not found in the selected sample.")
        _, self.gt_image = process_and_rasterize_svg(self.gt_svg_code)
        self.done = False
        self.first_round = True

        obs = {
            'text_template': IMAGE_PLACEHOLDER,
            "multi_modal_data": {IMAGE_PLACEHOLDER: [self.gt_image]}
        }
        return obs, {}

    def _step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Args:
          action: generated svg code
        Returns:
          obs, reward, done, info
        """
        if not isinstance(action, str):
            reward = 0.0
            info = {"error": "Action must be string"}
            obs = {"latest_action": action}
            self.done = False
            return obs, reward, self.done, info

        try:
            _, gen_image = process_and_rasterize_svg(action)
        except Exception as e:
            obs = {"latest_action": action}
            info = {"error": f"Fail generate SVG code: {e}"}
            self.done = False
            return obs, 0.0, True, info

        # calculate reward by reward model
        score = self.reward_model.calculate_DINOv2_similarity_score(gt_im=self.gt_image, gen_im=gen_image)
        reward = score
        self.done = False  # single step task
        self.gen_svg_code = action

        obs = {
            "latest_action": action,
        }
        info = {
            "gt_svg_code": self.gt_svg_code,
            "gen_svg_code": action,
            "dino_score": score
        }
        return obs, reward, self.done, info
    # @TODO does it be used in training? return gt first
    def _render(self, mode='text'):
        assert mode == 'text'
        if self.first_round:
            self.first_round = False
            return self.gt_svg_code
        else:
            return self.gen_svg_code
    def close(self):
        pass

@register(name="svg")
class SVGInterface(BaseInterface):

    def __init__(self, env_config: Dict, interface_config: Dict):

        super().__init__(env_config)
        self.env_config = env_config
        self.interface_config = interface_config
        self.dataset_path = env_config.get('data_dir', '')
        self.env = SVGEnv(dataset_path=self.dataset_path)

        self.max_action_per_step = interface_config.get('max_action_per_step', 1)
        self.max_action_penalty = interface_config.get('max_action_penalty', 0.0)
        self.format_reward = interface_config.get('format_reward', 0.0)
        self.format_penalty = interface_config.get('format_penalty', 0.0)

        self.INVALID_ACTION = 0

    @classmethod
    def _extract_one_action(cls, text):
        """Extract single action from text, the input text should ensure only one action contained"""

        return text

    #@TODO check if text_template must be used in forming prompt
    def _reset(self, seed: Optional[int] = None) -> Dict:
        obs, _ = self.env._reset(seed=seed)

        self.traj_reward = 0
        env_state = self.env._render(mode='text') # svg_code
        _, image = process_and_rasterize_svg(env_state)
        return {"text_template": IMAGE_PLACEHOLDER, "multi_modal_data": {IMAGE_PLACEHOLDER: [image]}}, {}

    def extract_svg_code(self, text: str) -> str:
        svg_match = re.search(r'<svg.*?</svg>', text, re.DOTALL)
        if svg_match:
            return svg_match.group(0)

        if '<svg' in text and '</svg>' in text:
            start_idx = text.find('<svg')
            end_idx = text.rfind('</svg>') + 6  # 6 is the length of '</svg>'
            if start_idx < end_idx:
                return text[start_idx:end_idx]

        return ""

    def _step(self, raw_text: str) -> Tuple[Any, float, bool, Dict]:

        reward, done, final_info = 0, False, {}
        
        #@TODO Where to define INVALID_ACTION? Here or inside
        preprocess_result = preprocess(raw_text, self._extract_one_action, self.INVALID_ACTION) 
        think = preprocess_result.think
        action_list = preprocess_result.action_list
        answer = preprocess_result.answer
        final_info['llm_raw_response'] = preprocess_result.llm_raw_response

        # Avoid if preprocess does not work in svg scenerio @TODO integrate this code
        if not action_list:
            svg_code = self.extract_svg_code(final_info['llm_raw_response'])
            if svg_code:
                action_list = [svg_code]
        else:
            action_list = [self.extract_svg_code(action_list[0])]

        if not action_list:
            reward += self.interface_config['format_penalty']
            env_state = "Invalid answer"
            done = False
            info = {}

        else:
            reward += self.interface_config['format_reward']
            if len(action_list) > self.interface_config['max_action_per_step']:
                reward += self.interface_config['max_action_penalty']
                action_list = action_list[:self.interface_config['max_action_per_step']]
                preprocess_result.action_list = action_list
            _, env_reward, done, info = self.env.step(action_list[0])
            reward += env_reward
            env_state = self.env._render(mode='text')

        self.traj_reward += reward

        final_info.update(info) # NOTE currently only use the last step info
        if env_state == "Invalid answer" or "":
            return {"text_template": env_state}, reward, done, final_info
        _, image = process_and_rasterize_svg(env_state)
        
        #@TODO clean this part + sometimes cause image token out of memory (why limit_mm_per_prompt doesn't work?)
        observation = IMAGE_PLACEHOLDER
        text_template = init_observation_template.format(
            observation=observation,
        )
        return {"text_template": text_template, "multi_modal_data": {IMAGE_PLACEHOLDER: [image]}}, reward, done, final_info

    def close(self):
        self.env.close()

    @classmethod
    #@TODO revise this prompt (ValueError: The prompt (total length 1321) is too long to fit into the model (context length 1280). Make sure that max_model_len is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number of image tokens depends on the number of images, and possibly their aspect ratios as well.)
    def config_repr(cls, env_config: Dict, interface_config: Dict) -> str:
        """
        Create a string representation of the configuration.

        Args:
            env_config: Dictionary containing environment configuration
            interface_config: Dictionary containing interface configuration

        Returns:
            String representation of the configuration

        Raises:
            ValueError: If required keys are missing from the configuration
        """

        required_keys = ['data_dir', 'dataset_name']
        missing_keys = [key for key in required_keys if key not in env_config]
        if missing_keys:
            raise ValueError(f"Missing required keys in env_config: {missing_keys}")

        env_config_str = (
            f"SVGImage2Code(data_dir={env_config['data_dir']}, "
            f"dataset_name={env_config['dataset_name']})"
        )
        interface_config_str = (
            f"SVGInterface(max_action_per_step={interface_config.get('max_action_per_step', 1)}, "
            f"max_action_penalty={interface_config.get('max_action_penalty', 0.0)}, "
            f"format_reward={interface_config.get('format_reward', 0.0)}, "
            f"format_penalty={interface_config.get('format_penalty', 0.0)})"
        )

        system_prompt = (
            "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the answer."
        )
        instruction = (
            "You are a SVG image-to-code generator.\n\n"
            "Task: Given an image, generate SVG code that reproduces the image as accurately as possible.\n\n"
            "Your response should be formatted as follows:\n"
            "<think> ... your reasoning ... </think><answer> ... your SVG code ... </answer>"
        )

        prompt_summary = f"System Prompt:\n{system_prompt}\n\nInstruction:\n{instruction}"

        return f"{env_config_str}, {interface_config_str}\n\nPrompt Summary:\n{prompt_summary}"
    def get_task_instruction(self) -> str:
        return instruction_template.format(
            format_reward=self.interface_config['format_reward'],
            format_penalty=self.interface_config['format_penalty'],
        )

    def get_traj_reward(self):
        return self.traj_reward
