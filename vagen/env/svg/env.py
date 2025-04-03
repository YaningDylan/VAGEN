import os
import re
import torch
import random 
from typing import Dict, Any, Optional, List, Union, Tuple
from datasets import Dataset
from PIL import Image
import random
from vagen.env.register import register
from vagen.env.base import BaseInterface, BaseEnv, IMAGE_PLACEHOLDER
from vagen.env.utils import preprocess, PreprocessResult, postprocess
from vagen.env.svg.svg_utils import process_and_rasterize_svg
from vagen.env.svg.score import calculate_total_score
from vagen.env.svg.prompt import (
    init_observation_template,
    action_template,
    instruction_template,
)

class SVGEnv(BaseEnv):
    """
    Input Example (env_config):
        data_dir: str
        dataset_name: str
        split: str
            Dataset split, e.g., "train" or "test".
        score_config: dict
                model_size: str
                dino_only: bool
                dino_weight: float (optional)
                structural_weight: float (optional)
                color_weight: float (optional)
                code_weight: float (optional)
        seed: int

    Output:
        obs: dict
        reward: float
        done: bool
        info: dict
            - gt_svg_code: ground truth SVG
            - gen_svg_code: generated SVG
            - scores: dict of individual and total scores
    """
    def __init__(self, env_config: dict):
        """
        Args:
            dataset_path: 'data/svg/train(test).parquet'
            device: for dino reward model
        """
        #@TODO avoid double loading (one from here and one from trainer) check!
        self.env_config = env_config
        self.dataset_path = self.env_config.get('data_dir', '')
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        # load dataset
        self.dataset_path =  os.path.join(self.dataset_path, 'train.parquet')
        self.dataset = Dataset.from_parquet(self.dataset_path)
        self.done = False
        # random seed
        self.rng = random.Random()
        if "seed" in env_config:
            self.rng.seed(env_config["seed"])
        #@TODO Do we really need this?
        self.first_round = True
        self.infos = {}

        self.current_sample = None
        self.img_id = None
        self.gt_svg_code = None
        self.gt_image = None
        self.gen_svg_code = ""
        self.gt_image = None

    def _reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        dataset_length = len(self.dataset)
        index = self.rng.randint(0, dataset_length - 1)
        self.current_sample = self.dataset[index]

        self.current_sample = self.dataset[index]
        self.gt_svg_code = self.current_sample['extra_info']['env_config'].get('svg_code', '')
        self.img_id = self.current_sample['extra_info']['env_config'].get('svg_filename', '')
        if not self.gt_svg_code:
            raise ValueError("Ground truth SVG code not found in the selected sample.")
        _, self.gt_image = process_and_rasterize_svg(self.gt_svg_code)
        self.done = False
        self.first_round = True

        obs = {
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
        #@TODO check obs latest action workflow
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
        
        self.gen_svg_code = action

        # calculate reward by reward model
        scores = self.calculate_total_score(
          gt_im=self.gt_image, 
          gen_im=gen_image, 
          gt_code=self.gt_svg_code, 
          gen_code=self.gen_svg_code, 
          score_config=self.env_config["score_config"]
        )
        reward = scores["total_score"]
        self.done = False  # single step task
        

        obs = {
            "latest_action": action
        }
        info = {
            "gt_svg_code": self.gt_svg_code,
            "gen_svg_code": action,
            "scores": scores 
        }
        return obs, reward, self.done, info
    
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
        self.env = SVGEnv(env_config=env_config)

        self.max_action_per_step = interface_config.get('max_action_per_step', 1)
        self.max_action_penalty = interface_config.get('max_action_penalty', 0.0)
        self.format_reward = interface_config.get('format_reward', 0.0)
        self.format_penalty = interface_config.get('format_penalty', 0.0)

        self.INVALID_ACTION = 0

    @classmethod
    def _extract_one_action(cls, text):
        """Extract single action from text, the input text should ensure only one action contained"""
        return text

    def _reset(self, seed: Optional[int] = None) -> Dict:
        obs, _ = self.env._reset(seed=seed)

        self.traj_reward = 0
        observation = IMAGE_PLACEHOLDER
        text_template = init_observation_template.format(
            observation=observation,
        )
        obs["text_template"] = text_template
        return obs, {}

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
            done = True
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
        #@ Add a "Trash Bin" here
        if env_state == "Invalid answer" or "":
            return {"text_template": env_state}, reward, done, final_info
        _, image = process_and_rasterize_svg(env_state)
        
        #@TODO sometimes cause image token out of memory (why limit_mm_per_prompt doesn't work?)
        observation = IMAGE_PLACEHOLDER
        text_template = action_template.format(
            observation=observation,
            reward=reward
        )
        obs = {"text_template": text_template, "multi_modal_data": image}
        return obs, reward, done, final_info

    def close(self):
        self.env.close()

    @classmethod
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

        return f"{env_config_str}, {interface_config_str}"
    
    def get_task_instruction(self) -> str:
        return instruction_template

    def get_traj_reward(self):
        return self.traj_reward
