from datasets import load_dataset
from vagen.env.create_dataset import DatasetCreator
from typing import Union, List, Dict, Optional
import os

class SVGDatasetCreator(DatasetCreator):
    """Dataset creator for SVG data, adapting to Hugging Face dataset format"""

    def __init__(self, config: Dict):
        super().__init__(config)
        # Save dataset name/path
        self.dataset_name = self.env_config.get('dataset_name', 'default_dataset')
        
    def create_dataset(
        self, 
        force_gen: bool = False,
        train_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
    ):
        """
        Create SVG dataset utilizing Hugging Face dataset's built-in splits
        
        Args:
            force_gen: If True, regenerate the dataset even if files already exist
            train_samples: Number of training samples, use all if None
            test_samples: Number of test samples, use all if None
        """
        # Check output file paths
        train_file_path = os.path.join(self.data_dir, 'train.parquet')
        test_file_path = os.path.join(self.data_dir, 'test.parquet')
        
        # Check if files already exist
        if not force_gen and os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return
            
        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load Hugging Face dataset
        hf_dataset = load_dataset(self.dataset_name)
        
        # Process each split
        splits = ["train", "test"]
        sample_limits = [train_samples, test_samples]
        output_files = [train_file_path, test_file_path]
        
        for split, sample_limit, output_file in zip(splits, sample_limits, output_files):
            if split in hf_dataset:
                # Get split data
                split_data = hf_dataset[split]
                
                # If sample count specified and not using full dataset, sample accordingly
                if sample_limit is not None and not use_full_dataset:
                    sample_limit = min(sample_limit, len(split_data))
                    indices = list(range(sample_limit))
                    split_data = split_data.select(indices)
                
                # Create instances with environment configuration
                instances = []
                for idx, item in enumerate(split_data):
                    env_settings = {
                        'env_name': self.env_name,
                        'env_config': {
                            **self.env_config,
                            'svg_filename': item['Filename'],
                            'svg_code': item['Svg'],
                            'item_idx': idx
                        },
                        'interface_config': self.interface_config,
                        'seed': idx  # Use index as seed
                    }
                    
                    instances.append({
                        "data_source": self.env_name,
                        "prompt": [{"role": "user", "content": ''}],
                        "extra_info": {"split": split, **env_settings}
                    })
                
                # Create dataset and save
                if instances:
                    from datasets import Dataset
                    dataset = Dataset.from_list(instances)
                    dataset.to_parquet(output_file)
                    print(f"Created {split} dataset with {len(instances)} samples at {output_file}")
            else:
                print(f"Split '{split}' not found in dataset")
        
        print(f"Dataset successfully generated at {self.data_dir}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-gen', action='store_true', 
                      help='Force dataset generation even if files already exist')
    parser.add_argument('--data_dir', type=str, default='data/svg',
                      help='Directory to save the processed dataset')
    parser.add_argument('--dataset_name', type=str, default='starvector/svg-emoji-simple',
                      help='Hugging Face dataset name or path')
    # @TODO customize sample number
    parser.add_argument('--train_samples', type=int, default=None,
                      help='Number of training samples to use')
    parser.add_argument('--test_samples', type=int, default=None, 
                      help='Number of test samples to use')
    
    # interface_config
    parser.add_argument('--max_action_per_step', type=int, default=1,
                        help='Maximum number of actions per step')
    parser.add_argument('--max_action_penalty', type=float, default=0,
                        help='Penalty for exceeding the maximum number of actions per step')
    parser.add_argument('--format_reward', type=float, default=0,
                        help='Reward for correct formatting')
    parser.add_argument('--format_penalty', type=float, default=0,
                        help='Penalty for incorrect formatting')
    
    args, colab_par = parser.parse_known_args()

    args.name = 'svg'    
    args.env_config = {'dataset_name': args.dataset_name}
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
        'format_penalty': args.format_penalty,
    }
    # Create dataset
    creator = SVGDatasetCreator(config=vars(args))
    creator.create_dataset(
        force_gen=args.force_gen,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
