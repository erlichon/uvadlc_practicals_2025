"""
Evaluation script for Q2.8.b - Testing accumulated knowledge of the GPT model
with various generation parameters.

Usage:
    python evaluate_generation.py --model_weights_folder ./logs/gpt-mini/version_0/checkpoints

Output:
    - generation_results.json: Results in JSON format for further processing
"""

import argparse
import os
import json
import torch
import pytorch_lightning as pl
from dataset import TextDataset, CharTokenizer
from cfg import get_config
from gpt import GPT
from generate import generate, GPTLightningModule


def main():
    # Parse arguments
    args = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_folder', type=str, default='./logs/gpt-mini/version_0/checkpoints')
    parser.add_argument('--output_file', type=str, default='generation_results.json')
    parser.add_argument('--num_generated_tokens', type=int, default=100)
    parser.add_argument('--pretrained_tokenizer', action='store_true')
    
    # Allow custom prompts and configurations via JSON file
    parser.add_argument('--config_file', type=str, default=None, 
                        help='Optional JSON file with custom prompts and configurations')
    
    gen_args = parser.parse_args()
    
    for key, value in vars(gen_args).items():
        setattr(args, key, value)

    pl.seed_everything(args.seed)

    # Load model (same as generate.py)
    print("Loading model...")
    model_weights_path = os.path.join(args.model_weights_folder, sorted(os.listdir(args.model_weights_folder))[-1])
    state_dict = torch.load(model_weights_path)

    if state_dict['hyper_parameters'].get('compile', False) and 'state_dict' in state_dict:
        cleaned_state_dict = {}
        for key, value in state_dict['state_dict'].items():
            new_key = key.replace('model._orig_mod.', 'model.')
            cleaned_state_dict[new_key] = value
        state_dict['state_dict'] = cleaned_state_dict

    default_cfg = GPT.get_default_config()
    saved_cfg = state_dict['hyper_parameters']
    saved_cfg = argparse.Namespace(**saved_cfg)

    default_cfg_dict = vars(default_cfg)
    saved_cfg_dict = vars(saved_cfg)
    combined_cfg = {**default_cfg_dict, **saved_cfg_dict}
    cfg = argparse.Namespace(**combined_cfg)
    
    gpt_model = GPT(cfg)

    if args.pretrained_tokenizer:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        args.vocab_size = tokenizer.max_token_value
    else:
        tokenizer = CharTokenizer(args.txt_file)
        args.vocab_size = tokenizer.vocab_size

    dataset = TextDataset(args, args.txt_file, args.block_size, tokenizer)
    model = GPTLightningModule(cfg, gpt_model, dataset)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()

    device = next(model.parameters()).device
    print(f"Running on device: {device}")

    # Load custom config or use defaults
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
        prompts = custom_config.get('prompts', None)
        configurations = custom_config.get('configurations', None)
    else:
        prompts = None
        configurations = None

    # Default prompts if not specified
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The king said to",
            "In the dark forest",
            "And they lived happily",
            "The old woman gave",
        ]

    # Default configurations if not specified
    if configurations is None:
        configurations = [
            # Greedy
            {"name": "Greedy", "do_sample": False, "top_k": None, "top_p": None, "temperature": 1.0},
            
            # Top-p varying p at T=1.0 (full range)
            {"name": "Top-p (T=1.0, p=0.4)", "do_sample": True, "top_k": None, "top_p": 0.4, "temperature": 1.0},
            {"name": "Top-p (T=1.0, p=0.5)", "do_sample": True, "top_k": None, "top_p": 0.5, "temperature": 1.0},
            {"name": "Top-p (T=1.0, p=0.6)", "do_sample": True, "top_k": None, "top_p": 0.6, "temperature": 1.0},
            {"name": "Top-p (T=1.0, p=0.8)", "do_sample": True, "top_k": None, "top_p": 0.8, "temperature": 1.0},
            {"name": "Top-p (T=1.0, p=0.9)", "do_sample": True, "top_k": None, "top_p": 0.9, "temperature": 1.0},
            {"name": "Top-p (T=1.0, p=0.95)", "do_sample": True, "top_k": None, "top_p": 0.95, "temperature": 1.0},
            
            # Top-p with p=0.85 - varying temperature
            {"name": "Top-p (T=0.5, p=0.85)", "do_sample": True, "top_k": None, "top_p": 0.85, "temperature": 0.5},
            {"name": "Top-p (T=0.8, p=0.85)", "do_sample": True, "top_k": None, "top_p": 0.85, "temperature": 0.8},
            {"name": "Top-p (T=1.0, p=0.85)", "do_sample": True, "top_k": None, "top_p": 0.85, "temperature": 1.0},
            {"name": "Top-p (T=1.5, p=0.85)", "do_sample": True, "top_k": None, "top_p": 0.85, "temperature": 1.5},
            
            # Top-k with varying k
            {"name": "Top-k (T=1.0, k=5)", "do_sample": True, "top_k": 5, "top_p": None, "temperature": 1.0},
            {"name": "Top-k (T=1.0, k=20)", "do_sample": True, "top_k": 20, "top_p": None, "temperature": 1.0},
            {"name": "Top-k (T=1.0, k=50)", "do_sample": True, "top_k": 50, "top_p": None, "temperature": 1.0},
        ]

    results = {
        "metadata": {
            "model_weights_folder": args.model_weights_folder,
            "num_generated_tokens": args.num_generated_tokens,
            "seed": args.seed,
        },
        "prompts": prompts,
        "configurations": configurations,
        "generations": []
    }
    
    print("=" * 100)
    print("GENERATION EVALUATION - Testing Accumulated Knowledge")
    print("=" * 100)

    total_runs = len(prompts) * len(configurations)
    current_run = 0

    for prompt in prompts:
        print(f"\n{'='*100}")
        print(f"PROMPT: \"{prompt}\"")
        print("=" * 100)
        
        for config in configurations:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] {config['name']}")
            
            try:
                # Use the original generate function from generate.py
                # num_samples=1 to get a single output per configuration
                outputs = generate(
                    model=model,
                    model_type=cfg.model_type,
                    prompt=prompt,
                    num_samples=1,
                    n_steps=args.num_generated_tokens,
                    do_sample=config['do_sample'],
                    top_k=config['top_k'],
                    top_p=config['top_p'],
                    temperature=config['temperature'],
                    device=device,
                    verbose=False  # We'll print ourselves
                )
                
                output = outputs[0]  # Get the single sample
                
                generation_result = {
                    "prompt": prompt,
                    "config_name": config['name'],
                    "do_sample": config['do_sample'],
                    "top_k": config['top_k'],
                    "top_p": config['top_p'],
                    "temperature": config['temperature'],
                    "full_output": output,
                    "generated_only": output[len(prompt):],
                    "error": None
                }
                
                print("-" * 80)
                print(output[:200] + "..." if len(output) > 200 else output)
                
            except Exception as e:
                print(f"ERROR: {e}")
                generation_result = {
                    "prompt": prompt,
                    "config_name": config['name'],
                    "do_sample": config['do_sample'],
                    "top_k": config['top_k'],
                    "top_p": config['top_p'],
                    "temperature": config['temperature'],
                    "full_output": None,
                    "generated_only": None,
                    "error": str(e)
                }
            
            results["generations"].append(generation_result)

    # Save results to JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {args.output_file}")
    print(f"Total generations: {len(results['generations'])}")


if __name__ == "__main__":
    main()