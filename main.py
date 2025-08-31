
#!/usr/bin/env python3
import argparse
import json
import sys
import time
import logging
from pathlib import Path

from src.dataset_generator import DatasetGenerator
from src.convert_to_alpaca import convert_to_alpaca_format

def setup_logging():
    """Setup logging configuration"""
    log_file = f"logs/pipeline_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_smoke_test(lang: str) -> bool:
    """Run smoke test for specified language"""
    logging.info(f"Running smoke test for {lang}")
    
    try:
        generator = DatasetGenerator()
        results = generator.run_smoke_test(lang, target_count=20)
        
        print(f"\n=== SMOKE TEST RESULTS ({lang.upper()}) ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        if results["success"]:
            logging.info(f"Smoke test PASSED for {lang}")
            return True
        else:
            logging.error(f"Smoke test FAILED for {lang}")
            return False
            
    except Exception as e:
        logging.error(f"Smoke test error for {lang}: {e}", exc_info=True)
        return False

def run_full_generation(lang: str, target: int = 2000) -> bool:
    """Run full dataset generation"""
    logging.info(f"Running full generation for {lang}, target: {target}")
    
    try:
        generator = DatasetGenerator()
        results = generator.generate_full_dataset(lang, target)
        
        print(f"\n=== FULL GENERATION RESULTS ({lang.upper()}) ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        if results["success"]:
            logging.info(f"Full generation COMPLETED for {lang}")
            return True
        else:
            logging.warning(f"Full generation completed with issues for {lang}")
            return False
            
    except Exception as e:
        logging.error(f"Full generation error for {lang}: {e}", exc_info=True)
        return False

def merge_and_convert():
    """Merge datasets and create Alpaca format"""
    logging.info("Merging datasets and creating Alpaca format")
    
    try:
        ar_file = "data/generation_stage_B/ar/judgmental_ar_final.jsonl"
        en_file = "data/generation_stage_B/en/judgmental_en_final.jsonl"
        
        if not Path(ar_file).exists() or not Path(en_file).exists():
            logging.error("Missing final dataset files")
            return False
            
        # Convert to Alpaca format
        convert_to_alpaca_format(ar_file, "output/alpaca/judgmental_alpaca_ar.jsonl")
        convert_to_alpaca_format(en_file, "output/alpaca/judgmental_alpaca_en.jsonl")
        
        # Create train/val/test splits
        create_splits("output/alpaca/judgmental_alpaca_ar.jsonl", "ar")
        create_splits("output/alpaca/judgmental_alpaca_en.jsonl", "en")
        
        # Create Axolotl config
        create_axolotl_config()
        
        logging.info("Merge and conversion completed")
        return True
        
    except Exception as e:
        logging.error(f"Merge error: {e}", exc_info=True)
        return False

def create_splits(file_path: str, lang: str):
    """Create train/val/test splits"""
    import random
    
    with open(file_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
        
    random.shuffle(examples)
    total = len(examples)
    
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = examples[:train_size]
    val_data = examples[train_size:train_size + val_size]
    test_data = examples[train_size + val_size:]
    
    # Save splits
    base_dir = Path(f"data/generation_stage_B/{lang}")
    
    with open(base_dir / f"train_{lang}.jsonl", 'w', encoding='utf-8') as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    with open(base_dir / f"val_{lang}.jsonl", 'w', encoding='utf-8') as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    with open(base_dir / f"test_{lang}.jsonl", 'w', encoding='utf-8') as f:
        for ex in test_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def create_axolotl_config():
    """Create Axolotl configuration file"""
    config = {
        "base_model": "meta-llama/Llama-2-7b-hf",
        "model_type": "LlamaForCausalLM",
        "tokenizer_type": "LlamaTokenizer",
        
        "load_in_8bit": True,
        "load_in_4bit": False,
        "strict": False,
        
        "datasets": [
            {
                "path": "data/generation_stage_B/ar/train_ar.jsonl",
                "type": "alpaca"
            },
            {
                "path": "data/generation_stage_B/en/train_en.jsonl", 
                "type": "alpaca"
            }
        ],
        
        "val_set_size": 0.1,
        "output_dir": "./qlora-out",
        
        "adapter": "qlora",
        "lora_model_dir": "./qlora-out",
        
        "sequence_len": 2048,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        
        "lora_r": 32,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        
        "wandb_project": "aaoifi-judgmental",
        "wandb_entity": "",
        "wandb_watch": "",
        "wandb_name": "",
        "wandb_log_model": "",
        
        "gradient_accumulation_steps": 4,
        "micro_batch_size": 2,
        "num_epochs": 3,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.0002,
        
        "train_on_inputs": False,
        "group_by_length": False,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        
        "gradient_checkpointing": True,
        "early_stopping_patience": "",
        "resume_from_checkpoint": "",
        "local_rank": "",
        
        "logging_steps": 1,
        "xformers_attention": "",
        "flash_attention": True,
        
        "warmup_steps": 10,
        "evals_per_epoch": 4,
        "eval_table_size": "",
        "saves_per_epoch": 1,
        "debug": "",
        "deepspeed": "",
        "weight_decay": 0.0,
        "fsdp": "",
        "fsdp_config": "",
        "special_tokens": {}
    }
    
    with open("config.example.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="AAOIFI Dataset Generation Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--full", action="store_true", help="Run full generation") 
    parser.add_argument("--merge", action="store_true", help="Merge datasets and create Alpaca format")
    parser.add_argument("--lang", choices=["ar", "en"], help="Language for smoke/full generation")
    parser.add_argument("--target", type=int, default=2000, help="Target examples for full generation")
    
    args = parser.parse_args()
    
    # Setup logging
    import time
    log_file = setup_logging()
    logging.info(f"Pipeline started, logging to: {log_file}")
    
    try:
        if args.smoke:
            if not args.lang:
                print("Error: --lang required for smoke test")
                sys.exit(1)
            success = run_smoke_test(args.lang)
            sys.exit(0 if success else 1)
            
        elif args.full:
            if not args.lang:
                print("Error: --lang required for full generation")
                sys.exit(1)
            success = run_full_generation(args.lang, args.target)
            sys.exit(0 if success else 1)
            
        elif args.merge:
            success = merge_and_convert()
            sys.exit(0 if success else 1)
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
