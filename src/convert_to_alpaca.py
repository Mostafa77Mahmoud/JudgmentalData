
import json
from pathlib import Path
from typing import Dict, List

def convert_to_alpaca_format(input_file: str, output_file: str):
    """Convert judgmental dataset to Alpaca format"""
    
    alpaca_examples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            
            # Create Alpaca format
            instruction = "Verify the following claim against the provided AAOIFI text context and determine if it is True or False."
            
            input_text = f"Claim: {example['claim']}\n\nContext: {example['context_excerpt']}"
            
            if example['verdict'] == "True":
                output_text = f"VERDICT: True\nReference: {example['reference']}\nExplanation: {example['explanation']}"
            else:
                output_text = f"VERDICT: False\nExplanation: {example['explanation']}"
                
            alpaca_example = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "metadata": {
                    "original_id": example['id'],
                    "language": example['language'],
                    "chunk_id": example['context_chunk_id'],
                    "suspected_fabrication": example['suspected_fabrication'],
                    "confidence": example['meta'].get('confidence', 0.0)
                }
            }
            
            alpaca_examples.append(alpaca_example)
    
    # Save Alpaca format
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in alpaca_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
    print(f"Converted {len(alpaca_examples)} examples to {output_file}")
