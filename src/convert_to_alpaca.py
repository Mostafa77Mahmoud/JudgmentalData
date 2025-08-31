
import json
import argparse
from pathlib import Path

def convert_to_alpaca(input_file: str, output_file: str):
    """Convert judgmental dataset to Alpaca format"""
    alpaca_examples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            
            alpaca_example = {
                "instruction": "Verify this claim against the provided AAOIFI context and determine if it's true or false.",
                "input": f"Claim: {example['claim']}\nContext: {example['context_excerpt']}",
                "output": f"VERDICT: {example['verdict']}\nExplanation: {example['explanation']}\nReference: {example['reference']}"
            }
            alpaca_examples.append(alpaca_example)
    
    # Save Alpaca format
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in alpaca_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(alpaca_examples)} examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert to Alpaca format")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--out", required=True, help="Output Alpaca JSONL file")
    
    args = parser.parse_args()
    convert_to_alpaca(args.input, args.out)

if __name__ == "__main__":
    main()
