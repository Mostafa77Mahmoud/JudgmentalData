
#!/usr/bin/env python3
"""
Test script for the strict dataset generator
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from strict_generator import StrictDatasetGenerator
from strict_prompts import get_strict_prompt

def test_strict_generator():
    """Test the strict generator"""
    print("Testing Strict Dataset Generator...")
    
    # Initialize generator
    generator = StrictDatasetGenerator()
    
    # Test with Arabic
    print("\n1. Testing Arabic example generation...")
    arabic_claim = "لا يجوز اشتراط زيادة نقدية على القرض"
    example = generator.generate_strict_example(arabic_claim, "ar", "True")
    
    print(f"Generated example: {example['id']}")
    print(f"Label: {example['label']}")
    print(f"Confidence: {example['confidence']}")
    print(f"Evidence file: {example['evidence']['file_path']}")
    print(f"Match type: {example['evidence']['match_type']}")
    print(f"Suspected fabrication: {example['suspected_fabrication']}")
    
    # Test batch generation
    print("\n2. Testing batch generation...")
    batch = generator.generate_batch(10, "ar")
    print(f"Generated {len(batch)} examples")
    
    # Validate batch
    print("\n3. Validating batch...")
    validation_result = generator.validate_no_hallucination(batch)
    print(f"Validation success: {validation_result['success']}")
    print(f"Fabrication rate: {validation_result['fabrication_rate']:.2%}")
    print(f"Needs manual review: {validation_result['needs_manual_review']}")
    
    if validation_result['validation_errors']:
        print("Validation errors:")
        for error in validation_result['validation_errors']:
            print(f"  - {error}")
    
    # Test prompts
    print("\n4. Testing prompt generation...")
    arabic_prompt = get_strict_prompt("ar", 20, 8, 8, 4)
    print(f"Arabic prompt length: {len(arabic_prompt)} chars")
    
    english_prompt = get_strict_prompt("en", 20, 8, 8, 4)
    print(f"English prompt length: {len(english_prompt)} chars")
    
    # Save sample batch
    output_file = "test_strict_batch.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in batch:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n5. Saved test batch to {output_file}")
    
    return validation_result['success']

if __name__ == "__main__":
    success = test_strict_generator()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
