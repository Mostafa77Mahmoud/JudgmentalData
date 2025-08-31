
#!/usr/bin/env python3
"""
Test script for the strict dataset generator
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.strict_local_pipeline import StrictLocalPipeline
from src.strict_prompts import get_strict_arabic_prompt, get_strict_english_prompt

def test_strict_pipeline():
    """Test the strict local pipeline"""
    print("Testing Strict Local Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = StrictLocalPipeline()
        
        # Test Arabic smoke test
        print("\n1. Testing Arabic smoke test...")
        ar_results = pipeline.run_smoke_test("ar", 10)
        
        print(f"Arabic Results:")
        print(f"  Success: {ar_results['success']}")
        print(f"  Total generated: {ar_results.get('total_generated', 0)}")
        print(f"  Stats: {ar_results.get('stats', {})}")
        
        if ar_results.get('samples'):
            print(f"  Sample example:")
            sample = ar_results['samples'][0]
            print(f"    Claim: {sample['claim'][:100]}...")
            print(f"    Label: {sample['label']}")
            print(f"    Confidence: {sample['confidence']}")
            print(f"    Suspected fabrication: {sample['suspected_fabrication']}")
        
        # Test English smoke test
        print("\n2. Testing English smoke test...")
        en_results = pipeline.run_smoke_test("en", 10)
        
        print(f"English Results:")
        print(f"  Success: {en_results['success']}")
        print(f"  Total generated: {en_results.get('total_generated', 0)}")
        print(f"  Stats: {en_results.get('stats', {})}")
        
        # Test prompts
        print("\n3. Testing prompts...")
        ar_prompt = get_strict_arabic_prompt()
        en_prompt = get_strict_english_prompt()
        
        print(f"Arabic prompt length: {len(ar_prompt)} chars")
        print(f"English prompt length: {len(en_prompt)} chars")
        
        print(f"Arabic prompt contains 'attached_assets': {'attached_assets' in ar_prompt}")
        print(f"English prompt contains 'attached_assets': {'attached_assets' in en_prompt}")
        
        # Overall assessment
        overall_success = ar_results['success'] and en_results['success']
        print(f"\n=== OVERALL TEST RESULT: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_verification():
    """Test individual claim verification"""
    print("\n4. Testing individual verification...")
    
    try:
        pipeline = StrictLocalPipeline()
        
        # Test Arabic claim
        ar_claim = "لا يجوز اشتراط زيادة نقدية على القرض"
        if pipeline.arabic_chunks:
            from src.verify_local import verify_claim_locally
            result = verify_claim_locally(ar_claim, pipeline.arabic_chunks[0], "attached_assets/arabic_chunks.json")
            print(f"Arabic verification result: {result['verdict']} (confidence: {result['confidence']})")
        
        # Test English claim
        en_claim = "Interest-based loans are prohibited in Islamic finance"
        if pipeline.english_chunks:
            from src.verify_local import verify_claim_locally
            result = verify_claim_locally(en_claim, pipeline.english_chunks[0], "attached_assets/english_chunks.json")
            print(f"English verification result: {result['verdict']} (confidence: {result['confidence']})")
        
    except Exception as e:
        print(f"Individual verification test failed: {e}")

if __name__ == "__main__":
    print("Starting Strict Generator Tests...")
    
    success = test_strict_pipeline()
    test_individual_verification()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
