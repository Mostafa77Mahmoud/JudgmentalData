
#!/usr/bin/env python3

import sys
import json
import logging
from pathlib import Path
from src.data_processor import DataProcessor
from src.dataset_generator import DatasetGenerator

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        processor = DataProcessor()
        processor.load_data()
        
        generator = DatasetGenerator("config/keys.json")
        
        # Run Arabic smoke test
        logger.info("Running Arabic smoke test...")
        results = generator.run_smoke_test(language="ar", target_count=20)
        
        if results['success']:
            logger.info("✅ Smoke test completed successfully!")
            print(f"SMOKE TEST: generated {results['stats']['generated_count']}, local_verified {results['stats']['verified_local']}, model_verified {results['stats']['verified_model']}, fabrication_rate {results['stats']['fabrication_rate']:.1f}%")
            
            if results['stats']['failed_raw_paths']:
                print("Top 5 failing raw files:")
                for path in results['stats']['failed_raw_paths'][:5]:
                    print(f"  {path}")
            
            # Save failure summary to logs
            Path("logs").mkdir(exist_ok=True)
            failure_summary = {
                "fabrication_rate": results['stats']['fabrication_rate'],
                "total_candidates": results['stats']['generated_count'],
                "manual_review_count": len(results['stats']['failed_raw_paths']),
                "failing_raw_paths": results['stats']['failed_raw_paths'][:10]
            }
            
            with open("logs/smoke_failure_summary.json", "w", encoding="utf8") as f:
                json.dump(failure_summary, f, indent=2, ensure_ascii=False)
                
        else:
            logger.error(f"❌ Smoke test failed: {results.get('error')}")
            print(f"SMOKE TEST FAILED: {results.get('error')}")
            
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        print(f"CRITICAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
