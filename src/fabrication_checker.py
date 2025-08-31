
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class FabricationChecker:
    """Post-processing checker to monitor and reduce fabrication rates"""
    
    def __init__(self, max_fabrication_rate: float = 0.10):
        self.max_fabrication_rate = max_fabrication_rate
    
    def compute_fabrication_rate(self, results: List[Dict]) -> Tuple[float, int, int]:
        """Compute fabrication rate from verification results"""
        total = len(results)
        if total == 0:
            return 0.0, 0, 0
            
        fabricated = sum(1 for r in results if r.get("suspected_fabrication", False))
        rate = fabricated / total
        
        logger.info(f"Fabrication analysis: {fabricated}/{total} = {rate:.2%}")
        return rate, fabricated, total
    
    def check_evidence_quality(self, result: Dict) -> bool:
        """Check if evidence meets minimum quality standards"""
        evidence = result.get("evidence", [])
        
        if not evidence:
            return False
            
        for ev in evidence:
            excerpt = ev.get("excerpt", "")
            start_char = ev.get("start_char")
            end_char = ev.get("end_char")
            
            # Minimum excerpt length
            if len(excerpt.strip()) < 10:
                return False
                
            # Valid character positions
            if start_char is None or end_char is None or start_char >= end_char:
                return False
                
        return True
    
    def post_process_results(self, results: List[Dict]) -> List[Dict]:
        """Apply post-processing rules to reduce fabrication"""
        processed_results = []
        
        for result in results:
            # Check evidence quality
            if not self.check_evidence_quality(result):
                result["suspected_fabrication"] = True
                result["verdict"] = "Unknown"
                result["confidence"] = 0.0
                logger.warning(f"Marked as fabricated due to poor evidence: {result.get('id')}")
            
            # Mark empty evidence as fabrication
            if not result.get("evidence"):
                result["suspected_fabrication"] = True
                result["verdict"] = "Unknown"
                
            processed_results.append(result)
        
        # Check overall fabrication rate
        fab_rate, fab_count, total = self.compute_fabrication_rate(processed_results)
        
        if fab_rate > self.max_fabrication_rate:
            logger.warning(f"High fabrication rate detected: {fab_rate:.2%} > {self.max_fabrication_rate:.2%}")
            logger.warning("Consider: reducing temperature, fewer claims per chunk, stricter prompts")
        
        return processed_results
    
    def generate_quality_report(self, results: List[Dict]) -> Dict:
        """Generate quality metrics report"""
        fab_rate, fab_count, total = self.compute_fabrication_rate(results)
        
        verdicts = {}
        for result in results:
            verdict = result.get("verdict", "Unknown")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        avg_confidence = sum(r.get("confidence", 0.0) for r in results) / max(len(results), 1)
        
        return {
            "total_examples": total,
            "fabrication_rate": fab_rate,
            "fabricated_count": fab_count,
            "verdict_distribution": verdicts,
            "average_confidence": avg_confidence,
            "quality_passed": fab_rate <= self.max_fabrication_rate
        }
