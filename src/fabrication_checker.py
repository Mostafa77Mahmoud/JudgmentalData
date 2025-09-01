# src/fabrication_checker.py
import json
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class FabricationChecker:
    """Post-processing checker to monitor and reduce fabrication rates.

    Safer default: do not flip verdicts automatically; instead add flags for manual review.
    """

    def __init__(
        self,
        max_fabrication_rate: float = 0.10,
        verification_confidence_threshold: float = 0.6,
        min_excerpt_len: int = 10,
        enforce_strict: bool = False,
    ):
        """
        enforce_strict: if True, will mark examples as fabricated and set verdict="Unknown"
                        when evidence quality fails. If False, only annotate with
                        needs_manual_review and fabrication_reasons (monitoring mode).
        """
        self.max_fabrication_rate = max_fabrication_rate
        self.verification_confidence_threshold = verification_confidence_threshold
        self.min_excerpt_len = min_excerpt_len
        self.enforce_strict = enforce_strict

    def compute_fabrication_rate(
            self, results: List[Dict]) -> Tuple[float, int, int]:
        total = len(results)
        if total == 0:
            return 0.0, 0, 0
        fabricated = sum(1 for r in results
                         if r.get("suspected_fabrication", False))
        rate = fabricated / total
        logger.info(f"Fabrication analysis: {fabricated}/{total} = {rate:.2%}")
        return rate, fabricated, total

    def _normalize(self, s: str) -> str:
        # light normalization for comparisons
        return " ".join(s.split()).strip()

    def check_evidence_quality(
            self,
            result: Dict,
            context_text: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Returns (ok:bool, reasons:List[str])
        If context_text provided, also validate start/end indices and excerpt matching.
        """
        reasons = []
        evidence = result.get("evidence", [])
        if not evidence:
            reasons.append("no_evidence")
            return False, reasons

        for i, ev in enumerate(evidence):
            excerpt = ev.get("excerpt", "") or ""
            start_char = ev.get("start_char")
            end_char = ev.get("end_char")

            if len(self._normalize(excerpt)) < self.min_excerpt_len:
                reasons.append(f"excerpt_too_short[{i}]")
                continue

            if start_char is None or end_char is None:
                reasons.append(f"missing_indices[{i}]")
                continue

            if not (isinstance(start_char, int) and isinstance(end_char, int)):
                reasons.append(f"bad_index_type[{i}]")
                continue

            if start_char < 0 or end_char <= start_char:
                reasons.append(f"bad_index_range[{i}]")
                continue

            if context_text is not None:
                if end_char > len(context_text):
                    reasons.append(f"index_out_of_bounds[{i}]")
                    continue
                # compare normalized excerpt with substring
                substr = self._normalize(context_text[start_char:end_char])
                if substr != self._normalize(excerpt):
                    reasons.append(f"excerpt_mismatch[{i}]")
                    continue

        ok = len(reasons) == 0
        return ok, reasons

    def post_process_results(
            self,
            results: List[Dict],
            context_map: Optional[Dict[int, str]] = None) -> List[Dict]:
        """
        Process results. If context_map provided, it should map chunk_id -> full chunk text
        for index-based validation. This function will annotate results with:
          - fabrication_reasons: list of strings (why flagged)
          - needs_manual_review: bool
          - (optionally) suspected_fabrication and changed verdict/confidence if enforce_strict True
        """
        processed_results = []

        for result in results:
            reasons = []
            # prefer explicit verification_confidence if available
            verification_conf = result.get("meta",
                                           {}).get("verification_confidence")
            if verification_conf is not None and isinstance(
                    verification_conf, (int, float)):
                if verification_conf < self.verification_confidence_threshold:
                    reasons.append("low_verification_confidence")

            # evidence quality check (with context if available)
            chunk_id = result.get("context_chunk_id")
            context_text = None
            if context_map and chunk_id is not None:
                context_text = context_map.get(chunk_id)

            ok, ev_reasons = self.check_evidence_quality(
                result, context_text=context_text)
            if not ok:
                reasons.extend(ev_reasons)

            # Extra: if generator indicated finish_reason issues or raw_response truncated -> flag
            gen_meta = result.get("meta", {})
            if gen_meta.get("finish_reason") == "MAX_TOKENS":
                reasons.append("truncated_by_max_tokens")
            if not result.get("raw_response_path"):
                reasons.append("no_raw_response_saved")

            # decide action
            if reasons:
                # annotate for review
                result["fabrication_reasons"] = reasons
                result["needs_manual_review"] = True

                if self.enforce_strict:
                    # aggressive: mark as fabricated
                    result["suspected_fabrication"] = True
                    result["verdict"] = "Unknown"
                    result["confidence"] = 0.0
                    logger.warning(
                        f"Marked as fabricated due to: {reasons} id={result.get('id')}"
                    )
                else:
                    # monitoring mode: do not change verdict, just log
                    logger.warning(
                        f"Needs manual review (not auto-fabricated): {reasons} id={result.get('id')}"
                    )
            else:
                # clear/ensure flags show OK
                result.pop("fabrication_reasons", None)
                result["needs_manual_review"] = False
                # ensure suspected_fabrication false unless already set intentionally
                if "suspected_fabrication" in result and result[
                        "suspected_fabrication"]:
                    # keep existing positive marks (if any) but log
                    logger.info(
                        f"Kept existing suspected_fabrication for id={result.get('id')}"
                    )

            processed_results.append(result)

        # compute overall fabrication rate (based on suspected_fabrication flag)
        fab_rate, fab_count, total = self.compute_fabrication_rate(
            processed_results)
        if fab_rate > self.max_fabrication_rate:
            logger.warning(
                f"High fabrication rate detected: {fab_rate:.2%} > {self.max_fabrication_rate:.2%}"
            )
            logger.warning(
                "Recommendations: reduce temperature, fewer claims per chunk, stricter prompts, increase max_output_tokens, ensure raw responses are saved"
            )

        return processed_results

    def generate_quality_report(self, results: List[Dict]) -> Dict:
        fab_rate, fab_count, total = self.compute_fabrication_rate(results)

        verdicts = {}
        for result in results:
            verdict = result.get("verdict", "Unknown")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1

        avg_confidence = sum(r.get("confidence", 0.0)
                             for r in results) / max(len(results), 1)

        needs_review = sum(1 for r in results if r.get("needs_manual_review"))
        return {
            "total_examples": total,
            "fabrication_rate": fab_rate,
            "fabricated_count": fab_count,
            "verdict_distribution": verdicts,
            "average_confidence": avg_confidence,
            "needs_manual_review": needs_review,
            "quality_passed": fab_rate <= self.max_fabrication_rate
        }
