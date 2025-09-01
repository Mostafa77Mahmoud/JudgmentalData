
def post_check_item(item, context_text):
    """
    Strict post-check validation for generated items
    Returns (is_valid, reason)
    """
    try:
        evidences = item.get("evidence", [])
        
        if not evidences:
            return False, "no_evidence"
            
        for ev in evidences:
            # Check if indices exist and are integers
            start_char = ev.get("start_char")
            end_char = ev.get("end_char")
            
            if not (isinstance(start_char, int) and isinstance(end_char, int)):
                return False, "bad_indices_type"
                
            # Check if indices are within bounds
            if not (0 <= start_char < end_char <= len(context_text)):
                return False, "bad_indices_range"
                
            # Extract actual substring and compare
            actual_excerpt = context_text[start_char:end_char]
            claimed_excerpt = ev.get("excerpt", "")
            
            if actual_excerpt != claimed_excerpt:
                return False, "excerpt_mismatch"
                
            # Check excerpt length
            if len(actual_excerpt) > 750:
                return False, "excerpt_too_long"
                
        # All checks passed
        return True, "ok"
        
    except Exception as e:
        return False, f"check_error: {str(e)}"


def mark_fabrication_if_invalid(item, context_text):
    """
    Apply post-check and mark as fabrication if validation fails
    """
    is_valid, reason = post_check_item(item, context_text)
    
    if not is_valid:
        item["suspected_fabrication"] = True
        item["verdict"] = "Unknown"
        item["explanation"] = f"Validation failed: {reason}"
        item["meta"] = item.get("meta", {})
        item["meta"]["validation_failure"] = reason
        
    return item
