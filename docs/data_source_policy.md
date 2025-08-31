
# Data Source Policy

## Canonical Data Sources

### Text Chunks
- **Primary**: `inputs/arabic_chunks.json` - Pre-chunked content with token budgets and preserved chunk IDs
- **Fallback**: `inputs/arabic_cleaned.txt` - Used only if chunks file is missing, will be auto-chunked

### QA Pairs
- **Purpose**: `inputs/arabic_qa_pairs (2000).json` should be used ONLY for QA training tasks
- **Restriction**: Do NOT use QA pairs as primary evidence for judgmental labels

## Dataset Strategy

### Judgmental Dataset Construction
- Build from: claims + matching context chunks
- Each JSONL judgment example must contain: claim + context excerpt + verdict + reference
- Prefer deterministic perturbations for negative (False) examples over model-generated false examples
- This improves label quality and reduces fabrication risk

### Generation Process
1. Extract claims from QA seeds
2. Match claims to relevant context chunks
3. Apply local verification for exact/high-overlap matches
4. Send ambiguous cases to model verification in batches
5. Validate schema and save to JSONL format

This policy ensures consistent data flow and improves the reliability of the judgmental dataset.
