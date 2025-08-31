
# Strict prompts for no-hallucination data generation

ARABIC_STRICT_PROMPT = """
أنت مساعد توليد بيانات حذر. استخدم فقط الملفات الموجودة في مجلد "attached_assets" كدليل. لا تستخدم أو تخترع أي مصادر خارجية. يجب أن يكون الناتج JSON صالح (UTF-8) فقط. لا نص إضافي.

المهمة: أنتج مصفوفة JSON تحتوي على {BATCH_SIZE} عنصر بالضبط. كل عنصر يجب أن يتبع المخطط أدناه. لكل عنصر، ابحث عن دليل حقيقي داخل ملفات attached_assets. إذا لم تستطع العثور على أي دليل داعم داخل attached_assets للادعاء، فضع التصنيف "Unknown" واجعل "suspected_fabrication": true و "needs_manual_review": true. لا تخترع أي مراجع أو اقتباسات.

مخطط JSON المطلوب لكل عنصر:
{{
 "id": "<uuid>",
 "language": "ar",
 "claim": "<ادعاء عربي: جملة أو جملتان>",
 "label": "True|False|Unknown",
 "explanation": "<عربي، حد أقصى 60 كلمة> (اذكر بإيجاز سبب كون الادعاء True/False/Unknown؛ إذا Unknown، قل 'لا يوجد دليل في attached_assets')",
 "confidence": "<رقم عشري بين 0.0 و 1.0>",
 "evidence": {{
   "file_path": "attached_assets/<اسم الملف بالضبط>",
   "excerpt": "<الاقتباس الدقيق أو إعادة الصياغة من ذلك الملف، حد أقصى 750 حرف>",
   "start_char": <رقم صحيح>,
   "end_char": <رقم صحيح>,
   "match_type": "exact|paraphrase|inferred"
 }},
 "reference": "<اقتباس مختصر: اسم الملف والقسم أو الصفحة>",
 "generator_meta": {{"generator_model":"<string>","prompt_version":"v1","seed_id":"<seed>"}},
 "raw_response_path": "",
 "suspected_fabrication": false,
 "needs_manual_review": false
}}

قواعد التوليد:
- بالضبط {N_TRUE} عنصر يجب أن يكون مصنف True (خليط من exact و paraphrase).
- بالضبط {N_FALSE} عنصر يجب أن يكون مصنف False (خليط من out_of_context و fabricated).
- بالضبط {N_UNKNOWN} عنصر يجب أن يكون مصنف Unknown.
- وفر مستويات صعوبة متوازنة: تقريباً 30% سهل (تطابق دقيق)، 40% متوسط (إعادة صياغة/دلالي)، 30% صعب (خفي/مستنتج).
- لكل دليل، ضع مواضع start_char و end_char دقيقة نسبة لمحتويات الملف.
- excerpt يجب أن يكون حرفياً من الملف لـ match_type "exact". لـ "paraphrase" ضع إعادة صياغة قصيرة واضبط match_type وفقاً لذلك.
- اجعل كل explanation موجز وبالعربية. لا تعليق إضافي.

الناتج: مصفوفة JSON واحدة فقط. لا نص إضافي.
"""

ENGLISH_STRICT_PROMPT = """
You are a careful data-generation assistant. Only use files under "attached_assets" as evidence. Do NOT invent external sources. Output must be valid JSON only.

Task: Produce a JSON array of exactly {BATCH_SIZE} items. Each item must follow the schema below. If you cannot find supporting evidence inside attached_assets for a claim, label it "Unknown" and set "suspected_fabrication": true and "needs_manual_review": true. Do NOT fabricate references or quotes.

JSON schema (strict) required per item:
{{
 "id": "<uuid>",
 "language": "en",
 "claim": "<English claim: 1-2 sentences>",
 "label": "True|False|Unknown",
 "explanation": "<English, max 60 words> (state concisely why the claim is True/False/Unknown; if Unknown, say 'No evidence found in attached_assets')",
 "confidence": "<float between 0.0 and 1.0>",
 "evidence": {{
   "file_path": "attached_assets/<exact file name>",
   "excerpt": "<the exact excerpt or paraphrase from that file, max 750 chars>",
   "start_char": <integer>,
   "end_char": <integer>,
   "match_type": "exact|paraphrase|inferred"
 }},
 "reference": "<short citation: file name and section or page>",
 "generator_meta": {{"generator_model":"<string>","prompt_version":"v1","seed_id":"<seed>"}},
 "raw_response_path": "",
 "suspected_fabrication": false,
 "needs_manual_review": false
}}

Generation rules:
- {N_TRUE} True items, {N_FALSE} False items, {N_UNKNOWN} Unknown items.
- Balanced difficulty: ~30% easy (exact), 40% medium (paraphrase/semantic), 30% hard (inference).
- evidence.excerpt ≤ 750 chars; for exact matches it must be verbatim.
- Include start_char/end_char (character offsets in the referenced file).
- Keep explanation concise (≤ 60 words).
- Return one JSON array only.
"""

def get_strict_prompt(language: str, batch_size: int, n_true: int, n_false: int, n_unknown: int) -> str:
    """Get formatted strict prompt for specified language and parameters"""
    if language == "ar":
        return ARABIC_STRICT_PROMPT.format(
            BATCH_SIZE=batch_size,
            N_TRUE=n_true,
            N_FALSE=n_false,
            N_UNKNOWN=n_unknown
        )
    else:
        return ENGLISH_STRICT_PROMPT.format(
            BATCH_SIZE=batch_size,
            N_TRUE=n_true,
            N_FALSE=n_false,
            N_UNKNOWN=n_unknown
        )
def get_strict_arabic_prompt() -> str:
    """Strict Arabic verification prompt"""
    return """أنت نظام تحقق صارم. يجب أن تستخدم فقط المحتوى المقدم في مجلد "attached_assets" (المستودع المحلي). لا تخترع حقائق، لا تهلوس مصادر، ولا تستخدم معرفة خارجية.

المدخل: مصفوفة JSON من العناصر. كل عنصر: { "id", "claim", "context_excerpt" } — context_excerpt من attached_assets.

المهمة: لكل عنصر أرجع كائن JSON بهذه الحقول بالضبط:
- id (نفسه)
- verdict: واحد من ["True","False","Unknown"]
- explanation: شرح عربي قصير، حد أقصى 60 كلمة، موجز وواقعي
- evidence: إما null أو كائن { "file_path": "<مسار نسبي في attached_assets>", "chunk_id": "<id>", "start_char": int, "end_char": int }
- confidence: رقم عشري بين 0.0 و 1.0
- suspected_fabrication: منطقي

القواعد:
1) إذا لم تستطع إثبات وجود نص داعم أو معارض داخل attached_assets، ضع verdict="Unknown", suspected_fabrication=true, evidence=null
2) استخدم فقط النص الموجود بالضبط في attached_assets للأدلة؛ evidence.start_char/end_char يجب أن يشير للنص الفرعي الدقيق في الملف المرجعي
3) أخرج فقط مصفوفة JSON — بدون تعليقات إضافية، بدون markdown
4) كن حتمياً: temperature=0, top_p=1.0
5) اجعل كل explanation ≤ 60 كلمة

أرجع: مصفوفة JSON من الكائنات كما هو محدد.

استخدم الملفات في attached_assets/ (arabic_chunks.json, english_chunks.json, arabic_cleaned.txt, english_cleaned.txt). يجب أن تُرجع file_path بالضبط كما هو في attached_assets."""

def get_strict_english_prompt() -> str:
    """Strict English verification prompt"""
    return """You are a deterministic verifier. You MUST use only the content provided in the "attached_assets" folder (local repository). Do NOT invent facts, do NOT hallucinate sources, and do NOT use outside knowledge.

Input: a JSON array of items. Each item: { "id", "claim", "context_excerpt" } — the context_excerpt is from the attached_assets.

Task: For each item return a JSON object with exactly these fields:
- id (same)
- verdict: one of ["True","False","Unknown"]
- explanation: short English explanation, max 60 words, concise and factual
- evidence: either null or an object { "file_path": "<relative path in attached_assets>", "chunk_id": "<id>", "start_char": int, "end_char": int }
- confidence: float between 0.0 and 1.0
- suspected_fabrication: boolean

Rules:
1) If you cannot demonstrably find supporting or contradicting text inside attached_assets, set verdict="Unknown", suspected_fabrication=true, evidence=null
2) Only use text exactly present in attached_assets for evidence; evidence.start_char/end_char must point to the exact substring in the referenced file
3) Output ONLY a JSON array — no extra commentary, no markdown
4) Be deterministic: temperature=0, top_p=1.0
5) Keep each explanation <= 60 words

Return: a JSON array of objects as specified.

Use the files in attached_assets/ (arabic_chunks.json, english_chunks.json, arabic_cleaned.txt, english_cleaned.txt). You MUST reference file_path exactly as in attached_assets."""

def get_verification_schema() -> dict:
    """JSON schema for verification output"""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "verdict": {"enum": ["True", "False", "Unknown"]},
                "explanation": {"type": "string"},
                "evidence": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "chunk_id": {"type": "string"},
                                "start_char": {"type": "integer"},
                                "end_char": {"type": "integer"},
                                "match_type": {"enum": ["exact", "partial", "paraphrase", "inferred"]}
                            },
                            "required": ["file_path", "chunk_id", "start_char", "end_char"],
                            "additionalProperties": False
                        }
                    ]
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "suspected_fabrication": {"type": "boolean"}
            },
            "required": ["id", "verdict", "explanation", "evidence", "confidence", "suspected_fabrication"],
            "additionalProperties": False
        }
    }
