
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
