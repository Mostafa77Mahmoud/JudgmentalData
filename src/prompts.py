# Strict verification prompts for Arabic and English
ARABIC_VERIFIER_PROMPT = """أنت "Verifier" دقيق جداً. لديك:
- CLAIM: "{claim}"
- CONTEXT: "{context}"

المطلوب: قرر ما إذا كان CLAIM:
  - "True" إذا يمكن إثباته حرفيًا أو باستنتاج منطقي مباشر ومستند إلى جملة/جزء واضح في CONTEXT.
  - "False" إذا النص يُنقض الادعاء صراحة.
  - "Unknown" إذا لا يوجد دليل كافٍ داخل CONTEXT.

قواعد صارمة:
1) **يُسمح فقط بالاستدلال على أساس ما هو داخل CONTEXT**. ممنوع جلب أو اختلاق مراجع خارجية.
2) إن ثبت الادعاء، يجب إظهار قائمة الأدلة evidence[] حيث كل عنصر يحتوي على:
   - chunk_id: {chunk_id}
   - start_char, end_char: إحداثيات substring داخل CONTEXT (0-indexed)
   - excerpt: النص المُقتبس حرفيًا (≤750 chars)
3) explanation: جملة موجزة ≤ 60 كلمة تشرح سبب القرار بالاعتماد على المحتوى المذكور فقط.
4) suspected_fabrication: boolean. **يجب أن يكون true** إذا لم تستطع دعم CLAIM بأي اقتباس/إثبات من CONTEXT أو إن الاستنتاج يتطلب معلومات خارجية.
5) confidence: عدد من 0.0 إلى 1.0
6) model_used: اسم النموذج المستخدم

أعد **JSON object only** بالهيكل التالي (لا كلام آخر):

{{
 "id":"{claim_id}",
 "verdict":"True|False|Unknown",
 "evidence":[{{"chunk_id":{chunk_id},"start_char":<int>,"end_char":<int>,"excerpt":"<≤750 chars>"}}],
 "explanation":"<≤60 words>",
 "confidence":0.0-1.0,
 "suspected_fabrication": true|false,
 "model_used":"gemini-2.5-flash"
}}"""

ENGLISH_VERIFIER_PROMPT = """You are a precise "Verifier". You have:
- CLAIM: "{claim}"
- CONTEXT: "{context}"

Required: Decide if CLAIM is:
  - "True" if it can be proven literally or by direct logical inference based on a clear sentence/part in CONTEXT.
  - "False" if the text explicitly contradicts the claim.
  - "Unknown" if there is insufficient evidence within CONTEXT.

Strict rules:
1) **Only inference based on what is inside CONTEXT is allowed**. No external references or fabrication allowed.
2) If claim is proven, you must show evidence[] list where each element contains:
   - chunk_id: {chunk_id}
   - start_char, end_char: substring coordinates within CONTEXT (0-indexed)
   - excerpt: literal quoted text (≤750 chars)
3) explanation: brief sentence ≤ 60 words explaining the decision based only on mentioned content.
4) suspected_fabrication: boolean. **Must be true** if you cannot support CLAIM with any quote/proof from CONTEXT or if inference requires external information.
5) confidence: number from 0.0 to 1.0
6) model_used: model name used

Return **JSON object only** with the following structure (no other text):

{{
 "id":"{claim_id}",
 "verdict":"True|False|Unknown",
 "evidence":[{{"chunk_id":{chunk_id},"start_char":<int>,"end_char":<int>,"excerpt":"<≤750 chars>"}}],
 "explanation":"<≤60 words>",
 "confidence":0.0-1.0,
 "suspected_fabrication": true|false,
 "model_used":"gemini-2.5-flash"
}}"""

ARABIC_GENERATOR_PROMPT = """انت مُحرر/مولد بيانات (Dataset Generator). عندك فقط هذا النص (CONTEXT) — لا يُسمح لك بالرجوع لأي مصدر خارجي ولا بصنع مراجع. المطلوب:
1) اقرأ CONTEXT بالتمام.
2) استخرج حتى 3 ادعاءات قصيرة (claims) قابلة للتمييز عن CONTEXT، كل ادعاء يجب أن يكون:
   - عبارة عن جملة واحدة قصيرة (≤30 كلمة).
   - مشتقة مباشرة من CONTEXT (لا إضافات فكرية من عندك).
3) لكل ادعاء أدرج حقل evidence واحد على الأقل مأخوذ حرفياً من CONTEXT:
   - evidence.excerpt يجب أن يكون substring موجود حرفياً في CONTEXT وبحد أقصى 750 حرف.
   - evidence.start_char و evidence.end_char (مواقع في CONTEXT، 0-index).
4) أعد نتيجة بصيغة JSON array فقط، الشكل لكل عنصر:
{{
 "id":"{uuid}",
 "language":"ar", 
 "claim":"<short claim>",
 "evidence":[{{"chunk_id":{chunk_id},"start_char":<int>,"end_char":<int>,"excerpt":"<≤750 chars>"}}],
 "meta":{{"generator_stage":"generation_only"}}
}}
5) إن لم تستطع استخراج أي ادعاء مباشر من النص، ارجع [] (مصفوفة فارغة).
6) **صريح**: لا تضيف تفسير أو استدلال. لا تُخْرِج نص خارجي أو تعليقات. فقط JSON.

CONTEXT:
{context}"""

ENGLISH_GENERATOR_PROMPT = """You are a Dataset Generator. You have only this text (CONTEXT) — you are NOT allowed to reference any external sources or create references. Required:
1) Read CONTEXT completely.
2) Extract up to 3 short claims that can be distinguished from CONTEXT, each claim must be:
   - A single short sentence (≤30 words).
   - Directly derived from CONTEXT (no intellectual additions from you).
3) For each claim include at least one evidence field taken literally from CONTEXT:
   - evidence.excerpt must be a substring that exists literally in CONTEXT and max 750 characters.
   - evidence.start_char and evidence.end_char (positions in CONTEXT, 0-indexed).
4) Return result as JSON array only, format for each element:
{{
 "id":"{uuid}",
 "language":"en",
 "claim":"<short claim>",
 "evidence":[{{"chunk_id":{chunk_id},"start_char":<int>,"end_char":<int>,"excerpt":"<≤750 chars>"}}],
 "meta":{{"generator_stage":"generation_only"}}
}}
5) If you cannot extract any direct claim from the text, return [] (empty array).
6) **Explicit**: Do not add interpretation or reasoning. Do not output external text or comments. Only JSON.

CONTEXT:
{context}"""