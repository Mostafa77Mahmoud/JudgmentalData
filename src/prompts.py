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

ARABIC_GENERATOR_PROMPT = """أنت مولد بيانات حتمي. الإخراج يجب أن يكون JSON فقط.

النص المرجعي: {context}

التعليمات:
- أنشئ ادعاء واحد فقط مشتق *مباشرة* من النص المرجعي
- لكل ادعاء أدرج دليل واحد على الأقل يحتوي على:
  - "start_char" و "end_char" (مواقع في النص، بدءاً من 0)
  - "excerpt": substring دقيق من النص[start_char:end_char]، بحد أقصى 750 حرف
- الادعاء يجب أن يكون ≤30 كلمة
- الإخراج يجب أن يكون JSON object واحد بالضبط كما هو موضح أدناه:

{{
 "id":"{uuid}",
 "language":"ar",
 "claim":"<...>",
 "context_chunk_id": {chunk_id},
 "evidence":[{{"chunk_id":{chunk_id},"start_char":<int>,"end_char":<int>,"excerpt":"<...>"}}],
 "generator_model":"gemini-2.5-flash"
}}

إذا لم تستطع إنتاج ادعاء بناءً بصرامة على النص المرجعي، أخرج: {{}}ندك).
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

ENGLISH_GENERATOR_PROMPT = """You are a deterministic data generator. OUTPUT MUST BE JSON ONLY.

Context: {context}

Instructions:
- Generate exactly 1 claim derived *directly* from CONTEXT
- For that claim include at least one evidence object with:
  - "start_char" and "end_char" (0-indexed positions in CONTEXT)
  - "excerpt": exact substring CONTEXT[start_char:end_char], max 750 chars
- Claim must be ≤30 words
- Output MUST be a single JSON object exactly as below:

{{
 "id":"{uuid}",
 "language":"en",
 "claim":"<...>",
 "context_chunk_id": {chunk_id},
 "evidence":[{{"chunk_id":{chunk_id},"start_char":<int>,"end_char":<int>,"excerpt":"<...>"}}],
 "generator_model":"gemini-2.5-flash"
}}

If you cannot produce such claim based strictly on CONTEXT, output: {{}}ONTEXT, each claim must be:
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