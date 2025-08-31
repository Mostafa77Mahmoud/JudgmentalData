
# Strict verification prompts for Arabic and English
ARABIC_VERIFIER_PROMPT = '''
تحقق من الادعاءات التالية اعتمادًا على المقتطفات المرفقة فقط.
أعد فقط JSON صالح (بدون أي نص إضافي). كل عنصر يجب أن يحتوي:
id، verdict (True/False/Unknown)، explanation (≤ 200 حرف، موجز جدًا)، reference.
لا تُضِف مقدمات أو تعليقات.
'''

ENGLISH_VERIFIER_PROMPT = '''
Verify the following claims strictly using the provided excerpts only.
Return ONLY valid JSON (no extra text). Each item must have:
id, verdict (True/False/Unknown), explanation (≤ 200 chars, very concise), reference.
No preface or commentary.
'''
