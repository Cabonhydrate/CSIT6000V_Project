"""Unified response-language policy for paper QA, skills and external learning.

The saved profile is the default. An explicit instruction in the current user
question takes precedence; the language of the question alone does not.
"""
import re


def response_language(question: str, preferred: str = 'en') -> str:
    text = question or ''
    # Only unambiguous instructions override the saved choice.  A Chinese
    # question by itself is not a request for a Chinese answer.
    en = re.search(r'(?i)\b(?:answer|respond|write|explain|reply)\s+(?:me\s+)?(?:only\s+)?in\s+english\b|'
                   r'请用英文(?:回答|解释|说明)|用英语(?:回答|解释|说明)|(?:英文|英语)(?:回答|作答)', text)
    zh = re.search(r'(?i)\b(?:answer|respond|write|explain|reply)\s+(?:me\s+)?(?:only\s+)?in\s+(?:chinese|mandarin)\b|'
                   r'请用中文(?:回答|解释|说明)|用汉语(?:回答|解释|说明)|中文(?:回答|作答)', text)
    if en and not zh:
        return 'en'
    if zh and not en:
        return 'zh'
    return 'zh' if preferred == 'zh' else 'en'


def language_instruction(language: str) -> str:
    if language == 'zh':
        return ('OUTPUT LANGUAGE REQUIREMENT: Respond in Simplified Chinese. '
                'Use that language in every free-text JSON field and in visible headings. '
                'Preserve source quotes, mathematical expressions, citations and proper nouns.')
    return ('OUTPUT LANGUAGE REQUIREMENT: Respond entirely in English. '
            'All generated text, headings, examples, error messages and JSON free-text '
            'values must be in English, even if the question, earlier instructions, '
            'the user learning note or source excerpts are in Chinese. '
            'This overrides any earlier language choice in the template. '
            'Do not translate verbatim source quotations, mathematical notation, '
            'citation identifiers or proper nouns. Never expose your internal plan.')
