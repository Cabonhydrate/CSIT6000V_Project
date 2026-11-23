"""Routing and conservative rendering helpers for paper + external concept comparisons.

No domain-specific definitions or Gold answers live in this module. A comparison
requires a named paper acronym, a distinct external concept, and an explicit
similarity cue. The two sources remain separate throughout the answer.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from ..agent.background_learning import search_topic
from .definition_grounding import definition_subject

_SIMILAR = re.compile(r'类似|相似|相像|相近|有点像|有些像|很像|共同点|共通之处|'
                      r'(?i:similar\s+to|comparable\s+to|resembles?)|一样吗|是否一样|是不是一样')


def implicit_background_comparison(question: str) -> tuple[str, str] | None:
    """Recognize a similarity *statement*, not an ordinary concept explanation.

    The paper-side acronym is extracted from the user's own question and then
    checked against the PDF by the workflow. The outside subject comes from the
    same words the user typed and is independently verified by web retrieval.
    """
    if not _SIMILAR.search(question or ''):
        return None
    paper_term = definition_subject(question)
    external_term = search_topic(question)
    # A Chinese concept name after a connector is equally explicit. Avoid
    # choosing the only English token (the paper acronym) as the web topic.
    if paper_term and external_term.casefold() == paper_term.casefold():
        chinese = re.search(
            r'(?:和|与|跟|同)\s*([\u4e00-\u9fff]{2,24}?)\s*'
            r'(?=有点像|很像|有些类似|有点类似|类似|相似|相近|一样吗|是一样|有些相似)',
            question)
        if chinese:
            external_term = chinese.group(1)
    if not paper_term or not external_term:
        return None
    paper_norm = unicodedata.normalize('NFKC', paper_term).casefold().strip()
    external_norm = unicodedata.normalize('NFKC', external_term).casefold().strip()
    if paper_norm == external_norm or len(external_norm) < 4:
        return None
    # Do not turn a single-concept illustration ("举一个类似例子") into a
    # cross-source comparison. Require two distinct explicit named terms.
    english_tokens = re.findall(r'(?<![A-Za-z])[A-Za-z][A-Za-z-]*(?: +[A-Za-z][A-Za-z-]*)*(?![A-Za-z])', question)
    if not any(paper_norm == token.casefold().strip() for token in english_tokens):
        return None
    if external_norm not in question.casefold():
        return None
    return paper_term, external_term


def external_source_matches(source: dict, term: str) -> bool:
    """Do not use a search hit just because an unrelated result has a URL."""
    parsed = urlparse(str(source.get('source_url') or ''))
    if parsed.scheme != 'https' or (parsed.hostname or '').lower() not in (
            'en.wikipedia.org', 'zh.wikipedia.org', 'doi.org'):
        return False
    title = str(source.get('title') or source.get('section') or '').casefold()
    body = str(source.get('content') or '')[:350].casefold()
    tokens = re.findall(r'[a-z][a-z0-9-]+', term.casefold())
    if tokens:
        return all(x in title for x in tokens) or all(x in body for x in tokens)
    chinese = re.sub(r'\s+', '', term.casefold())
    return bool(re.search(r'[\u4e00-\u9fff]', chinese)) and (chinese in title or chinese in body)


def validate_comparison_fields(payload: object) -> tuple[dict[str, str], str]:
    """Validate the model's comparison structure, not its factual correctness.

    Source-side definition checking is performed separately in the workflow;
    source IDs are never accepted from the model for this workflow.
    """
    if not isinstance(payload, dict):
        return {}, '模型没有返回比较结果对象'
    fields = {}
    for key in ('paper', 'external', 'similarity', 'difference'):
        value = payload.get(key)
        if not isinstance(value, str):
            return {}, '缺少比较字段：' + key
        value = re.sub(r'\[\d+\]', '', value).strip()
        if len(value) < 8 or len(value) > 450:
            return {}, '比较字段不完整：' + key
        fields[key] = value
    if not re.search(r'不同|区别|前者|后者|而|但是|相比|versus|whereas|while',
                     fields['difference'], re.I):
        return {}, '缺少两者区别的明确陈述'
    if any(re.search(r'^\s*(?:首先.*用户|我需要|根据提示|my task is)', v, re.I)
           for v in fields.values()):
        return {}, '答案包含内部分析文本'
    return fields, ''
