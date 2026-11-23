"""Routing and conservative rendering helpers for paper + external concept comparisons.

No domain-specific definitions or Gold answers live in this module. A comparison
requires a named paper acronym, a distinct external concept, and an explicit
similarity cue. The two sources remain separate throughout the answer.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from ..agent.background_learning import search_topic, source_title_relevance
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
    return source_title_relevance(source, term) >= .76



def source_acronym_expansion(acronym: str, paper_evidence: list[dict]) -> str:
    """Read a long-form acronym expansion literally from paper text.

    Only accept the suffix whose initial letters spell the user's acronym;
    never infer a name from a term's meaning or the LLM's reply.
    """
    if not re.fullmatch(r'[A-Z]{2,8}', acronym or ''):
        return ''
    pat = re.compile(r'(?<![A-Za-z])([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){1,9})'
                     r'\s*\(\s*' + re.escape(acronym) + r'\s*\)', re.I)
    for ev in paper_evidence:
        text = str(ev.get('content') or '')
        for match in pat.finditer(text):
            words = match.group(1).split()
            if len(words) < len(acronym):
                continue
            ending = words[-len(acronym):]
            initials = ''.join(x[0].upper() for x in ending)
            if initials == acronym:
                return ' '.join(ending)
    return ''


def validate_comparison_grounding(fields: dict[str, str], *, paper_term: str,
                                  paper_sources: list[dict], paper_excerpt: str,
                                  external_excerpt: str) -> str:
    """Catch high-confidence contradictions before marking a comparison done.

    This deliberately does not claim full natural-language entailment. It
    checks a literal acronym expansion and the concrete category error where
    a retrieval metric and a code-execution metric are both called code-based.
    """
    expansion = source_acronym_expansion(paper_term, paper_sources)
    if expansion:
        candidates = [
            re.escape(paper_term) + r'\s*\(\s*([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,7})\s*\)',
            r'([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,7})\s*\(\s*' + re.escape(paper_term) + r'\s*\)',
        ]
        for pattern in candidates:
            for match in re.finditer(pattern, unicodedata.normalize('NFKC', fields.get('paper', '')), re.I):
                phrase = match.group(1).strip()
                # The long form is no longer than the acronym itself in word
                # count; avoid inadvertently matching a preceding sentence.
                words = phrase.split()
                alleged = ' '.join(words[-len(paper_term):])
                if alleged.casefold() != expansion.casefold():
                    return f'论文原文把 {paper_term} 展开为 {expansion}，不是 {alleged}'
    similarity = unicodedata.normalize('NFKC', fields.get('similarity', '')).casefold()
    paper = paper_excerpt.casefold()
    external = external_excerpt.casefold()
    paper_is_retrieval = bool(re.search(r'(?i)retriev|corpus.{0,20}(chunk|record)', paper))
    external_is_code = bool(re.search(r'(?i)source code|code coverage|program.{0,30}execut', external))
    if paper_is_retrieval and external_is_code:
        # Evaluate each claim clause separately: a legitimate comparison may
        # mention code execution in the DIFFERENCE, but must never attribute
        # that property to both metrics as a similarity. The old regex only
        # matched "源代码" / "程序代码" and missed "代码执行情况".
        shared = r'(?:两者|二者|它们|双方|均|都|共同|both|each|all)'
        code = r'(?:程序代码|源代码|代码行|代码|程序|source code|program code|source lines|program execution)'
        execution = r'(?:执行|运行|被测试执行|execut|\brun\b)'
        for clause in re.split(r'[，,；;。.!?\n]', similarity):
            if not re.search(shared, clause, re.I):
                continue
            # Do not reject a contrast like "二者都衡量覆盖程度，而代码覆盖率
            # 关注代码执行". This is not a claim that BOTH execute code.
            if re.search(r'(?:但|而|不过|其中|前者|后者|分别|whereas|while|but)', clause, re.I):
                continue
            has_exclusive_execution = (
                re.search(code + r'.{0,25}' + execution, clause, re.I)
                or re.search(execution + r'.{0,25}' + code, clause, re.I)
            )
            if has_exclusive_execution and not re.search(r'(?:并非|不是|不都|不能说|not both|neither)', clause, re.I):
                return '相似点将论文的检索覆盖错误说成程序代码执行覆盖；共同点只能概括双方确有的性质'
    return ''


def sourced_shared_coverage(paper_excerpt: str, external_excerpt: str,
                           paper_expansion: str, external_title: str) -> str:
    """Derive a minimal commonality ONLY when both *sources* measure coverage.

    This is a conservative recovery for weak local models that repeat a false
    "both measure code execution" claim after one repair.  It does not supply
    a metric definition and makes no inference for two non-coverage concepts.
    The two distinct, cited definitions remain model-independent evidence.
    """
    paper=(paper_expansion + ' ' + paper_excerpt).casefold()
    external=(external_title + ' ' + external_excerpt).casefold()
    if not (re.search(r'coverage|覆盖',paper)
            and re.search(r'coverage|覆盖',external)):
        return ''
    if not (re.search(r'fraction|proportion|percentage|ratio|比例|百分比',paper)
            and re.search(r'fraction|proportion|percentage|ratio|比例|百分比',external)):
        return ''
    if not (re.search(r'\btest(?:s| suite| queries)?\b|测试',paper)
            and re.search(r'\btest(?:s| suite| queries)?\b|测试',external)):
        return ''
    if not (re.search(r'retriev|cover|检索|覆盖',paper)
            and re.search(r'execut|cover|执行|覆盖',external)):
        return ''
    return ('两者都利用覆盖比例反映测试套件触及目标对象的程度；'
            '但各自的覆盖对象与触发条件不同，不能把文档检索等同于程序执行。')


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
