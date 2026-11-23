"""Conservative source-backed checks for compact definitional paper answers.

This is a high-precision safety gate, not semantic entailment or a knowledge base.
No named-paper definitions or Gold answers are embedded here. A definition anchor
must be extracted from the current retrieved text before any check is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class DefinitionAnchor:
    index: int                         # 1-based index into the supplied evidence
    subject: str                       # literal subject copied from the question
    source_excerpt: str               # actual unmodified source span
    ratio: bool
    retrieved: bool


def _plain(value: str) -> str:
    return unicodedata.normalize('NFKC', str(value or ''))


def _subject(question: str) -> str:
    """Only extract an unambiguous named subject; otherwise avoid guessing."""
    q = _plain(question)
    acronyms = re.findall(r'(?<![A-Za-z0-9])[A-Z]{2,10}(?![A-Za-z0-9])', q)
    if acronyms:
        # E.g. 'CC 和 MRR 的区别' isn't a compact single-subject definition.
        distinct = list(dict.fromkeys(acronyms))
        if len(distinct) == 1:
            return distinct[0]
        return ''
    # A term explicitly quoted by the learner, without assuming any vocabulary.
    quoted = re.findall(r'[“"\']([A-Za-z][A-Za-z0-9 -]{2,55})[”"\']', q)
    if len(quoted) == 1:
        return quoted[0].strip()
    match = re.search(r'(?i)\bwhat\s+is\s+(?:an?\s+|the\s+)?([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,4})\s*\??$', q)
    return match.group(1).strip() if match else ''


def definition_subject(question: str) -> str:
    """Return an explicitly named definition subject, without guessing."""
    return _subject(question)


def _hits(text: str, subject: str) -> list[re.Match]:
    return list(re.finditer(r'(?<![A-Za-z0-9])'+re.escape(subject)+r'(?![A-Za-z0-9])',text,re.I))


def _source_excerpt(text: str, location: int) -> str:
    # Respect local source boundaries: keep the exact string for verification.
    start=max(text.rfind('\n',0,location),text.rfind('. ',0,location))
    start=0 if start<0 else start+1
    # PDF text often wraps mid-sentence; do not stop at a hard line break.
    # This is still a literal slice of the original source, not an inference.
    return text[start:min(len(text),max(start+350,location+370))].strip()


def definition_source(question: str, evidence: list[dict]) -> DefinitionAnchor | None:
    """Locate an explicit, *subject-attached* definition in the actual source.

    A page mentioning the term and a different metric later in the paragraph
    is not a definition. PDF line breaks are allowed inside a literal clause.
    The returned index is always relative to the supplied evidence list.
    """
    subject = _subject(question)
    if not subject:
        return None
    # A strict predicate immediately after the subject is more reliable than
    # seeing words such as 'ratio' or 'retrieved' somewhere on a long PDF page.
    predicate = re.compile(
        r'(?i)^\s*(?:\)\s*)?(?:[,;:]\s*)?'
        r'(?:measur(?:e|es|ed)|is\s+defined\s+as|refers?\s+to|'
        r'denotes?|represents?|is\s+(?:an?\s+|the\s+))\b'
    )
    candidates = []
    for index, ev in enumerate(evidence, 1):
        original = str(ev.get('content', ''))
        # Keep offsets in the original text so the excerpt is an authentic
        # source slice; collapse whitespace only for evaluating the clause.
        for hit in _hits(original, subject):
            after = _plain(original[hit.end():hit.end() + 190])
            after = re.sub(r'\s+', ' ', after)
            direct = predicate.search(after)
            if direct is None:
                # 'Full Name (CC), a ... . CC measures ...' is picked up by
                # the *second* subject occurrence. This first mention alone
                # isn't sufficient evidence of a precise numeric definition.
                continue
            excerpt = _source_excerpt(original, hit.start())
            # Read only the first clause; content after a sentence boundary
            # can define a completely different quantity.
            clause = re.split(r'[.!?](?=\s|$)', after[direct.end():], maxsplit=1)[0][:175]
            near = after[direct.start():direct.end()] + clause
            ratio = bool(re.search(r'(?i)\b(?:fraction|proportion|percentage|ratio)\b', near))
            retrieved = bool(re.search(r'(?i)retriev(?:ed|al)|fetched|returned\s+by\s+the\s+retriever', near))
            # Prefer a definition with the metric dimension and object stated
            # in the same clause. A generic 'is a method' remains eligible.
            score = 10 + (5 if ratio else 0) + (4 if retrieved else 0)
            if re.match(r'(?i)\s*measur', direct.group()):
                score += 2
            candidates.append((score, -index,
                               DefinitionAnchor(index, subject, excerpt, ratio, retrieved)))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def prioritize_definition_evidence(question: str, evidence: list[dict]) -> tuple[list[dict], DefinitionAnchor | None]:
    """Promote a real definition into the first five citation slots.

    This only changes the *answer context*, never the recorded retrieval ranks.
    No model-generated text is used to create or choose the source.
    """
    ordered = list(evidence)
    source = definition_source(question, ordered)
    if source is None:
        return ordered, None
    if source.index > 5:
        # Keep existing [1]...[5] citation IDs stable whenever the definition
        # was already visible; only promote a previously hidden hit.
        index = source.index - 1
        ordered.insert(0, ordered.pop(index))
    return ordered, definition_source(question, ordered)




def definition_issue(question: str, answer: str, evidence: list[dict],
                     cited_indices: list[int], anchor: DefinitionAnchor | None=None) -> str:
    """Return a concrete, conservative failure reason, or empty when unproven.

    Works only on a single-subject definition with a source anchor. It avoids
    pretending that lexical evidence matching proves arbitrary semantics.
    """
    anchor=anchor or definition_source(question,evidence)
    if anchor is None:
        return ''
    a=_plain(re.sub(r'\[\d+\]','',answer)).strip()
    if not a:
        return '定义回答为空'
    if not cited_indices:
        return '定义回答缺少论文引用'
    cited=[evidence[i-1] for i in cited_indices if type(i) is int and 1<=i<=len(evidence)]
    if not cited:
        return '定义回答没有有效论文引用'
    # Require *at least one* cited source to contain a subject-specific
    # explanation. Mere occurrence of a source number is insufficient.
    if definition_source(question,cited) is None:
        return '当前引用没有包含所问术语的直接定义，请引用实际定义该术语的论文段落'
    subject=re.escape(anchor.subject)
    # For ratio-valued source metrics, a bare scalar count is dimensionally
    # incorrect. Fractions and percentages are valid; '3 chunks' is not.
    if anchor.ratio:
        bare=re.search(r'(?<![A-Za-z0-9])'+subject+
                       r'\s*(?:\([^)]{0,12}\))?\s*(?:=|：|:)\s*'
                       r'(?P<value>\d+(?:\.\d+)?)\s*(?![\d/%％÷])(?=$|[，。,；;\s])',a,re.I)
        if bare:
            return '原文定义的是比例/分数，但答案把指标直接等同于没有分母的数量'
        # A definition that explicitly says 'unused/not retrieved' is the
        # complement of a source defining retrieved coverage. Do not reject
        # valid caveats such as 'does NOT measure un-retrieved items'.
        intro=re.split(r'[。！？\n]',a,maxsplit=1)[0][:260]
        intro=re.sub(r'^\s*#+\s*','',intro)
        if anchor.retrieved:
            positive_definition=re.search(r'(?:是|指|表示|衡量|定义为|等于|means?|is|measures?|denotes?)',intro,re.I)
            negated=re.search(r'(?:未被|没有被|未曾被|不曾被|未实际被|未检索|未曾检索|没有检索|未取回|not\s+(?:been\s+)?retrieved|unretrieved|unused)',intro,re.I)
            if positive_definition and negated:
                between=intro[max(0,positive_definition.start()-3):negated.start()]
                # '不衡量未被检索的数量' is a valid *negative* caveat.
                if not re.search(r'不(?:是|指|表示|衡量|等于|代表)|not\s+(?:the\s+)?(?:fraction|proportion|measure)',between,re.I):
                    return '原文定义覆盖已检索的对象，而答案却把指标解释成未检索/未使用对象'
    if not _hits(a, anchor.subject):
        return '回答没有定义所问术语'
    if anchor.ratio and anchor.retrieved:
        mentions_retrieval = bool(re.search(
            r'(?i)检索|召回|命中|取回|retriev|fetched|returned\s+by', a))
        mentions_quantity = bool(re.search(
            r'(?i)比例|占比|百分比|分数|比率|\b(?:fraction|proportion|percentage|ratio|coverage)\b|\d+\s*[/％%]', a))
        mentions_corpus_unit = bool(re.search(
            r'(?i)文档|语料|片段|记录|数据块|chunk|passage|corpus|record|document', a))
        if not (mentions_retrieval and mentions_quantity and mentions_corpus_unit):
            return '定义未体现原文要求的检索对象、语料单位及比例量纲'
        first_clause = re.split(r'[。！？\n]', a, maxsplit=1)[0][:255]
        if re.search(r'(?i)(?:无关|不相关|irrelevant|unrelated).{0,35}'
                     r'(?:比例|占比|百分比|fraction|ratio)', first_clause):
            if not re.search(r'(?i)(?:不(?:是|指|衡量|等于|代表)|not\s+(?:the|a|an)?\s*measure)', first_clause):
                return '答案把无关内容的占比当作原文定义的检索覆盖比例'
        # Only treat ungrounded formula identifiers as a problem when the
        # answer explicitly asserts an equation for this named metric.
        formula = re.search(r'(?:公式|计算(?:式|公式)|equation|formula).{0,20}(?:为|=|is)\s*(.{0,95})',a,re.I)
        if formula:
            lhs_rhs=formula.group(1)
            tokens=set(re.findall(r'(?<![A-Za-z])[A-Za-z]{3,}[A-Za-z0-9]*',lhs_rhs))
            symbols={t for t in tokens if re.search(r'[A-Z]',t) and
                     re.search(r'[a-z]',t) and len(t)>=4}
            if symbols and any(t.casefold() not in anchor.source_excerpt.casefold() for t in symbols):
                return '答案中的公式标识符没有出现在该术语的定义证据中'
    return ''


def definition_repair_context(anchor: DefinitionAnchor, evidence: list[dict]) -> str:
    """Offer only genuine indexed source text; never invent a new citation."""
    source=evidence[anchor.index-1]
    return (f'[{anchor.index}] section={source.get("section","")}, pages={source.get("page_numbers",[])}\n'
            f'{anchor.source_excerpt}')


def literal_definition_fallback(question: str, draft: str, evidence: list[dict],
                                anchor: DefinitionAnchor | None = None,
                                language: str = 'zh') -> str:
    """Recover a cited *verbatim* definition when a small model omits its citation.

    This is intentionally narrow: only ratio-valued retrieval metrics with a
    literal "SUBJECT measures ..." sentence. The model draft must have the
    correct subject, dimension and polarity; the returned answer, however, is
    made entirely from the real source text, not its uncited paraphrase.
    It cannot turn a repeatedly incorrect model draft into a success.
    """
    anchor = anchor or definition_source(question, evidence)
    if not anchor or not (anchor.ratio and anchor.retrieved):
        return ''
    if not 1 <= anchor.index <= len(evidence):
        return ''
    content = _plain(draft)
    if not _hits(content, anchor.subject):
        return ''
    if not re.search(r'(?i)比例|占比|百分比|分数|比率|fraction|proportion|percentage|ratio', content):
        return ''
    if not re.search(r'(?i)检索|召回|retriev|fetched|returned\s+by\s+the\s+retriever', content):
        return ''
    # Test the candidate with the actual source index, but NEVER publish the
    # generated draft with an automatically attached reference. The fallback
    # instead quotes the exact definition from the source.
    if definition_issue(question, content, evidence, [anchor.index], anchor):
        return ''
    source = _plain(evidence[anchor.index - 1].get('content', ''))
    compact = ' '.join(source.split())
    pattern = (r'(?<![A-Za-z0-9])' + re.escape(anchor.subject) +
               r'(?![A-Za-z0-9])(?:\s*\))?\s+(?:measures?|is\s+defined\s+as|denotes?|represents?|refers?\s+to)\b')
    for m in re.finditer(pattern, compact, re.I):
        start = m.start()
        # Retain the literal long form in "Long Name (ABC) measures ...".
        # Starting at ABC would otherwise produce the awkward "ABC) measures".
        if start and compact[start - 1] == '(':
            expansion = re.search(
                r'([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,8})\s*\($',
                compact[max(0,start-120):start])
            start = max(0,start-120) + expansion.start() if expansion else start-1
        tail = compact[start:start + 450]
        stop = re.search(r'[.!?](?=\s|$)', tail)
        quotation = tail[:stop.end()] if stop else tail[:360].rstrip(' ;:')
        if not (45 <= len(quotation) <= 450):
            continue
        if not (re.search(r'(?i)fraction|proportion|ratio|percentage', quotation) and
                re.search(r'(?i)retriev', quotation)):
            continue
        # The quote is lifted from this source verbatim apart from whitespace.
        if quotation not in compact:
            continue
        if language == 'zh':
            return (f'根据论文原文，{anchor.subject} 的定义是：\n\n'
                    f'“{quotation}” [{anchor.index}]')
        return f'The paper defines {anchor.subject} as: “{quotation}” [{anchor.index}]'
    return ''
