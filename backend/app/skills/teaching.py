"""Evidence-grounded, beginner-facing answer contract.

The model fills fields. Python validates and renders the final answer. Personal
learning memory controls style ONLY; it cannot supply academic evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from ..knowledge.excerpts import source_excerpt


@dataclass(frozen=True)
class TeachingResult:
    text: str = ''
    citation_ids: tuple[int, ...] = ()
    issue: str = ''


def _topic(question: str) -> str:
    # A named academic concept in English embedded in a Chinese prompt.
    hits = re.findall(r'\b[A-Z][a-z]+(?:[-\s]+[A-Z][a-z]+)+\b', question)
    return max(hits, key=len) if hits else ''


def prioritize_teaching_evidence(evidence: list[dict], question: str, limit: int = 3) -> list[dict]:
    """Promote actual definition-bearing chunks, not random benchmark results.

    Keep original evidence IDs; only ordering of already retrieved passages changes.
    """
    topic = _topic(question).casefold()
    scored = []
    for pos, ev in enumerate(evidence):
        content = str(ev.get('content') or '')
        low = content.casefold()
        topic_count = low.count(topic) if topic else 0
        # Use document-level evidence to rank definitions above metric tables.
        definition = bool(re.search(r'\b(?:define|defined|measure|measures|fraction|proportion|denote|refers to)\b', low))
        matches = list(re.finditer(re.escape(topic), low)) if topic else []
        near_definition = any(re.search(
            r'\b(?:define|defined|measure|measures|fraction|proportion|denote|refers to|ratio)\b',
            low[max(0, m.start()-150):m.end()+290]) for m in matches[:12])
        calculation_only = bool(re.search(r'\b(?:table|figure)\s*\d+\b', low))
        acronym = ''.join(part[0] for part in topic.split() if part) if topic else ''
        has_formal = bool(acronym and re.search(
            r'(?<![A-Za-z])' + re.escape(acronym) + r'\s*\([^)]{1,15}\)', content, re.I))
        score = (35 * int(has_formal) + 20 * int(near_definition) +
                 3 * min(topic_count, 3) + 2 * int(definition and topic_count > 0) -
                 int(calculation_only))
        scored.append((-score, pos, ev))
    scored.sort(key=lambda item: item[:2])
    return [entry[2] for entry in scored[:limit]]


def teaching_evidence_quote(evidence: dict, question: str, max_chars: int = 1300) -> str:
    """Return an actual source span near the named concept, never generated text."""
    source = str(evidence.get('content') or '')
    topic = _topic(question)
    if not source or not topic:
        return source_excerpt(source,max_chars,focus=question)
    positions = list(re.finditer(re.escape(topic), source, re.I))
    if not positions:
        return source_excerpt(source,max_chars,focus=question)
    acronym = ''.join(part[0] for part in topic.split() if part)
    formula = (re.search(r'(?<![A-Za-z])' + re.escape(acronym) +
                         r'\s*\([^)]{1,15}\)', source, re.I) if acronym else None)
    if formula:
        # Formal definition may occur after the section heading/diagram.
        start = max(0, formula.start()-480)
        return source_excerpt(source[start:],max_chars,focus=question)
    keywords = re.compile(r'\b(?:defined|define|measures?|fraction|proportion|denote|ratio)\b', re.I)
    best = max(positions, key=lambda m: (bool(keywords.search(
        source[max(0,m.start()-130):m.end()+240])), -m.start()))
    start = max(0, best.start()-260)
    return source_excerpt(source[start:],max_chars,focus=question)


def build_teaching_system(language: str = 'zh', skill_instructions: str = '') -> str:
    base = (
        '你是论文初学者教学助手。只输出一个合法 JSON 对象，字段恰好为：'
        '{"definition":"...","example":"...","takeaway":"...","citations":[1]}。'
        'definition：用1至2句直答概念定义，必须与下方论文证据一致，不能偷换领域；'
        'example：若用户要求举例，构造一个简短、具体的假设数字例子（不得当作论文实验）；'
        'takeaway：用1句说明直观含义或与易混概念的区别；'
        'citations：只填写直接支持definition的证据编号（从1开始），通常一个。'
        '不能根据学习记忆猜测论文事实；只有论文摘录是事实依据。'
        '严禁把 RAG 的文档片段误称成代码块，除非论文证据明确研究代码。'
        '无须展示思考步骤、用户画像、检索过程和证据列表。'
        '没有要求时，不讨论实验相关系数、熵、Gini或其他额外指标。'
        '全部文字使用中文，控制总长度在约120到260个汉字。'
        if language == 'zh' else
        'You are a beginner paper tutor. Output ONLY JSON with keys '
        '"definition", "example", "takeaway", "citations". Definition must be '
        'directly grounded in the supplied paper; example must be small, concrete '
        'and explicitly hypothetical; takeaway explains the intuition. '
        'Cite only the paper passage supporting the definition (1-based). '
        'Learning memories adjust style, never source facts. No reasoning trace, '
        'planning, benchmark details not requested, or user profile recitation.'
    )
    return base + ('\n' + skill_instructions if skill_instructions else '')


def validate_and_render_teaching(data, evidence: list[dict], question: str,
                                 language: str = 'zh') -> TeachingResult:
    if not isinstance(data, dict):
        return TeachingResult(issue='模型未返回结构化教学对象')
    fields = {}
    for field, maximum in (('definition', 320), ('example', 300), ('takeaway', 230)):
        value = data.get(field)
        if not isinstance(value, str):
            return TeachingResult(issue=f'缺少教学字段：{field}')
        value = re.sub(r'\[\d+\]', '', value).strip()
        if not value or len(value) > maximum:
            return TeachingResult(issue=f'教学字段为空或过长：{field}')
        fields[field] = value
    indices = data.get('citations')
    if not isinstance(indices, list):
        return TeachingResult(issue='缺少引用数组')
    cites = tuple(dict.fromkeys(i for i in indices if type(i) is int and 1 <= i <= len(evidence)))
    if not cites:
        return TeachingResult(issue='没有可验证的来源编号')
    topic = _topic(question)
    if topic and topic.casefold() not in fields['definition'].casefold():
        # A different metric can sound plausible and still answer the wrong question.
        return TeachingResult(issue='定义未准确指向用户提出的概念')
    combined = fields['definition'] + '\n' + fields['takeaway']
    if re.search(r'(?im)^\s*(?:首先.?用户|用户的学习背景|问题：|关键点：|从摘录中)', combined):
        return TeachingResult(issue='模型输出了内部规划文本')
    if re.search(r'(?i)\b(?:negative|negatively) correlated\b|负相关', combined) and not re.search(
            r'负相关|negatively correlated|negative correlation',
            '\n'.join(str(ev.get('content') or '') for ev in evidence), re.I):
        return TeachingResult(issue='回答引入了来源未支持的负相关结论')
    quoted = '\n'.join(str(evidence[i - 1].get('content') or '') for i in cites).casefold()
    if topic and topic.casefold() not in quoted:
        return TeachingResult(issue='引用未包含所解释的概念')
    # Generic domain-consistency guard; only activates when the selected evidence
    # refers to retrieval/document chunks and not to source code.
    retrieval_document = bool(re.search(r'\b(?:document|corpus|retriev\w*)\b', quoted)) and 'chunk' in quoted
    source_is_code = bool(re.search(r'\b(?:source code|code chunk|code coverage|program code)\b', quoted))
    if retrieval_document and not source_is_code:
        # A normal comparison with *code coverage* is not a claim that
        # document chunks are source-code fragments. The previous gate rejected
        # these useful comparisons even when paper evidence was available.
        # Flag only statements which directly define CC/chunks in code terms.
        misconception = re.compile(
            r'(?:Chunk\s+Coverage|(?<![A-Za-z])CC(?![A-Za-z])|'
            r'(?:文档|检索|文本)(?:块|片段)|(?<![A-Za-z])chunks?(?![A-Za-z]))'
            r'[^。；;\n，,]{0,24}(?:是|即|指|表示|衡量|统计|计算|等于|定义为)'
            r'[^。；;\n，,]{0,28}(?:代码块|代码片段|代码覆盖率|代码执行|'
            r'code chunks?|code fragments?)', re.I)
        if misconception.search(combined) and not re.search(
                r'不是|并非|不等于|不同于|区别于|而非|而不是|does not|not code',
                combined, re.I):
            return TeachingResult(issue='将检索文档片段错误解释为代码片段')
    if re.search(r'举例|例子|示例|example', question, re.I) and not re.search(
            r'\d|\{[^}]+\}', fields['example']):
        return TeachingResult(issue='缺少具体的数字或集合例子')
    if re.search(r'(?i)\b(?:table|figure)\s*\d+|相关系数|基尼|gini|correlation', combined) and not re.search(
            r'(?i)\b(?:table|figure)\s*\d+|相关系数|基尼|gini|correlation', question):
        return TeachingResult(issue='增加了用户未请求的实验或统计内容')
    # Keep the model from adding citation-looking text to hypothetical examples.
    if language == 'zh':
        answer = (f"**核心定义**：{fields['definition']} " + ''.join(f'[{i}]' for i in cites) +
                  f"\n\n**举个例子（假设）**：{fields['example']}" +
                  f"\n\n**关键理解**：{fields['takeaway']}")
    else:
        answer = (f"**Definition**: {fields['definition']} " + ''.join(f'[{i}]' for i in cites) +
                  f"\n\n**Hypothetical example**: {fields['example']}" +
                  f"\n\n**Takeaway**: {fields['takeaway']}")
    return TeachingResult(text=answer, citation_ids=cites)
