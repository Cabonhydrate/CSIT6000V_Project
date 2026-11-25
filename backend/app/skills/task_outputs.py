"""Bounded contract rendering for paper reading and learning-plan Skills.

These checks validate task completeness and citations, not semantic truth. Source
facts still come from paper passages; user memories may ONLY personalize plans.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskResult:
    text: str = ''
    citations: tuple[int, ...] = ()
    issue: str = ''


def task_mode(skill: str | None, question: str, memory_count: int = 0) -> str | None:
    if skill == 'guidePaperReading':
        return 'paper_guide'
    if skill == 'suggestLearningPath':
        if (re.search(r'保存|已有.*记录|哪些知识.*接触|哪些知识.*补充|学习情况|\b(?:my saved (?:learning )?(?:notes?|memories)|learning progress|what (?:have|did) i (?:studied|learn))\b', question)
                and not re.search(r'三阶段|学习路径|学习计划', question)):
            return 'learning_progress'
        return 'learning_path'
    return None


def _refs(refs, limit):
    if not isinstance(refs, list):
        return ()
    return tuple(dict.fromkeys(value for value in refs
                               if type(value) is int and 1 <= value <= limit))


def _small_text(value, *, min_length=8, max_length=460):
    return isinstance(value, str) and min_length <= len(value.strip()) <= max_length


def render_task_output(mode: str, data, evidence: list[dict], *, memories=(), language='zh') -> TaskResult:
    if not isinstance(data, dict):
        return TaskResult(issue='缺少结构化 Skill 输出')
    if mode == 'paper_guide':
        expected = ('研究背景', '现有方法的问题', '核心创新', '实验设计', '主要结论') if language=='zh' else ('Research background', 'Limitations of prior work', 'Main contribution', 'Experimental design', 'Key findings')
        sections = data.get('sections')
        if not isinstance(sections, list) or len(sections) != len(expected):
            return TaskResult(issue='论文导读必须提供五个指定部分')
        pieces = []
        all_refs = []
        for item, heading in zip(sections, expected):
            if not isinstance(item, dict) or item.get('heading') not in (heading, ('研究背景', '现有方法的问题', '核心创新', '实验设计', '主要结论')[len(pieces)]) or not _small_text(item.get('content')):
                return TaskResult(issue=f'论文导读缺少有效部分：{heading}')
            cited = _refs(item.get('citations'), len(evidence))
            if not cited:
                return TaskResult(issue=f'{heading} 没有可定位的论文证据')
            pieces.append(f'**{heading}**: {item["content"].strip()} '+''.join(f'[{i}]' for i in cited))
            all_refs.extend(cited)
        return TaskResult(text='\n\n'.join(pieces), citations=tuple(dict.fromkeys(all_refs)))

    if mode == 'learning_path':
        steps = data.get('steps')
        if not isinstance(steps, list) or len(steps) != 3:
            return TaskResult(issue='学习路径必须恰好包含三个阶段')
        parts = []
        topics=[]
        for index, step in enumerate(steps, 1):
            if not isinstance(step, dict) or any(not _small_text(step.get(k),min_length=3,max_length=240)
                                                 for k in ('topic','learn','why','check')):
                return TaskResult(issue=f'阶段 {index} 缺少学习内容、理由或自测')
            topics.append(step['topic'].strip().casefold())
            if language=='en':
                parts.append(f'**Stage {index}: {step["topic"].strip()}**\n'
                             f'Learn: {step["learn"].strip()}\n'
                             f'Why: {step["why"].strip()}\n'
                             f'Self-check: {step["check"].strip()}')
            else:
                parts.append(f'**阶段 {index}：{step["topic"].strip()}**\n'
                             f'学习内容：{step["learn"].strip()}\n'
                             f'为什么学：{step["why"].strip()}\n'
                             f'自我检查：{step["check"].strip()}')
        if len(set(topics))!=3:
            return TaskResult(issue='三个学习阶段重复')
        citations=_refs(data.get('citations'),len(evidence))
        if citations:
            parts.append(('Paper references: ' if language=='en' else '论文参考：')+''.join(f'[{i}]' for i in citations))
        return TaskResult(text='\n\n'.join(parts), citations=citations)

    if mode == 'learning_progress':
        if not memories:
            return TaskResult(text=('No relevant confirmed learning notes were found. Save one in Learning & Memory first.'
                                    if language=='en' else '当前没有找到与你的问题相关、经过确认的学习记忆。请先在 Learning & Memory 中保存学习记录。'))
        keys=('studied','not_yet_mastered','next_reading')
        if any(not _small_text(data.get(key),min_length=5) for key in keys):
            return TaskResult(issue='学习进度回顾缺少已接触、未掌握或下一步阅读内容')
        cites=_refs(data.get('citations'),len(evidence))
        # Learning facts are user records, not source claims; only the paper
        # recommendation receives a paper citation when one is provided.
        if language=='en':
            answer=(f'**Previously studied (not necessarily mastered):** {data["studied"].strip()}\n\n'
                    f'**Knowledge gaps:** {data["not_yet_mastered"].strip()}\n\n'
                    f'**Suggested next reading:** {data["next_reading"].strip()} '
                    +''.join(f'[{i}]' for i in cites))
        else:
            answer=(f'**已经接触过（不等于掌握）**：{data["studied"].strip()}\n\n'
                    f'**仍需补充**：{data["not_yet_mastered"].strip()}\n\n'
                    f'**下一步阅读**：{data["next_reading"].strip()} '
                    +''.join(f'[{i}]' for i in cites))
        return TaskResult(text=answer,citations=cites)
    return TaskResult(issue='未知的任务型 Skill')


def verified_references(evidence: list[dict], citations: tuple[int, ...], question: str = '') -> list[dict]:
    # Only sources already returned by the paper retrieval tool may be cited.
    result=[]
    for n in citations:
        ev=evidence[n-1]
        result.append({'number':n,'evidence_id':ev['evidence_id'],
                       'paper_id':ev.get('paper_id'),'chunk_id':ev.get('chunk_id'),
                       'section':ev.get('section'), 'page_numbers':ev.get('page_numbers',[]),
                       'paragraph_refs':ev.get('paragraph_refs',[]),
                       'source_url':ev.get('source_url'),
                       'quote':str(ev.get('content') or '')[:1250]})
    return result


def user_set_calculation(question: str, language: str = 'zh', explain_union: bool = False) -> str:
    """Safely calculate a user-supplied *hypothetical* union coverage example.

    Accept an explicit corpus set (C={c1,c2}) or an explicit corpus size
    ("a corpus has 6 chunks"), and at least two explicit query result sets.
    This NEVER reads paper experimental numbers or invents a corpus size.
    """
    question = str(question or '')
    corpus_match = re.search(r'\bC\s*=\s*\{([^}]{1,150})\}', question, re.I)
    size_match = re.search(
        r'\b(?:corpus|collection)\s+(?:has|contains)\s+(\d{1,5})\s+(?:document\s+)?chunks?\b',
        question, re.I)
    chinese_size = re.search(r'语料库[^\n。；]{0,30}?(?:共有|有|包含)\s*(\d{1,5})\s*个?(?:文档块|块|chunks?)', question)
    if not corpus_match and not (size_match or chinese_size):
        return ''

    def parse(raw):
        items = [x.strip() for x in raw.split(',')]
        if not 1 <= len(items) <= 25 or not all(re.fullmatch(r'[A-Za-z]\w{0,8}', x) for x in items):
            return set()
        # A literal set has no duplicate elements. Accept repeat notation but
        # only compute the cardinality after de-duplication.
        return set(items)

    corpus = parse(corpus_match.group(1)) if corpus_match else None
    if corpus_match and not corpus:
        return ''
    size = len(corpus) if corpus is not None else int((size_match or chinese_size).group(1))
    if not 1 <= size <= 10000:
        return ''
    if corpus is not None and (size_match or chinese_size):
        if int((size_match or chinese_size).group(1)) != size:
            return ''
    matches = re.findall(
        r'\b(?:query\s*|q)(\d{1,2})\s*(?:检索到|返回|retrieves?|returns?|=)\s*\{([^}]{1,150})\}',
        question, re.I)
    queries = [(int(index), parse(raw)) for index, raw in matches]
    if not 2 <= len(queries) <= 12 or len({i for i, _ in queries}) != len(queries):
        return ''
    if any(not items or (corpus is not None and not items <= corpus) for _, items in queries):
        return ''
    union = set().union(*(items for _, items in queries))
    if len(union) > size:
        return ''
    fmt = lambda items: '{' + ','.join(sorted(items)) + '}'
    query_list = ', '.join(f'Q{i}={fmt(items)}' for i, items in queries)
    # This is student-provided arithmetic, NOT a result reported by the paper.
    fraction = f'{len(union)}/{size}'
    percent = f'{len(union) / size:.2%}'
    percent = re.sub(r'\.00(?=%)', '', percent)
    if re.search(r'\.\d0(?=%)', percent):
        percent = re.sub(r'0(?=%)', '', percent)
    if language == 'en':
        explanation = (f'1. The query result sets are {query_list}.\n'
                       f'2. Their set union is {fmt(union)}: each chunk is counted once, '
                       f'even if several queries retrieved it.\n'
                       f'3. There are {len(union)} unique retrieved chunks among {size} corpus chunks.\n'
                       f'**Chunk Coverage = {fraction} = {percent}.**')
        if explain_union:
            explanation += ('\n\n**Why set union matters:** union collects every distinct element '
                            'from the query result sets. It is NOT the sum of their sizes: '
                            f'{" + ".join(str(len(items)) for _, items in queries)} counts repeated '
                            f'chunks again, while the union contains only {len(union)} chunks.')
        return explanation
    result = ('；'.join(f'q{i}={fmt(items)}' for i, items in queries)
              + f'。去重后的并集={fmt(union)}，共 {len(union)} 个不同文档块；'
              + f'CC = {fraction} = {percent}。'
              + '集合并集会自动去掉在不同查询中重复出现的块。')
    if explain_union:
        result += '并集不是把每次检索到的数量直接相加，重复出现的文档块只能计算一次。'
    return result


def render_claim_verdicts(data, claims: list[str], evidence: list[dict], language='zh') -> TaskResult:
    """Keep the user's exact proposition in view; only generate verdict + reason.

    This enforces a complete one-to-one claim structure, not the correctness of a
    model's actual verdict. Model judgment is evaluated separately.
    """
    items=data.get('verdicts') if isinstance(data,dict) else None
    if not isinstance(items,list) or len(items)!=len(claims):
        return TaskResult(issue=f'必须逐一判断原始的 {len(claims)} 个断言')
    parts=[]
    all_refs=[]
    for i,(statement,item) in enumerate(zip(claims,items),1):
        if not isinstance(item,dict):
            return TaskResult(issue=f'第 {i} 项缺少判断对象')
        verdict=item.get('verdict')
        reason=item.get('reason')
        accepted={'正确':'supported','错误':'refuted','证据不足':'insufficient evidence',
                  'supported':'supported','refuted':'refuted','insufficient evidence':'insufficient evidence'}
        if verdict not in accepted or not _small_text(reason,min_length=8,max_length=460):
            return TaskResult(issue=f'第 {i} 项缺少明确真假判断或解释')
        citations=_refs(item.get('citations'),len(evidence))
        if not citations:
            return TaskResult(issue=f'第 {i} 项缺少论文引用')
        if language=='en':
            parts.append(f'{i}. **Claim: {statement}**\n\n'
                         f'**Verdict: {accepted[verdict]}.** {reason.strip()} '
                         +''.join(f'[{n}]' for n in citations))
        else:
            parts.append(f'{i}. **原断言：{statement}**\n\n'
                         f'**判断：{verdict}。** {reason.strip()} '
                         +''.join(f'[{n}]' for n in citations))
        all_refs.extend(citations)
    return TaskResult(text='\n\n'.join(parts),citations=tuple(dict.fromkeys(all_refs)))
