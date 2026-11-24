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


def user_set_calculation(question: str) -> str:
    """Compute a small *user-provided hypothetical* set example without LLM math.

    Only activates when a corpus and >=2 query retrieval sets are explicit.
    Never invent source evidence or transform a scholarly experiment.
    """
    corpus_match = re.search(r'\bC\s*=\s*\{([^}]{1,100})\}', question, re.I)
    if not corpus_match:
        return ''
    def parse(raw):
        parts = [x.strip() for x in raw.split(',')]
        return set(parts) if 1 <= len(parts) <= 25 and all(re.fullmatch(r'[A-Za-z]\w{0,8}',x) for x in parts) else set()
    corpus = parse(corpus_match.group(1))
    matches = re.findall(r'\bq(\d+)\s*(?:检索到|返回|retrieves?|=)\s*\{([^}]{1,100})\}',question,re.I)
    queries = [(int(index),parse(raw)) for index,raw in matches]
    if not corpus or len(queries)<2 or any(not items or not items<=corpus for _,items in queries):
        return ''
    total=set().union(*(items for _,items in queries))
    display=lambda items:'{' + ','.join(sorted(items)) + '}'
    return ('；'.join(f'q{i}={display(items)}' for i,items in queries)
            +f'。去重后的并集={display(total)}，共 {len(total)} 个不同文档块；'
            +f'CC = {len(total)}/{len(corpus)} = {len(total)/len(corpus):.0%}。'
            +'集合并集会自动去掉在不同查询中重复出现的块。')


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
