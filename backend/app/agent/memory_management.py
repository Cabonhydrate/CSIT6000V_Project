"""Read-only memory commands, separate from retrieval-augmented Q&A.

A learner's confirmed notes are private profile data, not paper evidence.
Listing them must not invoke PDF or web search or an LLM.
"""
import re


_EN_LIST = re.compile(
    r'^\s*(?:please\s+)?(?:list|show|display|view|summari[sz]e|review)\s+'
    r'(?:(?:all|the|my|saved|confirmed|learning|study)\s+)*'
    r'(?:notes?|memories|memory|learning progress|study progress)'
    r'(?:\s+(?:i have saved|that i saved|for me))?\s*[.!?]?\s*$', re.I)
_ZH_LIST = re.compile(
    r'^\s*(?:请)?(?:列出|显示|查看|展示|总结|回顾|告诉我)\s*'
    r'(?:我|我的|所有|全部|已保存的|学习|确认的|相关的|\s)*'
    r'(?:学习记录|学习笔记|学习记忆|记忆|笔记|记录)\s*[。！？?]?\s*$')


def is_memory_listing_request(question: str) -> bool:
    return bool(_EN_LIST.fullmatch(question or '') or _ZH_LIST.fullmatch(question or ''))


def format_memory_list(records: list[dict], total: int, language: str = 'en') -> str:
    if not records:
        return ('No saved learning notes yet.' if language == 'en' else '目前没有已保存的学习记录。')
    if language == 'en':
        lines = [f'You have {total} saved learning note(s):']
    else:
        lines = [f'你目前有 {total} 条已保存的学习记录：']
    for i, item in enumerate(records, 1):
        # These are user-authored notes. Treat as inert text, not model/system
        # instructions or evidence. A line break must not create a new record.
        clean=lambda x: re.sub(r'\s+', ' ', str(x or '')).strip()
        topic=clean(item.get('topic'))[:120]
        domain=clean(item.get('domain'))[:120]
        content=clean(item.get('content'))[:1000]
        mastery=clean(item.get('mastery'))[:30]
        lines.append(f'{i}. {topic} ({domain}; {mastery}) — {content}')
    if len(records)<total:
        lines.append((f'Showing the most recent {len(records)} notes. Open Learning & Memory for the full list.'
                      if language == 'en' else f'这里只展示最近 {len(records)} 条，请在 Learning & Memory 页面查看完整列表。'))
    return '\n'.join(lines)
