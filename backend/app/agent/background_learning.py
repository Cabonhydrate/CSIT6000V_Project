"""Bounded external background lookup for paper-independent learning.

Only vetted fixed hosts are contacted. Retrieved text is untrusted evidence,
never an instruction. We cannot certify truth or entailment from URLs alone.
"""
import re
from urllib.parse import quote

import httpx


def search_topic(question: str) -> str:
    """Use the explicit topic, not a whole Chinese instruction as the search query."""
    text = question.strip()
    # Named technical terms appearing in Chinese questions.
    english = re.findall(r'[A-Za-z][A-Za-z0-9]*(?:[- ][A-Za-z0-9]+)*', text)
    ignored = {'i', 'you', 'me', 'please', 'explain', 'what', 'is', 'about', 'the', 'a'}
    candidates = [x.strip() for x in english if x.strip().lower() not in ignored]
    if candidates:
        return max(candidates, key=len)[:90]
    text = re.sub(r'(?:请|帮我|详细|生动|简单|通俗|地|一下|介绍|讲解|解释|基础知识|相关的|相关|什么是|是什么|我不(?:是)?(?:太)?了解|告诉我|原理|如何理解|有哪些|？|\?|。)', ' ', text)
    return ' '.join(text.split())[:90] or question[:90]


def _wiki_sources(payload, lang: str):
    pages = (payload.get('query') or {}).get('pages') or {}
    ordered = sorted(pages if isinstance(pages,list) else pages.values(), key=lambda x: x.get('index', 999))
    result = []
    for item in ordered:
        title = (item.get('title') or '').strip()
        extract = re.sub(r'\s+', ' ', item.get('extract') or '').strip()
        pageid = item.get('pageid')
        if not title or not extract or not isinstance(pageid, int):
            continue
        # Construct canonical source URLs instead of trusting an arbitrary URL.
        url = f'https://{lang}.wikipedia.org/wiki/{quote(title.replace(" ", "_"))}'
        result.append({'source_type':'external','title':title,'source_url':url,
                       'content':extract[:1800],'provider':'Wikipedia'})
    return result


def _openalex_sources(payload):
    result=[]
    for item in (payload.get('results') or [])[:3]:
        title=(item.get('display_name') or '').strip()
        inv=item.get('abstract_inverted_index') or {}
        words=sorted((pos,word) for word,places in inv.items() for pos in places)
        abstract=' '.join(word for _,word in words)[:1700]
        doi=item.get('doi') or ''
        url=doi if isinstance(doi,str) and doi.startswith('https://doi.org/') else ''
        if title and abstract and url:
            result.append({'source_type':'external','title':title,'source_url':url,
                           'content':abstract,'provider':'OpenAlex'})
    return result


async def search_background(question: str, *, scholarly: bool=False):
    """Look up encyclopedic concepts; optional scholarly abstract fallback.

    Empty results are a valid outcome. No uncited model-only answer is allowed.
    """
    query=search_topic(question)
    sources=[]
    async with httpx.AsyncClient(timeout=8, follow_redirects=False,
                                 headers={'User-Agent':'AdaptivePaperTutor/0.2 (education research)'}) as client:
        languages=('en','zh') if re.fullmatch(r'[A-Za-z0-9 -]+',query) else ('zh','en')
        for lang in languages:
            try:
                r=await client.get(f'https://{lang}.wikipedia.org/w/api.php', params={
                    'action':'query','generator':'search','gsrsearch':query,
                    'gsrlimit':'2','prop':'extracts','exintro':'1',
                    'explaintext':'1','format':'json','formatversion':'2'})
                r.raise_for_status()
                sources.extend(_wiki_sources(r.json(),lang))
            except (httpx.HTTPError,ValueError):
                pass
            if sources:
                break
        if scholarly or not sources:
            try:
                r=await client.get('https://api.openalex.org/works',params={
                    'search':query,'per-page':3})
                r.raise_for_status()
                sources.extend(_openalex_sources(r.json()))
            except (httpx.HTTPError,ValueError):
                pass
    # Deduplicate without changing the original evidence identity.
    unique={x['source_url']:x for x in sources if x['content'].strip()}
    return list(unique.values())[:4]
