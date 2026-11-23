"""Bounded external background lookup for paper-independent learning.

Only vetted fixed hosts are contacted. Retrieved text is untrusted evidence,
never an instruction. We cannot certify truth or entailment from URLs alone.
"""
import re
import unicodedata
from difflib import SequenceMatcher
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



def _normalise_title(text: str) -> str:
    """Normalise typography, not the meaning of the searched concept."""
    text = unicodedata.normalize('NFKC', str(text or '')).casefold()
    text = re.sub(r'\([^)]*\)\s*$', '', text)  # Wikipedia disambiguation suffix
    return ' '.join(re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]+', text))


def source_title_relevance(source: dict, topic: str) -> float:
    """Score *the page's subject*, not a mere keyword buried in its abstract.

    A typo such as 'Maching Learning' may match 'Machine learning', but must
    not make 'Journal of Machine Learning Research' a definition source.
    This is a title/topic heuristic, not a semantic-entailment validator.
    """
    target = _normalise_title(search_topic(topic))
    title = _normalise_title(source.get('title') or source.get('section') or '')
    if not target or not title:
        return 0.0
    if target == title:
        return 1.0
    direct = SequenceMatcher(None, target, title).ratio()
    if re.search(r'[\u4e00-\u9fff]', target):
        # Do not accept an unrelated subject simply because the body contains
        # the query in a passing mention.
        return direct
    qwords = target.split()
    twords = title.split()
    if not qwords or not twords:
        return direct
    matched = [max(SequenceMatcher(None, q, t).ratio() for t in twords)
               for q in qwords]
    covered = sum(1 for similarity in matched if similarity >= .78)
    recall = covered / len(qwords)
    precision = covered / len(twords)
    token_f1 = 2 * recall * precision / (recall + precision) if covered else 0.0
    # A page about a narrower/related subject is not a definition of the
    # requested subject.  Token F1 alone scores "Quantum machine learning"
    # too highly for the (possibly misspelled) query "Machine learning".
    # Preserve close spelling corrections where the title has the same number
    # of tokens, but reject extra qualifiers until we have direct evidence.
    score = max(direct, token_f1)
    if len(twords) > len(qwords):
        # A longer title is only a safe synonym if its OWN article explicitly
        # gives the shorter queried name as an alias; otherwise it may be a
        # different subfield (Quantum ML) or artifact (an ML journal).
        body=unicodedata.normalize('NFKC', str(source.get('content') or '')[:450]).casefold()
        escaped=re.escape(target)
        explicit_alias=bool(re.search(
            r'\b(?:also\s+(?:known as|called)|(?:known|called)\s+as)\s+'
            r'(?:a|an|the)?\s*' + escaped + r'\b', body))
        if not explicit_alias:
            return min(score, 0.74)
    return score


def relevant_sources(sources: list[dict], topic: str, *, threshold: float = .76) -> list[dict]:
    """Keep sources whose *titles* identify the requested concept.

    Stable tie ordering follows the provider's search order. Do not let a
    highly-ranked but off-topic result suppress an exact match below it.
    """
    scored = [(source_title_relevance(source, topic), index, source)
              for index, source in enumerate(sources)]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [source for score, _, source in scored if score >= threshold]


def _wiki_sources(payload, lang: str):
    pages = (payload.get('query') or {}).get('pages') or {}
    ordered = sorted(pages if isinstance(pages,list) else pages.values(), key=lambda x: x.get('index', 999))
    result = []
    for item in ordered:
        title = (item.get('title') or '').strip()
        extract = re.sub(r'\s+', ' ', item.get('extract') or '').strip()
        pageid = item.get('pageid')
        if (not title or not extract or not isinstance(pageid, int)
                or title.casefold().endswith('(disambiguation)')):
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


async def _wiki_spelling_suggestion(client, lang: str, query: str) -> str:
    """Ask MediaWiki to correct a spelling error; never infer a new topic.

    ``list=search`` publishes ``searchinfo.suggestion`` / ``rewrittenquery``.
    A suggestion is only a query candidate; the page must still be retrieved
    and pass the same title-relevance gate as any other search hit.
    """
    try:
        response = await client.get(f'https://{lang}.wikipedia.org/w/api.php', params={
            'action':'query', 'list':'search', 'srsearch':query,
            'srinfo':'suggestion|rewrittenquery', 'srlimit':'1',
            'format':'json', 'formatversion':'2'})
        response.raise_for_status()
        info = (response.json().get('query') or {}).get('searchinfo') or {}
        candidate = str(info.get('suggestion') or info.get('rewrittenquery') or '').strip()
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return ''
    if not candidate or candidate.casefold() == query.casefold():
        return ''
    # Protect against topic drift caused by a broad search rewrite.
    if SequenceMatcher(None, _normalise_title(candidate),
                       _normalise_title(query)).ratio() < .77:
        return ''
    return candidate[:90]


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
            url=f'https://{lang}.wikipedia.org/w/api.php'
            try:
                r=await client.get(url, params={
                    'action':'query','generator':'search','gsrsearch':query,
                    'gsrlimit':'8','prop':'extracts','exintro':'1',
                    'explaintext':'1','format':'json','formatversion':'2'})
                r.raise_for_status()
                sources.extend(relevant_sources(_wiki_sources(r.json(),lang), query))
            except (httpx.HTTPError,ValueError):
                pass
            if sources:
                break
            # Misspellings sometimes return only related titles, e.g. a
            # specialised subfield instead of the requested parent concept.
            # Use the provider's own spelling suggestion and re-query. Never
            # promote the suggested string directly into an unchecked URL.
            suggested=await _wiki_spelling_suggestion(client,lang,query)
            if suggested:
                try:
                    r=await client.get(url, params={
                        'action':'query','generator':'search','gsrsearch':suggested,
                        'gsrlimit':'8','prop':'extracts','exintro':'1',
                        'explaintext':'1','format':'json','formatversion':'2'})
                    r.raise_for_status()
                    sources.extend(relevant_sources(_wiki_sources(r.json(),lang),query))
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
