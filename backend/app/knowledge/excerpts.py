"""Bounded citation previews from stored source text, never generated content.

A preview is deliberately smaller than the original source paragraph/chunk.
Do not treat a bounded preview as a complete indexed chunk or full PDF page.
"""
import re


def source_excerpt(source: str, limit: int = 1300, *, focus: str = '') -> str:
    """Select a contiguous, readable source substring near the requested topic.

    Return text exactly copied from the source (no fabricated ellipses); the UI
    labels it a preview. For long text, prefer sentence and whitespace borders
    so quotes do not end mid-word/mid-sentence when reasonable.
    """
    if len(source) <= limit:
        return source
    limit=max(80,limit)
    start=0
    focus_terms=[term for term in re.findall(r'[A-Za-z][A-Za-z0-9-]{2,}',focus)
                 if len(term)>3 and term.lower() not in {
                     'what','where','when','why','which','the','and','from','with',
                     'explain','does','that','this','about','again','same',
                     'correct','answer','answers','high','repeatedly','retrieving'}]
    tokens=list(dict.fromkeys(term.lower() for term in focus_terms))[:12]
    if tokens:
        positions=[]
        lower=source.lower()
        for term in tokens:
            for match in list(re.finditer(re.escape(term),lower))[:12]:
                positions.append(match.start())
        best=(-1,0)
        for position in positions:
            begin=max(0,position-65)
            window=lower[begin:begin+limit]
            score=sum((4 if len(term)>=8 else 1) for term in tokens if term in window)
            score+=2*bool(re.search(r'\b(measures?|defined|fraction|proportion|definition)\b',window))
            if score>best[0]:best=(score,begin)
        if best[0]>=0:
            start=best[1]
    # Prefer beginning just after a sentence boundary near the target.
    if start:
        left=source.rfind('. ',max(0,start-130),start+1)
        if left!=-1:start=left+2
        else:
            left=source.rfind('\n',max(0,start-130),start+1)
            if left!=-1:start=left+1
    end=min(len(source),start+limit)
    if end<len(source):
        window=source[max(start,end-290):end]
        boundaries=[m.end() for m in re.finditer(r'[.!?。！？](?:\s|$)',window)]
        if boundaries:
            cut=max(start,end-290)+boundaries[-1]
            if cut-start>=int(limit*.60):end=cut
        elif end>start+100:
            # Only as a fallback; never silently replace source text.
            space=max(source.rfind(' ',end-100,end),source.rfind('\n',end-100,end))
            if space>start+int(limit*.60):end=space
    return source[start:end].strip()
