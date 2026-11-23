"""One corpus, three retrieval modes. Gold labels are never read here."""
from collections import Counter
import math
import re
import time
from pathlib import Path
import numpy as np
from .. import db
from ..config import settings
from ..logging_setup import get_logger

logger = get_logger("retrieval")


class DenseUnavailable(RuntimeError):
    pass


def tokenize(s):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s.lower())


def bm25_scores(query, texts):
    tokenized = [tokenize(t) for t in texts]
    q=tokenize(query)
    if not q: return np.zeros(len(texts))
    try:
        from rank_bm25 import BM25Okapi
        return np.array(BM25Okapi(tokenized).get_scores(q),dtype='float32')
    except ImportError:
        pass
    n=len(texts)
    avg_len=sum(map(len,tokenized))/max(n,1)
    df=Counter(t for tokens in tokenized for t in set(tokens))
    scores=np.zeros(n,dtype='float32')
    for i,words in enumerate(tokenized):
        cnt=Counter(words)
        norm=1.5*(0.25+0.75*len(words)/max(avg_len,1))
        for term in q:
            if term in cnt:
                idf=math.log(1+(n-df[term]+0.5)/(df[term]+0.5))
                scores[i]+=idf*cnt[term]*2.5/(cnt[term]+norm)
    return scores


class DenseEncoder:
    def __init__(self):
        self.model=None

    def encode(self, texts):
        try:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model=SentenceTransformer(settings.embedding_model,device='cpu')
            return np.asarray(self.model.encode(texts,normalize_embeddings=True,show_progress_bar=False),dtype='float32')
        except (ImportError,OSError) as exc:
            raise DenseUnavailable('Dense retrieval requires sentence-transformers and a locally cached embedding model. '
                                   'Install requirements-ai.txt and download the model on a networked machine.') from exc


encoder=DenseEncoder()


def _dense_scores(query, chunks):
    if not chunks: return np.zeros(0)
    # Cache by stable chunk IDs + content digest + embedding model; deletions cannot leak stale rows.
    import hashlib
    fingerprint=hashlib.sha256((settings.embedding_model+'|'+"|".join(f"{c['id']}:{hashlib.sha256(c['content'].encode()).hexdigest()}" for c in chunks)).encode()).hexdigest()[:20]
    cache=settings.data_dir/'indexes'/f'{fingerprint}.npz'
    if cache.exists():
        with np.load(cache) as bundle:
            vectors=bundle['vectors']
        if vectors.shape[0]!=len(chunks):
            cache.unlink(missing_ok=True)
            vectors=encoder.encode([c['content'] for c in chunks])
            np.savez_compressed(cache,vectors=vectors)
    else:
        vectors=encoder.encode([c['content'] for c in chunks])
        np.savez_compressed(cache,vectors=vectors)
    with db.connect() as c:
        for paper_id in set(x['paper_id'] for x in chunks):
            c.execute('INSERT OR IGNORE INTO vector_cache(cache_file,paper_id) VALUES(?,?)',(cache.name,paper_id))
    q=encoder.encode([query])[0]
    try:
        import faiss
        index=faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        d,idx=index.search(q.reshape(1,-1),len(chunks))
        scores=np.empty(len(chunks),dtype='float32')
        scores[idx[0]]=d[0]
        return scores
    except ImportError:
        return vectors @ q  # exact cosine search on CPU, same ranking semantics


def _rerank(query, evidence):
    try:
        from sentence_transformers import CrossEncoder
        cross=CrossEncoder(settings.reranker_model,device='cpu')
        pairs=[(query,x['content']) for x in evidence]
        scores=cross.predict(pairs)
        return sorted([dict(x,rerank_score=float(s)) for x,s in zip(evidence,scores)],key=lambda x:x['rerank_score'],reverse=True)
    except (ImportError,OSError) as exc:
        raise DenseUnavailable('Reranking requires sentence-transformers and a local CrossEncoder model') from exc


def search(query, paper_ids, strategy='keyword', top_k=5, rerank=False, run_id=None):
    if not paper_ids or len(paper_ids)>8:
        raise ValueError('Select between 1 and 8 papers')
    t0=time.perf_counter()
    placeholders=','.join('?' for _ in paper_ids)
    chunks=db.rows(f'''SELECT c.* FROM chunks c JOIN papers p ON p.id=c.paper_id
                        WHERE c.paper_id IN ({placeholders}) AND p.owner_id=? AND p.status='ready'
                        ORDER BY c.paper_id,c.seq''',(*paper_ids,settings.owner_id))
    if not chunks:
        logger.warning('run=%s retrieval strategy=%s papers=%s chunks=0', run_id or '-',strategy, paper_ids)
        return {'query':query,'strategy':strategy,'evidences':[],'latency_ms':round((time.perf_counter()-t0)*1000,2),'rerank_applied':False}
    if strategy not in ('keyword','dense','hybrid'):
        raise ValueError('Unknown retrieval strategy')
    texts=[x['content'] for x in chunks]
    if strategy=='keyword':
        scores=bm25_scores(query,texts)
        order=np.argsort(-scores).tolist()
    elif strategy=='dense':
        scores=_dense_scores(query,chunks)
        order=np.argsort(-scores).tolist()
    else:
        lexical=bm25_scores(query,texts)
        semantic=_dense_scores(query,chunks)
        # Reciprocal Rank Fusion; rankings are robust to incomparable score scales.
        fusion=np.zeros(len(chunks),dtype='float32')
        for r in (np.argsort(-lexical),np.argsort(-semantic)):
            for rank,index in enumerate(r): fusion[index]+=1/(60+rank+1)
        scores=fusion
        order=np.argsort(-scores).tolist()
    # Avoid returning zero lexical matches as valid evidence.
    if strategy=='keyword': order=[i for i in order if scores[i]>0]
    candidate_n=min(len(order),max(12,top_k*3) if rerank else top_k)
    evidence=[]
    for i in order[:candidate_n]:
        c=chunks[i]
        evidence.append({'evidence_id':c['id'], 'source_type':'paper','paper_id':c['paper_id'],
                         'chunk_id':c['id'],'section':c['section'],'content':c['content'],
                         'paragraph_refs':db.decode(c['paragraph_refs'],[]),
                         'page_numbers':db.decode(c['pages'],[]),'score':float(scores[i])})
    if rerank and evidence:
        evidence=_rerank(query,evidence)
    latency_ms=round((time.perf_counter()-t0)*1000,2)
    logger.info('run=%s retrieval strategy=%s papers=%d chunks=%d hits=%d ms=%.1f rerank=%s top=%s',
                run_id or '-',strategy,len(paper_ids),len(chunks),len(evidence[:top_k]),latency_ms,rerank,
                [(x['chunk_id'],round(x['score'],3)) for x in evidence[:3]])
    return {'query':query,'strategy':strategy,'evidences':evidence[:top_k],
            'latency_ms':latency_ms,'rerank_applied':rerank and bool(evidence)}


def acronym_in_question(question):
    """Detect explicit Latin initialisms without treating adjacent Chinese as letters."""
    return list(dict.fromkeys(re.findall(r'(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])', question)))


def explicit_expansion(text, acronym):
    """Only count a long form in the same passage: e.g. Chunk Coverage (CC)."""
    a=re.escape(acronym)
    pattern=rf'([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){{0,8}})\s*\(\s*{a}\s*\)'
    for match in re.finditer(pattern,text,re.I):
        terms=re.findall(r'[A-Za-z]+',match.group(1))
        for size in range(1,min(8,len(terms))+1):
            candidate=terms[-size:]
            if ''.join(t[0].upper() for t in candidate)==acronym:
                return ' '.join(candidate)
    # Accept explicitly reversed definitions, without guessing from unrelated words.
    reverse=rf'(?<![A-Za-z]){a}\s*\(\s*([A-Za-z]+(?:\s+[A-Za-z]+){{0,7}})\s*\)'
    for match in re.finditer(reverse,text,re.I):
        terms=match.group(1).split()
        for size in range(1,min(8,len(terms))+1):
            candidate=terms[:size]
            if ''.join(t[0].upper() for t in candidate)==acronym:
                return ' '.join(candidate)
    return None


def find_definition_evidence(question, paper_ids, limit=2):
    """Cheap local scan for explicit acronym expansions, used only by Adaptive RAG.

    Does not infer a definition from a keyword hit, and never accesses QASPER gold labels.
    """
    acronyms=acronym_in_question(question)
    if not acronyms or not paper_ids:
        return []
    placeholders=','.join('?' for _ in paper_ids)
    rows=db.rows(f"""SELECT c.* FROM chunks c JOIN papers p ON p.id=c.paper_id
                    WHERE c.paper_id IN ({placeholders}) AND p.owner_id=? AND p.status='ready'
                    ORDER BY c.paper_id,c.seq""",(*paper_ids,settings.owner_id))
    matches=[]
    for c in rows:
        expansion=None
        for a in acronyms:
            expansion=explicit_expansion(c['content'],a)
            if expansion: break
        if expansion:
            matches.append({'evidence_id':c['id'],'source_type':'paper','paper_id':c['paper_id'],
                            'chunk_id':c['id'],'section':c['section'],'content':c['content'],
                            'paragraph_refs':db.decode(c['paragraph_refs'],[]),
                            'page_numbers':db.decode(c['pages'],[]),'score':None})
            if len(matches)>=limit:break
    return matches


def get_passage(chunk_id,allowed_papers):
    if not allowed_papers: return None
    c=db.one('''SELECT c.* FROM chunks c JOIN papers p ON p.id=c.paper_id
                WHERE c.id=? AND p.owner_id=? AND p.status='ready' ''',(chunk_id,settings.owner_id))
    if not c or c['paper_id'] not in allowed_papers: return None
    return {'evidence_id':c['id'],'paper_id':c['paper_id'],'chunk_id':c['id'],
            'section':c['section'],'content':c['content'],
            'paragraph_refs':db.decode(c['paragraph_refs'],[]),
            'page_numbers':db.decode(c['pages'],[])}
