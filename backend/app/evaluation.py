"""Offline QASPER evaluation, kept separate from the agent and retrieval code."""
from collections import Counter
import re
from . import db


def normalized_words(text):
    t=re.sub(r'[^a-z0-9\s]',' ',text.lower())
    return [w for w in t.split() if w not in ('a','an','the')]


def f1(pred,reference):
    p=Counter(normalized_words(pred))
    r=Counter(normalized_words(reference))
    shared=sum((p&r).values())
    if not p and not r:return 1.0
    if not shared:return 0.0
    precision=shared/sum(p.values())
    recall=shared/sum(r.values())
    return 2*precision*recall/(precision+recall)


def answer_f1(pred,answers):
    return max((f1(pred,ref) for ref in answers),default=0.0)


def evidence_metrics(pred_refs,gold_refs):
    predicted=set(pred_refs)
    gold=set(gold_refs)
    if not gold:return {'precision':None,'recall':None,'f1':None}  # Unresolved gold != confirmed negative
    tp=len(predicted&gold)
    p=tp/len(predicted) if predicted else 0.0
    r=tp/len(gold)
    return {'precision':p,'recall':r,'f1':(2*p*r/(p+r) if p+r else 0.0)}


def list_cases(split='dev',limit=10):
    cases=db.rows('SELECT * FROM eval_cases WHERE split=? ORDER BY question_id LIMIT ?',(split,limit))
    for c in cases:
        c['answers']=db.decode(c.pop('answers_json'),[])
        c['gold_evidence_refs']=db.decode(c.pop('evidence_json'),[])
    return cases
