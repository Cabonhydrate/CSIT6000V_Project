"""Lossless QASPER paragraph IDs and page-preserving PDF ingestion."""
import json
from pathlib import Path
from uuid import uuid4
from .. import db
from ..logging_setup import get_logger

logger=get_logger("ingest")


def paragraphs_to_chunks(paper_id, sections, target_chars=1700, overlap=1):
    """Paragraph-aware chunks; preserve original evidence refs (no invented page numbers)."""
    out = []
    for sec in sections:
        para = [p for p in sec['paragraphs'] if p.get('text','').strip()]
        start = 0
        while start < len(para):
            selected = []
            chars = 0
            j = start
            while j < len(para):
                p = para[j]
                if selected and chars + len(p['text']) > target_chars:
                    break
                selected.append(p)
                chars += len(p['text'])
                j += 1
            refs = [p['id'] for p in selected]
            pages = sorted(set(page for p in selected for page in p.get('pages',[])))
            out.append({'id':f'{paper_id}:c{len(out)}', 'paper_id':paper_id,
                        'seq':len(out), 'section':sec['name'],
                        'content':'\n\n'.join(p['text'] for p in selected),
                        'paragraph_refs':refs, 'pages':pages})
            if j >= len(para):
                break
            start = max(start + 1, j - overlap)
    return out


def qasper_answer(answer):
    a = answer.get('answer', answer)
    if a.get('unanswerable'):
        return 'Unanswerable', 'unanswerable'
    if a.get('yes_no') is not None:
        return ('Yes' if a['yes_no'] else 'No'), 'yes_no'
    if a.get('free_form_answer'):
        return str(a['free_form_answer']), 'free_form'
    if a.get('extractive_spans'):
        return ' '.join(a['extractive_spans']), 'extractive'
    return '', 'unknown'


def parse_qasper_record(paper_id, record, split):
    """Accept QASPER's canonical raw JSON and HuggingFace row-style records."""
    source_id = str(record.get('id', paper_id))
    sections = []
    abstract = record.get('abstract', '')
    if isinstance(abstract, list):
        abstract = '\n'.join(map(str, abstract))
    if abstract:
        sections.append({'name':'Abstract', 'paragraphs':[{'id':'abstract-p0', 'text':str(abstract), 'pages':[]}]})
    full = record.get('full_text', [])
    if isinstance(full, dict):
        full = [{'section_name':n, 'paragraphs':p} for n,p in zip(full.get('section_name',[]),full.get('paragraphs',[]))]
    paragraph_lookup = {}
    for si,section in enumerate(full):
        text_paras = section.get('paragraphs',[])
        if isinstance(text_paras, str):
            text_paras = [text_paras]
        paras=[]
        for pi,p in enumerate(text_paras):
            txt = str(p).strip()
            if not txt:
                continue
            pid = f's{si}-p{pi}'
            paras.append({'id':pid,'text':txt,'pages':[]})
            paragraph_lookup.setdefault(' '.join(txt.split()), []).append(pid)
        sections.append({'name':section.get('section_name') or f'Section {si+1}', 'paragraphs':paras})
    cases=[]
    for qi,qa in enumerate(record.get('qas', [])):
        refs=[]
        answers=[]
        typ='unknown'
        for annotation in qa.get('answers', []):
            a, t = qasper_answer(annotation)
            if a: answers.append(a)
            if typ=='unknown': typ=t
            raw = annotation.get('answer',annotation)
            for evidence in raw.get('evidence',[]) or []:
                normalized=' '.join(str(evidence).split())
                matches=paragraph_lookup.get(normalized,[])
                if not matches:
                    # Annotation can include offsets/prefixes; unresolved refs stay unresolved.
                    matches=[pid for text,ids in paragraph_lookup.items() if normalized and (normalized in text or text in normalized) for pid in ids]
                refs.extend(matches)
        cases.append({'question_id':str(qa.get('question_id') or f'{source_id}-q{qi}'),
                      'question':qa.get('question',''), 'paper_id':paper_id,
                      'split':split,'answers':list(dict.fromkeys(answers)),
                      'evidence_refs':list(dict.fromkeys(refs)), 'answer_type':typ})
    return sections,cases


def insert_paper(paper_id, title, source_type, sections, cases=None, document_path=None, parser=None):
    chunks = paragraphs_to_chunks(paper_id,sections)
    if not chunks:
        raise ValueError('No usable paragraphs found')
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,document_path,status,parser,created_at) VALUES(?,?,?,?,?,?,?,?)',
                  (paper_id,db.settings.owner_id,title,source_type,document_path,'ready',parser,db.now()))
        c.executemany('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                      [(x['id'],paper_id,x['seq'],x['section'],x['content'],db.encode(x['paragraph_refs']),db.encode(x['pages'])) for x in chunks])
        if cases:
            c.executemany('INSERT OR REPLACE INTO eval_cases(question_id,paper_id,split,question,answers_json,evidence_json,answer_type) VALUES(?,?,?,?,?,?,?)',
                          [(x['question_id'], paper_id,x['split'],x['question'],db.encode(x['answers']),db.encode(x['evidence_refs']),x['answer_type']) for x in cases])
    return len(chunks)


def import_qasper_file(path: Path, split='dev', limit=None):
    """Supports mapping {paper_id:paper} or list of per-paper records."""
    data=json.loads(path.read_text(encoding='utf-8'))
    records = data.items() if isinstance(data,dict) else ((r.get('id') or str(i),r) for i,r in enumerate(data))
    result={'imported':0,'skipped':0,'questions':0,'chunks':0}
    for source_id,record in records:
        if limit is not None and result['imported'] >= limit: break
        paper_id='qasper-'+str(source_id)
        if db.one('SELECT id FROM papers WHERE id=?',(paper_id,)):
            result['skipped']+=1
            continue
        sections,cases=parse_qasper_record(paper_id,record,split)
        try:
            n=insert_paper(paper_id,record.get('title','Untitled paper'),'qasper',sections,cases,parser='qasper-json')
        except ValueError:
            result['skipped']+=1
            continue
        result['imported']+=1
        result['questions']+=len(cases)
        result['chunks']+=n
    return result


def parse_pdf(pdf_path):
    """Docling first, PyMuPDF fallback. Page metadata comes from parser, never inferred."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        DocumentConverter=None
    if DocumentConverter:
        try:
            doc=DocumentConverter().convert(str(pdf_path)).document
            sections=[]
            current={'name':'Document','paragraphs':[]}
            counter=0
            for item,_ in doc.iterate_items():
                text=str(getattr(item,'text','') or '').strip()
                if not text: continue
                label=str(getattr(item,'label','')).lower()
                if 'heading' in label or 'title' in label or 'section_header' in label:
                    if current['paragraphs']: sections.append(current)
                    current={'name':text,'paragraphs':[]}
                else:
                    pages=sorted({int(p.page_no) for p in getattr(item,'prov',[]) if getattr(p,'page_no',None)})
                    current['paragraphs'].append({'id':f'doc-p{counter}','text':text,'pages':pages})
                    counter+=1
            if current['paragraphs']:sections.append(current)
            if sections:return sections,'docling'
        except Exception:
            # Conversion may fail on malformed/unsupported PDFs; use extractive fallback.
            pass
    # A lightweight local fallback is useful for a CPU-only development machine.
    import fitz
    sections=[]
    with fitz.open(pdf_path) as doc:
        for n,page in enumerate(doc):
            text=page.get_text('text').strip()
            if text:
                parts=[s.strip() for s in text.split('\n\n') if s.strip()]
                sections.append({'name':f'Page {n+1}', 'paragraphs':[
                    {'id':f'page{n+1}-p{i}', 'text':p, 'pages':[n+1]} for i,p in enumerate(parts)]})
    return sections,'pymupdf'


def process_pdf(paper_id,pdf_path):
    logger.info('paper=%s ingest started file=%s',paper_id,pdf_path.name)
    try:
        sections,parser=parse_pdf(pdf_path)
        chunks=paragraphs_to_chunks(paper_id,sections)
        if not chunks:
            raise ValueError('No extractable PDF text')
        with db.connect() as c:
            c.execute('DELETE FROM chunks WHERE paper_id=?',(paper_id,))
            c.executemany('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                [(x['id'],paper_id,x['seq'],x['section'],x['content'],db.encode(x['paragraph_refs']),db.encode(x['pages'])) for x in chunks])
            c.execute('UPDATE papers SET status=?, parser=?, error=NULL WHERE id=?',('ready',parser,paper_id))
        # Tables are supplemental; a parsing issue must not make a readable PDF unusable.
        try:
            from .tables import extract_pdf_tables, store_tables
            store_tables(paper_id, extract_pdf_tables(pdf_path))
        except Exception:
            logger.exception('paper=%s table indexing skipped',paper_id)
        logger.info('paper=%s ingest ready parser=%s chunks=%d',paper_id,parser,len(chunks))
    except Exception as ex:
        logger.exception('paper=%s ingest failed',paper_id)
        db.execute('UPDATE papers SET status=?, error=? WHERE id=?',('failed',str(ex)[:500],paper_id))
        # Background ingestion must persist failure instead of crashing the HTTP response.
        return
