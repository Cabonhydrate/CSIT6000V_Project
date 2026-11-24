"""Lossless QASPER paragraph IDs and page-preserving PDF ingestion."""
import json
import re
from pathlib import Path
from uuid import uuid4
from .. import db
from ..logging_setup import get_logger

logger=get_logger("ingest")


def _normalize_pdf_text(text: str) -> str:
    """Normalize *extracted* PDF whitespace; leave the PDF bytes untouched.

    Keep newlines and paragraph breaks, which are structural evidence. Avoid
    reflowing table rows, equations or line-wrapped scientific terminology.
    """
    text = str(text).replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\u00a0', ' ').replace('\u200b', '').replace('\ufeff', '')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# Highest priority first. A boundary is expressed as a cut OFFSET in the
# original string so that splitting cannot silently delete a punctuation mark.
_PDF_SPLIT_BOUNDARIES = (
    re.compile(r'\n{2,}'),                 # paragraph
    re.compile(r'\n'),                     # line
    re.compile(r'(?<=[.!?。！？])(?=\s|$)'), # sentence, not decimal points
    re.compile(r'(?<=[;；])(?=\s|$)'),     # clause
    re.compile(r'(?<=[,:，、：])(?=\s|$)'), # secondary clause
    re.compile(r'(?<=\S)(?=\s)'),        # final word-boundary fallback
)


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Recursively choose paragraph > line > sentence > clause > word cuts.

    The length limit is a hard cap. Splits are LOSSLESS relative to the already
    normalized text: neither sentence punctuation nor whitespace disappears.
    The algorithm does not modify QASPER source paragraphs or their gold IDs.
    """
    if max_chars < 16:
        raise ValueError('max_chars must be >= 16')
    if len(text) <= max_chars:
        return [text] if text else []
    out = []
    remainder = text
    while len(remainder) > max_chars:
        low = max(1, int(max_chars * .55))
        cut = None
        window = remainder[:max_chars]
        for boundary in _PDF_SPLIT_BOUNDARIES:
            valid = [m.end() for m in boundary.finditer(window)
                     if low <= m.end() <= max_chars]
            if valid:
                cut = max(valid)
                break
        if cut is None:
            cut = max_chars  # uninterrupted equations / identifiers
        assert 0 < cut <= max_chars
        out.append(remainder[:cut])
        remainder = remainder[cut:]
    if remainder:
        # Merge a short tail only when it fits in the same bounded chunk.
        if out and len(out[-1]) + len(remainder) <= max_chars:
            out[-1] += remainder
        else:
            out.append(remainder)
    assert all(0 < len(t) <= max_chars for t in out)
    assert ''.join(out) == text
    return out


def _pdf_heading(text: str) -> bool:
    """Conservative single-block recognition of English research headings.

    Don't infer a heading from an arbitrary wrapped paragraph or table row.
    Document layout boundaries are used when Docling supplies heading labels;
    this recognizer is for PyMuPDF's text-block fallback only.
    """
    line = text.strip()
    if '\n' in line or len(line) > 105 or not line:
        return False
    if re.fullmatch(r'(?i)(?:abstract|introduction|background|related work|'
                    r'experimental setup|experiments|methodology|methods|'
                    r'implementation|discussion|limitations|conclusions?|references|appendix)',line):
        return True
    return bool(re.fullmatch(r'(?i)(?:[1-9]\d?(?:\.\d+){0,3}\.?\s+|'
                             r'(?:appendix\s+)?[A-F]\.?\s+)'
                             r'[A-Za-z][A-Za-z0-9 /(),:&\-]{3,90}',line))


def paragraphs_to_chunks(paper_id, sections, target_chars=1700, overlap=1,
                         *, split_oversized=False):
    """Pack paragraph-aware chunks, retaining real page and paragraph refs.

    PDF: normalize, recursively split oversized blocks and merge adjacent
    short fragments in the SAME structural section. Respect hard character
    bounds including join separators. QASPER: keep the original paragraph text
    and gold evidence mapping unmodified (even if a paragraph is oversized).
    """
    if target_chars < 16:
        raise ValueError('target_chars must be >= 16')
    if overlap < 0:
        raise ValueError('overlap must not be negative')
    out = []
    for sec in sections:
        original = [p for p in sec['paragraphs'] if str(p.get('text','')).strip()]
        paragraphs = []
        for p in original:
            text = _normalize_pdf_text(p['text']) if split_oversized else p['text']
            if not text:
                continue
            fragments = (_split_long_text(text,target_chars)
                         if split_oversized and len(text) > target_chars
                         else [text])
            paragraphs.extend({**p,'text': fragment, '_fragment_of': p['id'] if len(fragments)>1 else None}
                              for fragment in fragments)
        start = 0
        while start < len(paragraphs):
            selected = []
            chars = 0
            j = start
            while j < len(paragraphs):
                p = paragraphs[j]
                addition = len(p['text']) + (2 if selected else 0)
                if selected and chars + addition > target_chars:
                    break
                selected.append(p)
                chars += addition
                j += 1
            assert selected
            refs = list(dict.fromkeys(p['id'] for p in selected))
            pages = sorted(set(page for p in selected for page in p.get('pages',[])))
            content = '\n\n'.join(p['text'] for p in selected)
            assert not split_oversized or len(content) <= target_chars
            out.append({'id':f'{paper_id}:c{len(out)}','paper_id':paper_id,
                        'seq':len(out),'section':sec['name'],
                        'content':content,'paragraph_refs':refs,'pages':pages})
            if j >= len(paragraphs):
                break
            next_start = max(start + 1, j - overlap)
            if split_oversized:
                # Never repeat one part of an overlong original paragraph as
                # an overlapping "paragraph". It would duplicate sentences and
                # undermine the one-retrieved-chunk grounding semantics.
                if (paragraphs[j-1]['id'] == paragraphs[j]['id'] or
                        paragraphs[j-1].get('_fragment_of')):
                    next_start = j
                # A single overlap paragraph is only valuable if there is
                # another non-overlapping segment after it, not a tail-only
                # duplicate of an already indexed chunk.
                if next_start < j and (j >= len(paragraphs) or
                        len(paragraphs[next_start]['text']) + 2 +
                        len(paragraphs[j]['text']) > target_chars):
                    next_start = j
            start = next_start
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
            # PyMuPDF text BLOCKS are closer to real paragraphs than a whole
            # page of plain text. Sections remain page-scoped so a citation
            # never claims an invented page number.
            blocks = [block[4] for block in page.get_text('blocks',sort=True)
                      if len(block) > 6 and block[6] == 0
                      and isinstance(block[4],str) and block[4].strip()]
            if not blocks:
                text=page.get_text('text')
                if text.strip():
                    blocks=[text]
            current={'name':f'Page {n+1}','paragraphs':[]}
            for i,raw in enumerate(blocks):
                text=_normalize_pdf_text(raw)
                if not text:
                    continue
                lines=text.split('\n')
                heading=''
                body=text
                # Academic PDFs sometimes put a section number and its heading
                # in the SAME text block as the first paragraph, e.g.:
                # "3.1\nInterpretation ...\nFrom a software ...".
                # Only recognize the first two lines if they look like a real
                # numbered heading, not page running titles or table cells.
                if (len(lines)>=2 and re.fullmatch(r'[1-9]\d?(?:\.\d+){0,3}\.?',lines[0])
                        and 3<=len(lines[1])<=95 and re.match(r'^[A-Za-z]',lines[1])):
                    candidate=lines[0]+' '+lines[1]
                    if _pdf_heading(candidate):
                        heading=candidate
                        body='\n'.join(lines[2:]).strip()
                elif _pdf_heading(lines[0]) and (len(lines)==1 or lines[0].strip()!=text):
                    heading=lines[0]
                    body='\n'.join(lines[1:]).strip()
                elif _pdf_heading(text):
                    heading=text
                    body=''
                if heading:
                    if current['paragraphs']:
                        sections.append(current)
                    current={'name':f'Page {n+1} / {heading[:95]}','paragraphs':[]}
                    current['paragraphs'].append({'id':f'page{n+1}-p{i}-heading',
                                                  'text':heading,'pages':[n+1]})
                if body:
                    current['paragraphs'].append({'id':f'page{n+1}-p{i}',
                                                  'text':body,'pages':[n+1]})
            if current['paragraphs']:
                sections.append(current)
    return sections,'pymupdf'


def process_pdf(paper_id,pdf_path):
    logger.info('paper=%s ingest started file=%s',paper_id,pdf_path.name)
    try:
        sections,parser=parse_pdf(pdf_path)
        chunks=paragraphs_to_chunks(paper_id,sections,split_oversized=True)
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
