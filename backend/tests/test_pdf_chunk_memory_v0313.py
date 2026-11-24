"""Regressions for PDF chunk boundaries, source previews and learner gaps.

No live model, web access, production database or private PDF is needed.
"""
import asyncio

from app import db
from app.agent.learning_context import _stated_learning_gap, make_learning_context
from app.agent.workflow import _source_quote, AgentRunner
from app.knowledge.excerpts import source_excerpt
from app.knowledge.ingest import _split_long_text, paragraphs_to_chunks, parse_pdf
from test_english_preferences_and_memory import create_run, MemoryAwareModel
from test_day1_learning_memory import fake_langgraph


def test_long_source_paragraph_is_split_without_loss():
    # One source paragraph may exceed the requested chunk size by many times.
    original = ('The retriever selects document chunks. ' * 140).strip()
    fragments = _split_long_text(original, 700)
    assert len(fragments) > 2
    assert ''.join(fragments) == original
    assert all(0 < len(f) <= 700 for f in fragments)
    sections=[{'name':'Page 4', 'paragraphs':[
        {'id':'page4-p0', 'text':original, 'pages':[4]}]}]
    chunks=paragraphs_to_chunks('paper',sections,target_chars=700,
                                overlap=0,split_oversized=True)
    assert len(chunks) > 2
    assert all(len(c['content']) <= 700 for c in chunks)
    assert all(c['pages'] == [4] for c in chunks)
    assert all(c['paragraph_refs'] == ['page4-p0'] for c in chunks)
    assert ''.join(c['content'] for c in chunks)==original


def test_qasper_default_does_not_change_long_paragraph_identity():
    long_text='abc def. ' * 260
    sections=[{'name':'Abstract','paragraphs':[
        {'id':'original-gold-ref','text':long_text,'pages':[]}]}]
    chunks=paragraphs_to_chunks('qasper-id',sections,target_chars=500)
    assert len(chunks)==1
    assert chunks[0]['content']==long_text
    assert chunks[0]['paragraph_refs']==['original-gold-ref']


def test_pdf_parser_uses_separate_text_blocks_with_grounded_page_refs(tmp_path):
    import fitz
    doc=fitz.open()
    page=doc.new_page()
    page.insert_text((65,90),'First separate paragraph about retrieval testing.',fontsize=11)
    page.insert_text((65,170),'Second separate paragraph about document coverage.',fontsize=11)
    filename=tmp_path/'two-blocks.pdf'
    doc.save(filename)
    doc.close()
    sections,parser=parse_pdf(filename)
    assert parser in ('pymupdf','docling')
    assert len(sections)>=1
    if parser=='pymupdf':
        assert len(sections[0]['paragraphs'])>=2
        assert all(p['pages']==[1] for p in sections[0]['paragraphs'])
    joined=' '.join(p['text'] for section in sections for p in section['paragraphs'])
    assert 'First separate paragraph' in joined
    assert 'Second separate paragraph' in joined


def test_excerpt_is_a_literal_substring_and_ends_near_sentence_boundary():
    source=('Unrelated introductory paragraph. ' * 24 +
            'Chunk Coverage measures the fraction of unique corpus chunks retrieved. '
            'It does not measure output correctness. ' +
            'Unrelated later background paragraph. ' * 20)
    preview=source_excerpt(source,250,focus='What does Chunk Coverage measure?')
    assert preview in source
    assert 'Chunk Coverage measures' in preview
    assert len(preview) <= 250
    assert preview.endswith(('.', '?', '!', '\n')) or len(preview)==250
    item={'content':source}
    assert _source_quote(item,'What does Chunk Coverage measure?',chars=250) in source


def test_saved_learning_gap_is_extracted_as_context_data():
    note='I already understand the RAG retrieval pipeline, but I am not familiar with set union and how coverage is calculated.'
    assert 'set union' in _stated_learning_gap(note)
    memory={'domain':'RAG','topic':'Chunk Coverage','mastery':'studied','content':note}
    enabled=make_learning_context({'language':'en','explanation_level':'beginner'},
                                  [memory],include_memories=True)
    disabled=make_learning_context({'language':'en','explanation_level':'beginner'},
                                   [memory],include_memories=False)
    assert 'Learner-stated difficulty' in enabled
    assert 'set union' in enabled
    assert 'Learner-stated difficulty' not in disabled
    assert 'set union' not in disabled
    assert 'NOT paper evidence' in enabled


def test_learning_gap_reaches_generation_but_disappears_when_memory_off(monkeypatch):
    fake_langgraph(monkeypatch)
    question=('Explain what Chunk Coverage measures and why repeated retrieval '
              'does not increase coverage.')
    seen=[]
    for use_memory in (True,False):
        rid,req=create_run(monkeypatch,use_memory=use_memory,teaching=True,
                           question=question)
        db.execute('UPDATE memories SET content=? WHERE owner_id=?',
                   ('I already understand retrieval but am not familiar with set union.',
                    db.settings.owner_id))
        model=MemoryAwareModel()
        asyncio.run(AgentRunner(model).run(rid,req))
        metrics=db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(rid,))['metrics'])
        recorded='\n'.join(str(call) for call in model.chat_calls+model.stream_calls)
        seen.append((recorded,metrics))
    assert 'Learner-stated difficulty' in seen[0][0]
    assert 'set union' in seen[0][0]
    assert seen[0][1]['learning_memory_applied'] is True
    assert 'Learner-stated difficulty' not in seen[1][0]
    assert seen[1][1]['learning_memory_applied'] is False


def test_recursive_split_respects_english_structure_before_sentence_and_words():
    # A paragraph boundary at ~60% outranks a sentence near the window end.
    text=('A paragraph about a retrieval method with details. ' * 9 +
          '\n\n' + 'Next paragraph discusses a distinct experiment. ' * 12)
    fragments=_split_long_text(text,550)
    assert ''.join(fragments)==text
    assert len(fragments)>1
    assert fragments[0].endswith('\n\n')
    assert all(len(f)<=550 for f in fragments)


def test_recursive_split_keeps_sentence_and_words_when_paragraph_missing():
    sentence='The paper measures the fraction of documents retrieved in a test suite. '
    fragments=_split_long_text(sentence*33,480)
    assert ''.join(fragments)==sentence*33
    assert all(frag.endswith(('. ','.')) for frag in fragments)
    assert all(0<len(f)<=480 for f in fragments)


def test_normalization_handles_windows_mac_unix_and_controls():
    from app.knowledge.ingest import _normalize_pdf_text
    assert _normalize_pdf_text('Abstract\r\n  We  test.\r\n\rA new section.\x00\n\n\n') == (
        'Abstract\nWe test.\n\nA new section.')


def test_pdf_english_headings_are_detected_without_capturing_prose():
    from app.knowledge.ingest import _pdf_heading
    for text in ('Abstract','1 Introduction','3.2 Experiment Setup',
                 '2.1.3 Detailed Results','References','Conclusion'):
        assert _pdf_heading(text), text
    for text in ('This is a normal sentence with a conclusion in it.',
                 '1.1 2.2 3.3','Results\nThe table provides...','1234'):
        assert not _pdf_heading(text), text


def test_short_pdf_blocks_merge_but_no_fragment_crosses_max_chars():
    paragraphs=[{'id':f'p{i}','text':t,'pages':[8]} for i,t in enumerate([
        'First short paragraph.', 'A second short paragraph.',
        'A long paragraph. '*55, 'A very short conclusion.'])]
    chunks=paragraphs_to_chunks('test-paper',
        [{'name':'Results','paragraphs':paragraphs}],target_chars=250,
        overlap=0,split_oversized=True)
    assert chunks[0]['paragraph_refs']==['p0','p1']
    assert chunks[0]['content']=='First short paragraph.\n\nA second short paragraph.'
    assert all(len(x['content'])<=250 for x in chunks)
    assert all(x['pages']==[8] for x in chunks)
    assert 'A very short conclusion.' in chunks[-1]['content']


def test_oversize_fragments_are_not_repeated_by_overlap():
    paragraph='The same paragraph has many separate sentences. '*61
    chunks=paragraphs_to_chunks('paper',[{'name':'A','paragraphs':[
        {'id':'source-p7','pages':[7],'text':paragraph}]}],
        target_chars=500,overlap=1,split_oversized=True)
    assert ''.join(chunk['content'] for chunk in chunks)==paragraph.strip()
    assert all(chunk['paragraph_refs']==['source-p7'] for chunk in chunks)


def test_no_citation_is_falsely_claimed_as_full_pdf_page():
    # One indexed chunk is not a full PDF page. The API resolves its complete
    # content via the preserved paper/chunk ID; the frontend labels quote as preview.
    from app.knowledge.retrieval import get_passage
    from uuid import uuid4
    paper=uuid4().hex
    payload='Long grounded paragraph. '*65
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) '
                  'VALUES(?,?,?,?,?,?)', (paper,db.settings.owner_id,'Test paper','pdf',
                                          'ready',db.now()))
        c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) '
                  'VALUES(?,?,?,?,?,?,?)',
                  (paper+':c0',paper,0,'Results',payload,db.encode(['p2']),db.encode([3])))
    passage=get_passage(paper+':c0',[paper])
    assert passage['content']==payload
    assert passage['page_numbers']==[3]
    assert get_passage(paper+':c0',['not-authorized']) is None


def test_multi_question_with_memory_keeps_teaching_focus_in_system_prompt(monkeypatch):
    fake_langgraph(monkeypatch)
    question=('What does Chunk Coverage measure? '
              'Why does retrieving the same chunk twice not increase coverage?')
    rid,req=create_run(monkeypatch,use_memory=True,teaching=True,question=question)
    db.execute('UPDATE memories SET content=? WHERE owner_id=?',
               ('I understand retrieval but I am not familiar with set union.',
                db.settings.owner_id))
    model=MemoryAwareModel()
    asyncio.run(AgentRunner(model).run(rid,req))
    calls=model.chat_calls
    assert any('Use the saved learner context' in c[0]['content'] for c in calls)
    assert any('Learner-stated difficulty' in str(c) for c in calls)
    stats=db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(rid,))['metrics'])
    assert stats['learning_memory_applied'] is True
