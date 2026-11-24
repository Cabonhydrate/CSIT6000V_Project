'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import MarkdownAnswer from '../components/MarkdownAnswer';
import { BookOpen, MessageCircle, Settings2, UploadCloud, Plus, Trash2, Send, Square, Layers3, Sparkles, Activity, RotateCcw, History, Search, Pencil, Check, X, List } from 'lucide-react';
import {api, API, jsonInit, type Paper,type Citation,type Message,type Conversation,type Profile,type Memory} from '../lib/api';
const PdfReader = dynamic(()=>import('../components/PdfReader'),{ssr:false});

type Mode='basic'|'hybrid'|'adaptive';
type Scope='paper'|'general';
type Tab='workspace'|'learning'|'conversations';
const defaultProfile:Profile={language:'en',explanation_level:'beginner',domains:[],goals:[],memory_enabled:true};
// Read-only learning-note questions do not require an uploaded paper.
const isMemoryListQuestion=(q:string)=>/^(?:please\s+)?(?:list|show|display|view|summari[sz]e|review)\s+(?:(?:all|the|my|saved|confirmed|learning|study)\s+)*(?:notes?|memories|memory|learning progress|study progress)(?:\s+(?:i have saved|that i saved|for me))?\s*[.!?]?$/i.test(q.trim()) || /^(?:请)?(?:列出|显示|查看|展示|总结|回顾|告诉我)\s*(?:我|我的|所有|全部|已保存的|学习|确认的|相关的|\s)*(?:学习记录|学习笔记|学习记忆|记忆|笔记|记录)\s*[。！？?]?$/.test(q.trim());
export default function Home() {
  const [tab,setTab]=useState<Tab>('workspace');
  const [papers,setPapers]=useState<Paper[]>([]);
  const [selected,setSelected]=useState<string[]>([]);
  const [openPaper,setOpenPaper]=useState<string|null>(null);
  const [page,setPage]=useState(1);
  const [highlight,setHighlight]=useState('');
  const [citationChunkId,setCitationChunkId]=useState<string|null>(null);
  const [conversations,setConversations]=useState<Conversation[]>([]);
  const [conversation,setConversation]=useState<string|null>(null);
  const [conversationSearch,setConversationSearch]=useState('');
  const [renaming,setRenaming]=useState<string|null>(null);
  const [titleDraft,setTitleDraft]=useState('');
  const [messages,setMessages]=useState<Message[]>([]);
  const [question,setQuestion]=useState('');
  const [mode,setMode]=useState<Mode>('adaptive');
  const [scope,setScope]=useState<Scope>('paper');
  const [teaching,setTeaching]=useState(true);
  const [allowWeb,setAllowWeb]=useState(false);
  const [busy,setBusy]=useState(false);
  const [runId,setRunId]=useState<string|null>(null);
  const [lastRunId,setLastRunId]=useState<string|null>(null);
  const [steps,setSteps]=useState<string[]>([]);
  const [error,setError]=useState('');
  const [modelAvailable,setModelAvailable]=useState<boolean|null>(null);
  const [profile,setProfile]=useState<Profile>(defaultProfile);
  const [profileDirty,setProfileDirty]=useState(false);
  const [profileSaved,setProfileSaved]=useState(false);
  const [selectedMemoryCount,setSelectedMemoryCount]=useState(0);
  const [memories,setMemories]=useState<Memory[]>([]);
  const [newMemory,setNewMemory]=useState({domain:'',topic:'',content:'',mastery:'mentioned'});
  const [editingMemory,setEditingMemory]=useState<Memory|null>(null);
  const streamRef=useRef<EventSource|null>(null);
  const bottomRef=useRef<HTMLDivElement|null>(null);
  const fileRef=useRef<HTMLInputElement|null>(null);

  const loadPapers=useCallback(async()=>{
    const data=await api<Paper[]>('/api/papers');setPapers(data);
  },[]);
  const loadConversations=useCallback(async()=>setConversations(await api<Conversation[]>('/api/conversations')),[]);
  const loadLearning=useCallback(async()=>{
    const [p,m]=await Promise.all([api<Profile>('/api/learning/profile'),api<Memory[]>('/api/learning/memories')]);
    setProfile(p);setMemories(m);setProfileDirty(false);
  },[]);
  useEffect(()=>{
    loadPapers().catch(e=>setError(e.message));loadConversations().catch(()=>{});
    loadLearning().catch(()=>{});
    api<{model_available:boolean}>('/api/health').then(x=>setModelAvailable(x.model_available)).catch(()=>setModelAvailable(false));
    return ()=>{streamRef.current?.close()};
  },[loadPapers,loadConversations,loadLearning]);
  // Poll only while PDF ingestion is active; ready papers do not need repeated requests.
  useEffect(()=>{
    if(!papers.some(p=>p.status==='processing'||p.status==='pending'))return;
    const timer=setInterval(()=>loadPapers().catch(()=>{}),3000);
    return ()=>clearInterval(timer);
  },[papers,loadPapers]);
  // Modern Chromium may return a Promise from scrollIntoView().
  // An effect must not return that Promise as React's cleanup function.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({behavior:'smooth'});
  }, [messages, steps]);

  async function upload(file:File){
    setError('');
    const form=new FormData();form.append('file',file);
    try{
      const p=await api<Paper>('/api/papers',{method:'POST',body:form});
      await loadPapers();setSelected(s=>[...new Set([...s,p.id])]);setOpenPaper(p.id);
    }catch(e){setError((e as Error).message)}
  }
  function togglePaper(id:string){setSelected(s=>s.includes(id)?s.filter(x=>x!==id):s.length>=8?s:[...s,id]);setOpenPaper(id);setPage(1);setHighlight('');setCitationChunkId(null)}
  async function removePaper(id:string){
    if(!confirm('Delete this paper and its search index?'))return;
    try{await api('/api/papers/'+id,{method:'DELETE'});setSelected(s=>s.filter(x=>x!==id));
      if(openPaper===id)setOpenPaper(null);await loadPapers()}
    catch(e){setError((e as Error).message)}
  }
  async function newConversation(){
    if(busy)return;
    try{
      const c=await api<Conversation>('/api/conversations',jsonInit({title:'New conversation'}));
      setConversation(c.id);setMessages([]);setSteps([]);setSelectedMemoryCount(0);setError('');setLastRunId(null);
      setTab('workspace');await loadConversations();
    }catch(e){setError((e as Error).message)}
  }
  async function switchConversation(id:string){
    if(busy)return;
    try{
      const c=await api<{messages:Message[]}>('/api/conversations/'+id);
      setConversation(id);setMessages(c.messages);setSteps([]);setSelectedMemoryCount(0);setError('');setLastRunId(null);
      setTab('workspace');
    }catch(e){setError((e as Error).message)}
  }
  function beginRename(c:Conversation){setRenaming(c.id);setTitleDraft(c.title);setError('')}
  async function renameConversation(id:string){
    const title=titleDraft.trim();
    if(!title){setError('Conversation title cannot be empty.');return}
    try{
      await api('/api/conversations/'+id,jsonInit({title},'PATCH'));
      setRenaming(null);await loadConversations();setError('');
    }catch(e){setError((e as Error).message)}
  }
  async function deleteConversation(c:Conversation){
    if(busy)return;
    if(!window.confirm(`Delete "${c.title}" and all its messages? This cannot be undone.`))return;
    try{
      await api('/api/conversations/'+c.id,{method:'DELETE'});
      if(conversation===c.id){setConversation(null);setMessages([]);setSteps([]);setLastRunId(null)}
      if(renaming===c.id)setRenaming(null);
      await loadConversations();setError('');
    }catch(e){setError((e as Error).message)}
  }
  async function send(){
    if(!question.trim()||busy)return;
    if(scope==='paper'&&!selected.length&&!isMemoryListQuestion(question)){setError('Select at least one paper before asking.');return}
    if(scope==='paper'&&!isMemoryListQuestion(question)&&selected.some(id=>papers.find(p=>p.id===id)?.status!=='ready')){setError('Wait until selected papers are ready.');return}
    setBusy(true);setSteps([]);setSelectedMemoryCount(0);setError('');setLastRunId(null);
    const q=question.trim();setQuestion('');
    const current=[...messages,{role:'user',content:q} as Message];setMessages([...current,{role:'assistant',content:'',citations:[]}]);
    let cid=conversation;
    try{
      if(!cid){const c=await api<Conversation>('/api/conversations',jsonInit({title:q.slice(0,85)}));cid=c.id;setConversation(cid);await loadConversations()}
      const run=await api<{id:string}>('/api/agent/runs',jsonInit({conversation_id:cid,question:q,paper_ids:scope==='paper'&&!isMemoryListQuestion(q)?selected:[],scope,mode:scope==='general'?'adaptive':mode,
        teaching:teaching&&mode==='adaptive',use_memory:mode==='adaptive',allow_web:scope==='general'||(mode==='adaptive'&&allowWeb)}));
      setRunId(run.id);setLastRunId(run.id);
      const source=new EventSource(`${API}/api/agent/runs/${run.id}/events`);streamRef.current=source;
      source.addEventListener('skill_selected',(ev)=>{
        const data=JSON.parse((ev as MessageEvent).data) as {skills?:string[]};
        const selectedSkills=data.skills;
        if(selectedSkills?.length)setSteps(s=>[...s,`Skill · ${selectedSkills.join(' + ')}`]);
      });
      source.addEventListener('learning_context_selected',(ev)=>{
        const data=JSON.parse((ev as MessageEvent).data) as {count?:number};
        if(data.count)setSelectedMemoryCount(data.count);
      });
      source.addEventListener('step_started',(ev)=>{
        const data=JSON.parse((ev as MessageEvent).data);setSteps(s=>[...s,`${data.step}${data.strategy?' · '+data.strategy:''}`]);
      });
      source.addEventListener('tool_completed',(ev)=>{
        const data=JSON.parse((ev as MessageEvent).data);setSteps(s=>[...s,`${data.tool} · ${data.count} results`]);
      });
      source.addEventListener('answer_delta',(ev)=>{
        const text=JSON.parse((ev as MessageEvent).data).text as string;
        setMessages(s=>s.map((m,i)=>i===s.length-1?{...m,content:m.content+text}:m));
      });
      const finish=async()=>{
        source.close();streamRef.current=null;
        try{
          const state=await api<{answer:string;citations:Citation[];status:string;error?:string}>(`/api/agent/runs/${run.id}`);
          if(state.status==='succeeded'||state.status==='partial'){
            setMessages(s=>s.map((m,i)=>i===s.length-1?{...m,content:state.answer,citations:state.citations}:m));
            if(state.status==='partial')setError(scope==='general'?'External sources were unavailable or the answer could not be verified.':'Partial answer: at least one subquestion could not be answered from the retrieved evidence.');
          }
          else {
            setError(state.error||state.status);
            setMessages(s=>s.map((m,i)=>i===s.length-1&&m.role==='assistant'&&!m.content?{...m,content:'Answer generation failed. See the error below.'}:m));
          }
        }catch(e){setError((e as Error).message)}
        setBusy(false);setRunId(null);loadConversations().catch(()=>{});
      };
      source.addEventListener('run_completed',finish);
      source.addEventListener('run_failed',finish);
      source.addEventListener('stream_end',()=>{if(streamRef.current)finish()});
      source.onerror=()=>{source.close();streamRef.current=null;setBusy(false);setError('Event stream interrupted. Check run status or reopen the conversation.');};
    }catch(e){setBusy(false);setRunId(null);setError((e as Error).message)}
  }
  async function cancel(){if(!runId)return;try{await api(`/api/agent/runs/${runId}/cancel`,{method:'POST'})}catch(e){setError((e as Error).message)}}
  function navigate(c:Citation){
    if(c.source_url){if(/^https:\/\//.test(c.source_url))window.open(c.source_url,'_blank','noopener,noreferrer');return}
    if(c.paper_id){setOpenPaper(c.paper_id);setPage(c.page_numbers?.[0]||1);setHighlight(c.quote||'');setCitationChunkId(c.chunk_id||null)}
  }
  function changeProfile(next:Profile){setProfile(next);setProfileDirty(true);setProfileSaved(false)}
  async function saveProfile(){try{setProfile(await api<Profile>('/api/learning/profile',jsonInit(profile,'PUT')));setProfileDirty(false);setProfileSaved(true);setError('')}catch(e){setError((e as Error).message)}}
  async function addMemory(){try{
      await api('/api/learning/memories/confirm',jsonInit({...newMemory,memory_type:'progress'}));
      setNewMemory({domain:'NLP',topic:'',content:'',mastery:'mentioned'});await loadLearning();
    }catch(e){setError((e as Error).message)}}
  async function saveMemoryEdit(){
    if(!editingMemory || !editingMemory.topic.trim() || !editingMemory.content.trim())return;
    try{
      const m=editingMemory;
      await api('/api/learning/memories/'+m.id,
        jsonInit({domain:m.domain,topic:m.topic,memory_type:m.memory_type,content:m.content,mastery:m.mastery},'PATCH'));
      setEditingMemory(null);await loadLearning();setError('');
    }catch(e){setError((e as Error).message)}
  }
  async function deleteMemory(id:string){try{
    await api('/api/learning/memories/'+id,{method:'DELETE'});
    if(editingMemory?.id===id)setEditingMemory(null);
    await loadLearning();
  }catch(e){setError((e as Error).message)}}
  const currentPaper=papers.find(p=>p.id===openPaper)||null;
  const filteredConversations=conversations.filter(c=>c.title.toLowerCase().includes(conversationSearch.trim().toLowerCase()));

  return <main className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-logo"><BookOpen size={21}/></span><div><strong>Paper Tutor</strong><small>Adaptive Agentic RAG</small></div></div>
      <div className="nav">
        <button className={tab==='workspace'?'nav-selected':''} onClick={()=>setTab('workspace')}><MessageCircle size={17}/> Workspace</button>
        <button className={tab==='conversations'?'nav-selected':''} onClick={()=>setTab('conversations')}><History size={17}/> Conversations</button>
        <button className={tab==='learning'?'nav-selected':''} onClick={()=>{setTab('learning');loadLearning().catch(()=>{})}}><Settings2 size={17}/> Learning & Memory</button>
      </div>
      <div className="section-heading"><span>MY PAPERS</span><button title="Upload a PDF" onClick={()=>fileRef.current?.click()}><Plus size={17}/></button></div>
      <input ref={fileRef} type="file" accept="application/pdf,.pdf" hidden onChange={e=>{if(e.target.files?.[0])upload(e.target.files[0]);e.target.value=''}}/>
      <button className="upload" onClick={()=>fileRef.current?.click()}><UploadCloud size={18}/> Upload PDF</button>
      <div className="paper-list">
        {papers.map(p=><div key={p.id} className={'paper-item '+(selected.includes(p.id)?'selected':'')}>
          <label title={p.title}><input type="checkbox" checked={selected.includes(p.id)} onChange={()=>togglePaper(p.id)}/><span onClick={()=>{setOpenPaper(p.id);setPage(1)}} className="paper-title">{p.title}</span></label>
          <div className="paper-meta"><span className={'status '+p.status}>{p.status}</span><span>{p.source_type}</span>
            <button title="Delete paper" onClick={()=>removePaper(p.id)}><Trash2 size={13}/></button></div>
        </div>)}
        {papers.length===0&&<p className="quiet">Upload a paper or import QASPER using the backend CLI.</p>}
      </div>
      <div className="section-heading conversation-header"><span>CONVERSATIONS · {conversations.length}</span><div className="heading-actions"><button title="See all conversations" onClick={()=>setTab('conversations')}><List size={17}/></button><button title="New conversation" onClick={newConversation} disabled={busy}><Plus size={17}/></button></div></div>
      <div className="conversation-list">{conversations.slice(0,6).map(c=><button title={c.title} key={c.id} className={conversation===c.id?'active':''} onClick={()=>switchConversation(c.id)} disabled={busy}>{c.title}</button>)}
        {conversations.length>6&&<button className="all-conversations-link" onClick={()=>setTab('conversations')}>View all {conversations.length} conversations</button>}
      </div>
      <div className="sidebar-footer"><span className={'model-dot '+(modelAvailable?'online':'offline')}/>{modelAvailable?'Local model connected':'Local model offline'} <button title="Refresh model status" onClick={()=>api<{model_available:boolean}>('/api/health').then(r=>setModelAvailable(r.model_available)).catch(()=>setModelAvailable(false))}><RotateCcw size={14}/></button></div>
    </aside>
    {tab==='workspace' ? <div className="workspace">
      <section className="reader-panel"><div className="panel-heading"><Layers3 size={17}/><strong>Paper reader</strong><span>{selected.length} selected</span></div>
        {scope==='paper'?<PdfReader paper={currentPaper} page={page} highlight={highlight} chunkId={citationChunkId}/>:<div className="empty-reader"><div><strong>General Learning</strong><p>Ask about background concepts without selecting a paper. Answers use external sources with clickable links; they are not attributed to your PDFs.</p></div></div>}</section>
      <section className="chat-panel"><div className="panel-heading"><Sparkles size={18}/><strong>Research assistant</strong><span>{scope==='paper'?'Paper-grounded answers':'External knowledge learning'}</span></div>
        <div className="mode-bar"><label>Knowledge scope</label><select aria-label="Knowledge scope" value={scope} onChange={e=>{const next=e.target.value as Scope;setScope(next);if(next==='general')setMode('adaptive');setError('')}}><option value="paper">Paper QA</option><option value="general">General Learning</option></select><label>Mode</label><select value={mode} disabled={scope==='general'} onChange={e=>setMode(e.target.value as Mode)}><option value="basic">Basic RAG</option><option value="hybrid">Hybrid RAG</option><option value="adaptive">Adaptive Agentic RAG</option></select>
          <label className="toggle"><input type="checkbox" checked={teaching} onChange={e=>setTeaching(e.target.checked)} disabled={mode!=='adaptive'}/> Beginner skill</label>
          {scope==='paper'?<label className="toggle"><input type="checkbox" checked={allowWeb} onChange={e=>setAllowWeb(e.target.checked)} disabled={mode!=='adaptive'}/> Academic web</label>:<span className="scope-note">External knowledge search enabled · no PDF citations</span>}</div>
        <div className="message-list">
          {messages.length===0&&<div className="welcome"><span className="welcome-icon"><BookOpen size={28}/></span><h2>{scope==='paper'?'Explore research papers':'Learn a background concept'}</h2><p>{scope==='paper'?'Select a paper, then ask about its methods, experiments, findings or unfamiliar concepts. Sources are linked to passages.':'No paper needed. Search external knowledge and get a beginner-friendly explanation with source links.'}</p>
            <div className="suggestions">{(scope==='paper'?['Explain the main method of this paper with a simple example.','What are the main experimental findings?','What limitations do the authors discuss?']:['Explain Q-Learning in simple terms with an example.','What is self-attention?','How is reinforcement learning different from supervised learning?']).map(q=><button key={q} onClick={()=>setQuestion(q)}>{q}</button>)}</div></div>}
          {messages.map((m,i)=><div key={m.id||i} className={'message '+m.role}>
            <div className="message-role">{m.role==='user'?'You':'Paper Tutor'}</div><div className={'message-text'+(m.role==='assistant'?' markdown-answer':'')}>{m.role==='assistant'?<MarkdownAnswer content={m.content||'Generating…'}/>:m.content||'Generating…'}</div>
            {m.citations&&m.citations.length>0&&<div className="citations">{m.citations.map(c=><button key={c.evidence_id} onClick={()=>navigate(c)} title={c.quote}>
              [{c.number}] {c.source_url?'External · ':''}{c.section||'Evidence'} {c.page_numbers?.length?'· p.'+c.page_numbers[0]:''}</button>)}</div>}
          </div>)}
          {steps.length>0&&<div className="steps"><div><Activity size={13}/> Agent activity</div>{selectedMemoryCount>0&&<span>Learning memory · {selectedMemoryCount} relevant saved note{selectedMemoryCount===1?'':'s'} selected (see Run details for applied status)</span>}{steps.slice(-8).map((s,i)=><span key={i}>{s}</span>)}</div>}
          <div ref={bottomRef}/></div>
        <div className="compose">{error&&<div className="error">{error}{lastRunId&&<a style={{marginLeft:10,textDecoration:'underline'}} href={`${API}/api/agent/runs/${lastRunId}/debug`} target="_blank" rel="noopener noreferrer">Run details</a>}</div>}
          <div className="question-row"><textarea placeholder={scope==='paper'?'Ask a question about the selected papers…':'Ask a general learning question (no paper required)…'} value={question} onChange={e=>setQuestion(e.target.value)}
            onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}}} rows={3}/>
            {busy?<button className="action danger" onClick={cancel} title="Cancel run"><Square size={17}/></button>:
              <button className="action" onClick={send} disabled={!question.trim()||(scope==='paper'&&!selected.length&&!isMemoryListQuestion(question))} title="Send question"><Send size={17}/></button>}</div>
          <small>{scope==='paper'?'Sources are retrieved from selected papers. AI answers may still contain errors; check citations.':'General answers use external sources, not the selected PDFs. Check source links and factual claims.'}</small>
        </div>
      </section></div> : tab==='learning' ? <section className="learning-panel"><div className="learning-inner"><h1>Learning profile & memory</h1><p>Only explicitly confirmed memories are stored. A concept is not marked as mastered just because you asked about it.</p>
        <div className="settings-card"><h2>Teaching preferences</h2><div className="field-row"><label>Language<select value={profile.language} onChange={e=>changeProfile({...profile,language:e.target.value as 'zh'|'en'})}><option value="zh">Chinese</option><option value="en">English</option></select></label>
          <label>Explanation level<select value={profile.explanation_level} onChange={e=>changeProfile({...profile,explanation_level:e.target.value as Profile['explanation_level']})}><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></label></div>
          <label className="field">Learning domains (comma separated)<input value={profile.domains.join(', ')} onChange={e=>changeProfile({...profile,domains:e.target.value.split(',').map(x=>x.trim()).filter(Boolean)})}/></label>
          <label className="field">Learning goals (comma separated)<input value={profile.goals.join(', ')} onChange={e=>changeProfile({...profile,goals:e.target.value.split(',').map(x=>x.trim()).filter(Boolean)})}/></label>
          <label className="checkbox"><input type="checkbox" checked={profile.memory_enabled} onChange={e=>changeProfile({...profile,memory_enabled:e.target.checked})}/> Use confirmed learning memory in Adaptive RAG</label><p className="memory-hint">Relevant saved notes personalize Adaptive explanations; they are not used as paper evidence. Basic and Hybrid remain unpersonalized for fair comparisons.</p>
          <button className="primary" onClick={saveProfile}>Save preferences</button><p className="memory-hint" role="status">{profileDirty?'Unsaved changes — save preferences before the next question.':profileSaved?'Preferences saved. New questions will use these settings.':'Changes take effect after you save your preferences.'}</p></div>
        <div className="settings-card"><h2>Confirmed memories</h2><div className="field-row"><label>Domain<input placeholder="e.g. RAG" value={newMemory.domain} onChange={e=>setNewMemory({...newMemory,domain:e.target.value})}/></label><label>Topic<input value={newMemory.topic} onChange={e=>setNewMemory({...newMemory,topic:e.target.value})}/></label></div>
          <label className="field">Learning note<textarea rows={2} value={newMemory.content} onChange={e=>setNewMemory({...newMemory,content:e.target.value})} placeholder="E.g. I understand the difference between precision and recall."/></label>
          <label className="field">Mastery status<select value={newMemory.mastery} onChange={e=>setNewMemory({...newMemory,mastery:e.target.value})}><option value="mentioned">Mentioned</option><option value="studied">Studied</option><option value="confirmed">Confirmed mastered</option></select></label>
          <button className="primary" disabled={!newMemory.domain.trim()||!newMemory.topic.trim()||!newMemory.content.trim()} onClick={addMemory}>Confirm and save memory</button>
          <div className="memory-list">{memories.map(m=><div key={m.id} className={'memory-item'+(editingMemory?.id===m.id?' editing':'')}>
            {editingMemory?.id===m.id ? <div className="memory-editor">
              <div className="field-row"><label>Domain<input value={editingMemory.domain} onChange={e=>setEditingMemory(d=>d?{...d,domain:e.target.value}:d)}/></label>
                <label>Topic<input value={editingMemory.topic} onChange={e=>setEditingMemory(d=>d?{...d,topic:e.target.value}:d)}/></label></div>
              <label>Learning note<textarea rows={2} value={editingMemory.content} onChange={e=>setEditingMemory(d=>d?{...d,content:e.target.value}:d)}/></label>
              <label>Mastery<select value={editingMemory.mastery} onChange={e=>setEditingMemory(d=>d?{...d,mastery:e.target.value}:d)}>
                <option value="mentioned">Mentioned</option><option value="studied">Studied</option><option value="confirmed">Confirmed mastered</option></select></label>
              <div className="memory-actions"><button className="primary" disabled={!editingMemory.topic.trim()||!editingMemory.content.trim()} onClick={saveMemoryEdit}>Save changes</button>
                <button onClick={()=>setEditingMemory(null)}>Cancel</button></div>
            </div> : <><div><strong>{m.topic}</strong><small>{m.domain} · {m.mastery}</small><p>{m.content}</p></div>
              <div className="memory-actions"><button onClick={()=>setEditingMemory({...m})} title="Edit memory"><Pencil size={16}/></button>
                <button onClick={()=>deleteMemory(m.id)} title="Delete memory"><Trash2 size={16}/></button></div></>}
          </div>)}</div>
        </div>{error&&<div className="error">{error}</div>}</div></section> : <section className="conversations-panel"><div className="conversations-inner">
      <div className="conversations-toolbar"><div><h1>Conversations</h1><p>Browse, rename, reopen or delete previous paper discussions. {conversations.length} total.</p></div>
        <button className="primary" onClick={newConversation} disabled={busy}><Plus size={16}/> New conversation</button></div>
      <div className="conversation-search"><Search size={18}/><input aria-label="Search conversations" placeholder="Search conversation titles..." value={conversationSearch} onChange={e=>setConversationSearch(e.target.value)}/></div>
      {error&&<div className="error">{error}</div>}
      <div className="conversation-cards">
        {filteredConversations.map(c=><div key={c.id} className={'conversation-card '+(c.id===conversation?'current':'')}>
          <div className="conversation-details">
            {renaming===c.id?<div className="rename-row"><input aria-label="Conversation title" value={titleDraft} onChange={e=>setTitleDraft(e.target.value)} maxLength={120} onKeyDown={e=>{if(e.key==='Enter')renameConversation(c.id);if(e.key==='Escape')setRenaming(null)}} autoFocus/>
                <button title="Save title" onClick={()=>renameConversation(c.id)}><Check size={17}/></button><button title="Cancel rename" onClick={()=>setRenaming(null)}><X size={17}/></button></div>:
                <button className="conversation-open" disabled={busy} onClick={()=>switchConversation(c.id)}>{c.title}</button>}
            <small>{c.message_count||0} messages · {c.updated_at||c.created_at?new Date(c.updated_at||c.created_at!).toLocaleString():''} {conversation===c.id?' · Current':''}</small>
          </div>
          <div className="conversation-actions"><button title="Rename conversation" onClick={()=>beginRename(c)} disabled={busy}><Pencil size={16}/></button>
            <button title="Delete conversation" onClick={()=>deleteConversation(c)} disabled={busy}><Trash2 size={16}/></button></div>
        </div>)}
        {filteredConversations.length===0&&<p className="empty-conversations">No matching conversations.</p>}
      </div>
    </div></section>}
  </main>;
}
