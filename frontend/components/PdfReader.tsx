'use client';
import { useEffect, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import { api, API, Paper } from '../lib/api';

pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString();

type PaperChunk = {id:string;section:string;content:string;paragraph_refs:string[];pages:number[]};
export default function PdfReader({paper,page,highlight}:{paper:Paper|null;page:number;highlight?:string}) {
  const [count,setCount]=useState(0);
  const [pdfError,setPdfError]=useState('');
  const [visible,setVisible]=useState(1);
  const [width,setWidth]=useState(650);
  const [chunks,setChunks]=useState<PaperChunk[]>([]);
  useEffect(()=>{
    setCount(0);setPdfError('');
    setChunks([]);
    if(paper?.source_type==='qasper')api<PaperChunk[]>(`/api/papers/${paper.id}/chunks`).then(setChunks).catch(()=>setChunks([]));
  },[paper?.id,paper?.source_type]);
  useEffect(() => {
    setVisible(page || 1);
  }, [page, paper?.id]);
  useEffect(()=>{
    const update=()=>setWidth(Math.max(300,Math.min(730,window.innerWidth*0.44-55)));
    update();window.addEventListener('resize',update);return ()=>window.removeEventListener('resize',update);
  },[]);
  if(!paper)return <div className="empty-reader">Select a paper to read. Citations will link to its source page.</div>;
  if(paper.source_type!=='pdf')return <div className="qasper-reader"><div className="qasper-header"><strong>{paper.title}</strong><small>QASPER structured paper · original paragraph references retained</small></div>{chunks.map(c=><article key={c.id} className={'qasper-chunk '+(highlight&&c.content.includes(highlight)?'focused':'')}><h4>{c.section}</h4><p>{c.content}</p><small>Evidence refs: {c.paragraph_refs.join(', ')}</small></article>)}</div>;
  return <div className="pdf-holder">
    <div className="pdf-toolbar">
      <span>{paper.title}</span>
      <div className="pdf-nav">
        <button onClick={()=>setVisible(Math.max(1,visible-1))} disabled={visible<=1}>‹</button>
        <span>{visible} / {count||'…'}</span>
        <button onClick={()=>setVisible(Math.min(count,visible+1))} disabled={!count||visible>=count}>›</button>
      </div>
    </div>
    <Document key={paper.id} file={`${API}/api/papers/${paper.id}/document`}
      onLoadSuccess={({numPages})=>{setCount(numPages);setPdfError('');}}
      onLoadError={err=>setPdfError(err.message)}
      loading={<p className="reader-note">Loading PDF…</p>}
      error={<p className="reader-note">PDF not available ({pdfError || 'load failed'}). If the project folder was moved, restore backend/data/pdfs or re-upload this paper.</p>}>
      <Page pageNumber={Math.max(1,Math.min(visible,count||visible))} width={width}/>
    </Document>
    {highlight&&<div className="pdf-highlight-note">Citation excerpt: {highlight}</div>}
  </div>;
}
