export const API = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export type Paper = {id:string; title:string;status:string;source_type:string;parser?:string;error?:string};
export type Citation = {number:number; evidence_id:string;paper_id?:string;chunk_id?:string;source_type?:string;section?:string;page_numbers?:number[];paragraph_refs?:string[];quote?:string;source_url?:string};
export type Message = {id?:string;role:'user'|'assistant';content:string;citations?:Citation[]};
export type Conversation = {id:string;title:string;created_at?:string;updated_at?:string;message_count?:number};
export type Memory = {id:string;domain:string;topic:string;memory_type:string;content:string;mastery:string};
export type Profile = {language:'zh'|'en';explanation_level:'beginner'|'intermediate'|'advanced';domains:string[];goals:string[];memory_enabled:boolean};

export async function api<T>(path:string, init?:RequestInit):Promise<T> {
  const response=await fetch(API+path,init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const err=await response.json(); detail=typeof err.detail==='string' ? err.detail : detail; } catch {} 
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}
export const jsonInit=(value:unknown,method='POST'):RequestInit=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
