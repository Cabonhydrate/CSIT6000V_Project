import type {Metadata} from 'next';
import './globals.css';
import 'katex/dist/katex.min.css';
export const metadata:Metadata={title:'Paper Tutor | Adaptive Agentic RAG',description:'Learn research papers with grounded RAG explanations'};
export default function RootLayout({children}:{children:React.ReactNode}) {return <html lang="en"><body>{children}</body></html>}
