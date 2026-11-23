'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** Render LLM answers as Markdown, including inline and block LaTeX.
 * remark-math natively handles $...$ and $$...$$. Also accept the common
 * \(...\) / \[...\] delimiters used by many language models.
 */
function normalizeMathDelimiters(content: string): string {
  return content
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) => `\n$$\n${math.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, math: string) => `$${math.trim()}$`);
}

export default function MarkdownAnswer({content}: {content: string}) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
      {normalizeMathDelimiters(content)}
    </ReactMarkdown>
  );
}
