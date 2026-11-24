# v0.3.13: Structure-Aware PDF Chunking, Complete Citation Passages, and Learning Context

## Summary

This release addresses three distinct issues observed during local browser acceptance:

1. PDF parsing sometimes created large page/block-based passages. Splitting based on the last whitespace could cut inside an English sentence.
2. The PDF citation panel displayed a **bounded excerpt**, which could end in the middle of a paragraph and was mistaken for the complete indexed chunk.
3. Confirmed learning memories were retrieved, but a model could still ignore a previously stated learning gap. The teaching instructions now explicitly promote relevant gaps into answer construction, including the multi-question path.

**Importantly, a relevant memory being selected is not proof that an answer was personalized.** The existing run metrics distinguish recall from prompt injection, but only a controlled Memory-on/off comparison and factual review can establish a useful learning effect.

## Structural chunking design

The PDF ingestion path uses the following priority:

1. Normalize CRLF, CR, LF, repeated horizontal whitespace, non-breaking spaces, and invalid control characters **in extracted text only**. The original PDF is not changed.
2. Prefer Docling-provided heading/section labels. In the PyMuPDF fallback, use text blocks plus conservative English heading recognition, including `2 Background`, `2.1 Methodology`, `Abstract`, and `References`. Detect a heading at the beginning of a larger block where possible.
3. Recursively split oversized PDF blocks using, in order: blank paragraph separators, line breaks, sentence punctuation, semicolons, commas/colons, word whitespace, and finally the fixed character boundary. The output is reconstructed exactly from the normalized input. The upper bound is **1,700 characters**, including separators when merging.
4. Merge adjacent short passages **within the same section**, subject to the same hard length bound. Do not overlap fragments from the same oversized original paragraph: that would repeat text and distort retrieval.
5. Preserve original PDF page numbers and paragraph identifiers on every derived fragment. Never invent a page number for QASPER structured text.

The exact hierarchy balances structure preservation against a hard bound. Some single line breaks in PDFs are visual line wraps rather than genuine semantic boundaries. Therefore, **no heuristic segmentation can guarantee that every sentence is complete**, particularly for long equations, line-wrapped abstracts, malformed extraction, or scanned PDFs. Actual output must be inspected for the papers used in the demo.

### QASPER compatibility

The QASPER import path **does not** activate PDF-only splitting or normalization: original paragraph IDs and gold evidence references remain unchanged. This preserves comparability with previously recorded QASPER experiments. The re-ingested PDF chunks should not be compared numerically with prior chunk coverage measurements: changing segmentation changes the denominator.

## Citation display

A citation button continues to navigate to the original PDF page and displays a **source preview**. When the citation has an indexed chunk ID, the frontend also loads the complete corresponding source chunk through the existing, ownership-checked `GET /api/passage/{chunk_id}?paper_id={paper_id}` endpoint. Use **Show full indexed passage** under the preview to inspect all stored text.

The full indexed passage is **not** the same thing as the full PDF page. It does not reconstruct tables or mathematical glyphs absent from the parser. The preview and the full indexed passage are rendered as separate content so users do not mistake the short quote for the complete document.

Citation navigation and source relevance are distinct checks. A direct link to Page 9 is not sufficient evidence if a question asks for a definition on Page 6. Verify whether the linked passage actually supports the answer.

## Important: re-import PDFs safely

**Existing PDF indexes are not automatically migrated.** Index row IDs are sequential (`paper-id:c0`, `paper-id:c1`, ...). Re-indexing an existing paper in place can silently make historic conversations refer to a different passage under the same ID. To keep provenance reliable:

1. Back up the database and existing PDF if you need the old conversation or citations.
2. In the browser, **upload the PDF again**, as a new paper with a new ID. Wait for its status to be `ready`.
3. Select the new paper and open a citation, then inspect **Show full indexed passage**. Verify its page and content.
4. Keep the old paper until historic conversations are no longer required. Do not delete historic evidence just to refresh the current index.
5. Clear or rename old experiment result files so old and new chunk IDs are never treated as interchangeable. Regenerate embeddings implicitly against the new content fingerprints.

No TACC service or remote deployment is involved in this migration.

## Memory A/B check

The previously displayed A/B question accidentally included a second line, **`Copy question`**. The Agent treated that literal text as another subquestion. This was input contamination, not evidence that the multi-question processor would fail with the intended input. **Do not copy UI button labels into the question.**

With the same new PDF ID, Adaptive mode, Beginner skill enabled, and English preference, start **two new conversations**. Ask exactly the same question in both:

> Explain what Chunk Coverage measures, why repeatedly retrieving the same chunk does not increase coverage, and why high coverage does not guarantee correct answers.

For Memory ON, save a confirmed note with `Domain: RAG`, `Topic: Chunk Coverage`, `Mastery: Studied`, and:

> I understand the basic retrieval pipeline, but I am not familiar with set union and how the coverage formula is calculated.

For Memory OFF, disable confirmed-memory retrieval in the learning preferences before starting the second run. Keep all other settings and wording identical. **Do not assume the Memory-on answer is better merely because it selected one note.**

Inspect Run details for `learning_memory_candidates`, `learning_memory_applied`, `learning_context_applied`, and `answer_complete`. Then independently score: correct test-suite-level CC definition, no double-counting under union, no claim that CC guarantees correctness, PDF reference relevance, and whether the ON answer actually teaches set union without repeating basic RAG material. These are separate measurements.

## Local checks

```powershell
# From the backend directory
python -m pytest -q
python -m compileall -q app

# From the frontend directory; run on the complete Windows checkout
npm ci
npx tsc --noEmit
npm run build
```

The development container passed **334 Python tests** and Python source compilation; the edited TSX files passed a syntax/transpile check using TypeScript. The full Next.js production build and a real Windows/Ollama Memory A/B re-test **have not been run in this environment**.

## Known limitations

- Section heuristics may miss headings embedded in tables, figures, or nonstandard layouts.
- Figure/table extraction remains incomplete; the full indexed passage is plain extracted text, not a complete PDF view.
- New segmentation may change retrieval rankings and chunk-based evaluation metrics. Re-run the benchmarks if comparing results after migration.
- The local language model may ignore some memory guidance or generate an incomplete answer; automatic tests that simulate a model cannot establish real response-quality improvement.
