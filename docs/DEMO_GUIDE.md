# Live Demo Guide

**Purpose:** Demonstrate an end-to-end local academic paper assistant with paper-grounded answers, adaptive execution, external background learning, skills, memory and error handling.

**Preparation:** Set the application learning-profile language to **English** for submission screenshots and start new English-language conversations. The project may support Chinese as a product feature, but the formal demonstration material should be English. The backend and frontend run locally; Ollama `qwen3:4b` is available. Use a paper that is permitted to be shown. Do not show private IDs, personal notes, credentials or raw local file paths on screen.

## Pre-demo checks

- [ ] `http://127.0.0.1:8000/api/health` reports a healthy backend and available model.
- [ ] Run `python -m app.diagnose` from `backend/` to confirm actual model generation, not just an HTTP health response.
- [ ] Open `http://localhost:3000` and ensure the selected demo PDF is `ready`.
- [ ] Prime the embedding model with one paper search to avoid spending the demo on cold-start downloads.
- [ ] Confirm Internet access for external examples; have a local-only fallback ready.
- [ ] Confirm citation buttons lead to the corresponding excerpt and external source URLs.

## Suggested 4–6 minute demonstration

| Segment | Action | What to explain |
| --- | --- | --- |
| 1. Introduction | Show Paper QA and indexed papers | The app uses a locally hosted LLM instead of a commercial inference API |
| 2. Paper ingestion/reading | Open a PDF; optionally upload a new permitted paper | Parsing, passage indexing and page-grounded citations |
| 3. Retrieval strategies | Show Basic, Hybrid and Adaptive on a paper question | Dense versus fused retrieval; Adaptive can perform evidence checks and additional rounds |
| 4. Citation inspection | Click a PDF reference | The viewer reveals a real passage; citations are not merely generated footnotes |
| 5. Beginner mode | Request an easy explanation with an example | File-backed teaching Skills and fact-versus-hypothesis boundaries |
| 6. General Learning | Deselect papers; ask `What is machine learning?` | External knowledge is a separate scope, with external URL citations |
| 7. Memory and diagnostics | Show a confirmed note and run activity | Optional learner context; observable tools, states and recovery |
| 8. Limitations | Show the evaluation summary | Higher Agent cost and imperfect factual reliability are measured, not hidden |

## Demonstration prompts (English)

**Paper-grounded definition:**

> According to Section 3 of this paper, what is Chunk Coverage? Explain what is counted and what it does not measure. Cite the relevant passage.

**Formula and beginner explanation:**

> Explain the Chunk Coverage formula for a beginner. Define every symbol and give one explicitly hypothetical example of repeated retrieval.

**Mixed-source comparison (Adaptive + Academic Web enabled):**

> Chunk Coverage seems similar to code coverage in software testing. What do they have in common, and what is different? Cite the paper for Chunk Coverage and an external source for code coverage.

**General background learning (no paper required):**

> What is machine learning? Explain it to a beginner with a concrete example and provide a source.

**Memory example:**

> Based on my confirmed learning note, explain why set union prevents counting the same retrieved chunk twice.

## Narration points

- The **four layers** are frontend, application/Agent backend, model service, and hardware.
- The Agent controls *what to do*; the model service performs inference.
- A PDF citation and an external web citation are distinguished by source provenance.
- A teaching example is illustrative; it must not be portrayed as a reported experiment result.
- `succeeded` means the workflow completed, **not** that independent scientific verification is guaranteed.
- A `partial` response is sometimes the correct safety behavior when required evidence is unavailable.

## Failure handling during the demonstration

| Situation | Response |
| --- | --- |
| First paper search is slow | Explain that embedding weights may be warming up; show a previously validated run rather than rerunning many times |
| External website is unreachable | Demonstrate conservative `partial` behavior or switch to Paper QA with local sources |
| Local model fails to generate an answer | Show diagnostic run ID and logs; do not substitute a prewritten answer without disclosure |
| Source does not support the answer | State that limitation; use a different, prevalidated example |
| Long Adaptive run | Show its activity and compare cost with one-round Basic/Hybrid; avoid claiming it is always more accurate |

## Evidence to capture for submission

- One screenshot of the main frontend with a loaded paper.
- One screenshot showing a correct answer and its PDF excerpt.
- One screenshot showing an external link opened on the relevant page.
- One screenshot or sanitized log showing Agent activity and citation provenance.
- A terminal capture of backend tests and frontend production build from the **final** Git commit.
- A Git commit hash and the local model name used for acceptance.

If the instructor requires a specific video duration or submission format, **follow the instructor's requirement** rather than this suggested sequence.
