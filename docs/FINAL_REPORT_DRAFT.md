# Adaptive Agentic RAG: A Local Academic Paper Research and Learning Assistant

*CSIT6000V — Advanced AI Computing and Systems*  
*Final Project Report — English Draft*  
*Authors: [Insert team member names] | Student IDs: [Insert only if required]*  
*Submission date: [Insert course deadline] | Git revision: [Insert final commit hash]*

> **Editorial note (remove before submission):** This draft describes the implemented local prototype and the measurements already recorded. Update the bracketed fields, verify the course's required page/word limits, capture final-build evidence, and ensure all screenshots correspond to the submitted Git revision. Do not present the described future TACC work as completed.

## Abstract

Reading research papers can be difficult for students who are new to a technical field. Standard retrieval-augmented generation (RAG) can retrieve relevant passages from papers, but it may answer incompletely, cite weak evidence, or fail to explain prerequisite concepts. We developed **Adaptive Agentic RAG**, a locally hosted academic paper research and learning assistant that integrates document ingestion, multiple retrieval strategies, bounded Agent orchestration, evidence validation, beginner-oriented explanations, explicit learning memory, and external background knowledge search. A four-layer design separates the Next.js interface, FastAPI/LangGraph application, local Ollama model service, and compute hardware. The system runs with Qwen3:4b and supports Basic RAG, Hybrid RAG, and Adaptive Agentic RAG for paper question answering. A separate General Learning scope can answer foundational questions without a selected paper and provide external URL citations. We evaluated three retrieval modes on a QASPER development subset and a small cross-paper holdout, and examined a paper-specific development regression suite. All 18 holdout runs completed, but Adaptive did not consistently outperform Hybrid in answer quality and required more model calls. The results demonstrate a usable local prototype while identifying remaining challenges in factual grounding, source attribution, model reliability, and cold-start latency.

**Keywords:** retrieval-augmented generation, Agentic RAG, academic question answering, local language models, evidence grounding, learning assistant

## 1. Introduction

Academic papers often assume familiarity with terminology, mathematical notation, and prior research. Students may need to locate an exact claim in a paper, compare two methods, or learn a fundamental concept before returning to the paper. A general chatbot may provide fluent explanations, but it does not necessarily establish whether those explanations follow from the selected research article.

This project addresses that gap with a research assistant that treats **paper evidence and external background knowledge as different source categories**. A student can upload or select a paper, ask a question and inspect linked passages; alternatively, the student can switch to General Learning and request an introductory explanation supported by a web source. This design is intended to support learning rather than replace critical reading.

The project has four main objectives:

1. Implement an end-to-end local research assistant without depending on a paid hosted LLM API.
2. Compare one-round dense retrieval, one-round hybrid retrieval, and bounded adaptive retrieval on the same academic questions.
3. Expose evidence and execution traces so users can inspect sources and understand partial answers.
4. Support beginner-friendly explanations and explicit memory of user-confirmed learning context.

The implemented scope is a **single-user local prototype**. Remote TACC model serving, public deployment, multi-user security and high-throughput inference are outside the implementation evaluated here.

## 2. System Design

### 2.1 Four-layer architecture

The frontend is implemented in **Next.js** and provides a PDF reader, chat interface, retrieval mode selectors, citation controls and learning settings. The **FastAPI** backend manages ingestion, conversations, local storage, run events and the Agent. **LangGraph** coordinates bounded retrieval and evidence assessment. The AI service is a locally hosted **Ollama** endpoint serving **Qwen3:4b**. Local CPU, GPU, memory and storage form the hardware layer. This separation allows the application to communicate with a model-serving interface without embedding hardware scheduling into the Agent.

Two cross-cutting capabilities connect the layers. **Observability** records run IDs, tool calls, model interactions, citations, errors and timing information. **Evaluation** uses QASPER questions and a separate paper-specific review set to assess response quality, evidence coverage and runtime cost.

### 2.2 Paper ingestion and retrieval

Research PDFs are parsed and indexed into source-linked chunks with page references. The system can also import the original QASPER development JSON, preserving paragraph identifiers for offline evaluation. QASPER text is not converted into a fake PDF, so QASPER paragraph references are not presented as fabricated PDF page numbers.

Basic RAG performs one dense retrieval round. Hybrid RAG combines lexical and dense retrieval rankings. Adaptive Agentic RAG can select additional retrieval rounds after evaluating evidence sufficiency. The latter requires more orchestration work, making the accuracy-versus-cost trade-off an empirical question rather than a design assumption.

### 2.3 Adaptive reasoning and answer validation

An Adaptive run identifies the question type, retrieves supporting passages, checks evidence sufficiency, and produces an answer with citations. The workflow may perform bounded recovery when the model emits incomplete output or a claim lacks suitable support. Later development introduced source-grounded checks for terminology, formula units, hypothetical arithmetic and paper-to-external comparisons.

A key distinction is between a **reported paper fact**, a **derived calculation**, and an **illustrative hypothetical example**. For example, a hypothetical calculation in which 6 of 10 document chunks are retrieved gives a coverage fraction of 60%; that percentage should be arithmetically checked, but it should not be attributed to an experiment in the original paper. In contrast, actual experimental numbers must be supported by the cited source.

Conservative `partial` answers are used when source evidence or a validated answer is unavailable. This design reduces unsupported certainty, although overly strict checks can also reject useful answers. The system records validation attempts so that both false acceptance and false rejection can be investigated.

### 2.4 Teaching skills and confirmed learning memory

The application uses file-backed Skills for beginner explanations, formula explanations, paper reading, claim verification, comparisons, counterexamples and study planning. These Skills are **structured prompts and output contracts**, not separate specialist models. The user can configure an explanation level and store confirmed learning notes. The user may disable the retrieval of saved notes for a new conversation; stored preferences are not treated as scientific facts about a paper.

### 2.5 External knowledge and citation provenance

The application separates **Paper QA** from **General Learning**. The former retrieves indexed paper passages. The latter does not require a selected paper and searches external concepts. For paper questions that explicitly compare a paper concept with a background concept, Adaptive may use both source types. The UI labels **PDF** references separately from **External** references and opens the original source. External academic metadata and introductory articles are supplemental evidence, not replacements for the paper's actual claims.

## 3. Implementation and Local Deployment

The backend is written in Python; the frontend uses TypeScript/React. SQLite stores application records, and retrieved passages may be ranked using BM25, Sentence Transformers embeddings and FAISS. Optional reranking can be enabled separately. The model-client boundary supports local Ollama and a configurable OpenAI-compatible interface, although the latter was not used as a deployed TACC/vLLM service in this project phase.

The developer verified multiple real Ollama runs on a Windows machine. Live acceptance cases covered paper definition queries, PDF excerpts, external concept search, comparison of paper terminology with general software-testing terminology, memory toggling, and cancellation/error behavior. The latest final build should still be executed and recorded against the exact submission commit before delivery.

## 4. Experimental Design

### 4.1 Benchmark strategies and metrics

We compared the three paper-retrieval modes using QASPER v0.3 development questions. Answer quality was summarized using a **simplified token-overlap F1 proxy** and paragraph Evidence F1 where reference mapping was possible. These numbers are not official QASPER leaderboard scores. We also report successful run counts, mean latency, mean LLM calls and mean retrieval rounds. A run marked `succeeded` is operationally complete, but its answer may still be factually incorrect.

### 4.2 Fixed development subset

The initial QASPER subset contained three papers and five questions per paper. Each trial therefore comprised 45 mode-question runs. In Trial 2, Basic completed 13/15 runs, Hybrid 14/15, and Adaptive 11/15. The Answer F1 proxies were approximately 0.131, 0.172 and 0.082, respectively. Adaptive used approximately 3.67 model calls per question, compared with 2.00 for Basic and Hybrid, and approximately 2.80 retrieval rounds, compared with 1.00 for the baselines.

### 4.3 Cross-paper holdout

The holdout used two further development papers with three questions each, for 18 total runs. All 18 completed. Hybrid had the highest observed Answer F1 proxy (0.104), followed closely by Adaptive (0.102); Basic scored 0.026. Mean LLM calls were 2.00 for Hybrid and 3.50 for Adaptive. The holdout was small and not sufficient to establish a statistically meaningful mode ordering. In addition, an older Adaptive benchmarking configuration injected learner-profile context even with teaching and memory disabled; later code corrected this issue, but the recorded holdout was not repeated as a clean post-fix experiment.

### 4.4 Development regression testing

A separate eight-question set based on a Chunk Coverage research article tested formulas, table values, claims, supplemental experiments and unanswerable questions. A human review score increased from 6.38/10 in the first iteration to 8.38/10 in the fourth. This increase is evidence of **improvement on a repeatedly inspected development set** and cannot be used as an unbiased estimate of performance on unseen papers.

A final implementation-oriented validation also confirmed that Basic, Hybrid and Adaptive could each answer a direct Chunk Coverage definition question with a linked paper excerpt. A mixed-source comparison with code coverage demonstrated separate PDF and external citations, including source-based recovery after invalid model-generated similarities.

## 5. Results and Discussion

The system successfully integrates local inference, document retrieval, evidence presentation and interactive learning features. The baseline modes remain simpler to execute, while Adaptive exposes evidence checks and recovery opportunities. However, **Adaptive was not consistently better** in the observed QASPER results; its extra model calls and retrieval rounds impose real cost.

Several concrete failure modes were observed. In the holdout, a decoder described as an LSTM in the stored gold answer was incorrectly described as a Transformer by one Adaptive response. Other answers omitted specific requested facts such as a translation tool or a system score. A model could retrieve relevant evidence but still fail to answer the actual question, and a plausible answer could cite an overly broad or irrelevant excerpt. The project therefore distinguishes retrieval success, source attribution, answer completeness and factual correctness as separate dimensions.

The local deployment revealed a substantial **cold-start performance cost** for embedding initialization. One live mixed-source run spent around 32.25 seconds in retrieval setup versus about 7.18 seconds in model execution. This should not be interpreted as steady-state retrieval latency: subsequent retrieval operations were often much faster after weights were cached.

The interface-based citation mechanism also showed the benefits of maintaining source provenance. In the same question, a reference to a paper's Chunk Coverage definition could open a PDF excerpt, while the definition of software code coverage linked to an external encyclopedia page. This separation is necessary for pedagogical clarity and prevents a general background statement from being misrepresented as an author-specific research result.

## 6. Limitations and Threats to Validity

**Answer reliability:** The compact local model can hallucinate, omit essential details, leak reasoning text or repeat an invalid comparison. Validators reduce specific failure classes but do not provide general theorem proving or complete semantic entailment.

**Evaluation:** The QASPER selections are small and partly filtered for mappable evidence. Literal F1 is language-sensitive and cannot replace manual factual review. The CC8 development set was reused during optimization. Earlier benchmarking mode settings were not perfectly isolated. For these reasons, the project cannot claim a general performance advantage for Adaptive.

**Document and source processing:** Complex tables, scanned PDFs, equations and figures are not universally handled. Web search returns curated external source candidates, not a guaranteed authoritative knowledge base. Real URLs can be topically irrelevant; source-ranking safeguards reduce but do not eliminate this risk.

**Runtime and deployment:** The implementation is single-user and local. Background tasks are not a persistent distributed queue, short-term Agent execution is not fully durable across process restarts, and the system is not hardened for Internet-facing production use. Network-dependent sources and model downloads can fail. TACC/vLLM serving has not been implemented or measured.

## 7. Conclusion and Future Work

The project demonstrates a functioning local academic paper assistant combining multiple RAG retrieval strategies, bounded Agent orchestration, beginner-oriented Skills, confirmed learning memory and explicit PDF versus external evidence attribution. The evaluation shows that the architecture is viable, but it also shows why additional Agent steps cannot be assumed to improve answer quality. Future work should prioritize stronger source entailment checks, robust figure/table understanding, controlled cross-language evaluation, locally preloaded embedding models, broader untouched benchmarks, and, as a separate deployment milestone, TACC-hosted vLLM serving with reproducible GPU performance measurements.

## References

1. Dasigi, P., et al. *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers* (QASPER). Official dataset: https://huggingface.co/datasets/allenai/qasper
2. *Testing Retrieval-Augmented Generation Systems with Chunk Coverage*. Research paper used in the paper-specific development regression set; include the exact bibliographic entry from the uploaded PDF in the final report.
3. LangGraph documentation: https://langchain-ai.github.io/langgraph/
4. Ollama documentation: https://ollama.com/
5. Docling documentation: https://docling-project.github.io/docling/

**Before submission:** confirm the final report requirements; adjust length/format; replace bibliographic placeholder [2] from the paper itself; insert verified screenshots/figures and actual final-build evidence; remove this editorial note.
