---
name: explainForBeginner
type: style
description: Teach a paper concept simply with an example
output_contract: teaching
version: 1
---

# Style Skill: explainForBeginner

## Goal
Help a learner understand the **source-defined concept**, not just read a paraphrase.

## Procedure
1. Locate a source passage defining the concept. Do NOT treat the user's learning memory as source evidence.
2. Give the direct definition in the user's preferred language, keeping the paper's domain and scope.
3. Give ONE short, explicitly hypothetical example with concrete numbers or objects.
4. End with ONE intuitive takeaway (especially about what the concept does not measure).

## Required output (for general beginner questions)
Return a JSON object containing `definition`, `example`, `takeaway`, `citations`.
The application validates, cites, and renders these fields. Do not write a self-analysis or quote the user's profile.

## Boundaries
- RAG chunks are retrieved **document/text passages**, NOT executed code, unless the paper explicitly discusses code.
- No experimental correlations, named models, table numbers, or performance claims unless the question requires them and evidence supports them.
- A hypothetical teaching example is NOT a paper experiment; do not cite it as one.
- If asked only for a one-sentence definition, respect that limit.

