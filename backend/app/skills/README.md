# Paper Tutor Skills

Each reusable Skill is a **folder containing `SKILL.md`**. All seven Skills are registered in `registry.py`, with a validated manifest:

```yaml
---
name: explainForBeginner
type: style
description: Teach an academic concept to a beginner
output_contract: teaching
version: 1
---
```

The remainder describes **when to use it, the procedure, the response contract and evidence boundaries**. The registry validates the metadata and provides the instructions to AgentRunner. There is only one canonical Skill body; `app/agent/skills.py` is now a compatibility selector, not a second store of prompts.

| Skill | Role | Verification |
| --- | --- | --- |
| explainForBeginner | structured definition + hypothetical example + takeaway | `teaching.py` validates shape, source identity, citation, unsupported code-domain swap, unwanted claims; programmatic renderer |
| guidePaperReading | problem, method, evidence, limitations | source-grounded general QA |
| comparePapers | cross-paper problem/method/experiments | source-scoped retrieval, no evidence transfer |
| suggestLearningPath | prerequisite sequence with progress context | learning memory cannot become paper facts |
| explainFormula | formal equation and variable explanation | `outputs.py`, `quality.py` source/formula checks |
| verifyClaims | per-claim evidence-backed verdicts | structured `claims` contract, table checks |
| constructCounterexample | concrete mathematical refutation | `quality.py` numerical/set checks |

## Design notes

A Skill is a **task recipe/answer contract**, not a new LLM, autonomous agent, or knowledge source. Retrieval tools return paper passages. The Agent decides when to use them. The Skill constrains how to answer; a validator checks output obligations. Qwen remains a small local model. Stronger correctness is never guaranteed by metadata or a template alone.

Day 1 deliberately uses the new structured teaching response only for Adaptive/general beginner questions. The Basic/Hybrid benchmark baselines and established analytic answers remain unchanged. Additional skills are specified in `SKILL.md` but are not separate sub-agents or guaranteed independently verified outputs.
