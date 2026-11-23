---
name: constructCounterexample
type: task
description: Produce a concrete hypothetical mathematical witness
output_contract: example
version: 1
---

# Skill: constructCounterexample

When the user requests a mathematical counterexample, a verdict alone is not an answer.
- Identify the two mathematical expressions or claims being compared.
- Instantiate SMALL concrete mathematical objects in the same domain (for sets, explicitly list elements).
- Make the property responsible for the contradiction visible (e.g. an element shared by two sets).
- Show both calculated results from those objects, including the actual arithmetic and their difference.
- Explain exactly which assumption of the incorrect claim fails.
- Mark your construction as a hypothetical mathematical example, not a real paper experiment.
- Keep source citations for source-defined formulas; do not cite the paper as if it supplied your invented example.
- If the source does not supply a needed definition, say so rather than assuming it.
- When the user ALSO explicitly requests a formal formula, retain the complete `formula` and `variables`, and put the numerical counterexample in `explanation`. Otherwise a cited concrete construction is sufficient.
