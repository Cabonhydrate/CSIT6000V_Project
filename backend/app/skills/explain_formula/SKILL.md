---
name: explainFormula
type: task
description: Explain source-backed mathematical notation
output_contract: formula
version: 1
---

# Skill: explainFormula

Answer the mathematical question in the user's language.
- Read the formula from the supplied source rather than inventing notation.
- Give an actual equation, define EVERY symbol the user asks for, explain the
  mathematical operation (e.g. set union implies distinct items), and add a small
  hypothetical example only if it helps.
- Preferred JSON object:
  {"formula":"latex without delimiters","variables":[{"symbol":"T","meaning":"..."}],
   "explanation":"...","citations":[1]}
- The formula value must be a COMPLETE mathematical expression (typically an
  equality with left and right sides), never just a symbol such as "T".
- In JSON strings escape each LaTeX backslash correctly: a formula containing
  \frac must be encoded with \\frac in raw JSON. Likewise for \bigcup, \tau,
  and other LaTeX commands. Unescaped backslashes can become control characters.
- Do not substitute a symbol list for the actual requested equation.
- A legacy {"answer":"...","citations":[1]} is accepted for existing clients.
- If a variable or formula is not present in the evidence, do not guess.
