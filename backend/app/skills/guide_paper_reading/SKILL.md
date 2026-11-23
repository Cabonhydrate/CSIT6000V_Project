---
name: guidePaperReading
type: task
description: Guide an early-stage researcher through a paper's actual argument and evidence
output_contract: sections
version: 1
---

# Skill: guidePaperReading

## When to use
When the user requests a guided read-through, paper walkthrough, or structural summary.

## Procedure
1. Respect the outline and ordering requested by the user.
2. Explain the research motivation and shortcomings of previous work using paper evidence.
3. Identify the paper's actual contribution, not a background metric merely mentioned in it.
4. Give the experiment design, key findings, and conclusion without inventing numbers.
5. Cite each part directly. If a part cannot be substantiated, label that limitation.

## Contract
For the standard five-part walkthrough, generate exactly: 研究背景 / 现有方法的问题 / 核心创新 / 实验设计 / 主要结论. Each part must be distinct and have a valid source citation. Avoid substituting a generic definition for a paper outline.
