---
name: suggestLearningPath
type: task
description: Recommend prerequisites and review a learner's explicitly saved progress
output_contract: learning_path
version: 1
---

# Skill: suggestLearningPath

## When to use
The user requests a sequenced learning path, or a review of previous learning progress and next reading steps.

## Procedure
1. Use confirmed user-owned learning memories only to determine the starting point and known gaps. These are NOT paper citations.
2. For a learning-path request, provide the required number of progressive stages (default 3).
3. Each stage needs what to learn, why it matters, and a concrete self-check.
4. For progress review, separate already-studied topics from verified mastery. Do not claim mastery because the user once studied a concept.
5. Recommend a paper section using the retrieved paper passages, with citations when justified.

## Contract
Either an ordered learning path with distinct stages and self-checks, OR a progress review consisting of studied / not-yet-mastered / next reading. Never answer a learning-plan request by merely defining one academic term.
