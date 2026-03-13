---
description: 
alwaysApply: true
---

# Engineering Workflow Rule

The assistant must follow this workflow for all non-trivial tasks.

## 1. Plan Mode Default
- Enter plan mode for any task with more than 3 steps or architectural decisions.
- If progress deviates from expectations, stop and re-plan immediately.
- Use plan mode for verification steps, not only for building.
- Write specifications before implementation when ambiguity exists.

## 2. Context Management
- Keep the main context window clean.
- Offload exploration, research, and parallel analysis to separate reasoning threads when possible.
- Focus on one clearly defined task at a time.

## 3. Continuous Self-Improvement
- When a correction is provided, update the internal rule set with the lesson learned.
- Prevent repeating the same mistake.
- Reuse these lessons across sessions.

## 4. Verification Before Completion
Never mark a task as done without verification.

Verification includes:
- running tests
- checking logs
- validating behavior against the specification

Ask before completion:

> "Would a staff engineer approve this change?"

## 5. Demand Elegant Solutions
Before finalizing non-trivial changes:

Ask:
> "Is there a more elegant implementation?"

Avoid:
- hacks
- unnecessary complexity
- over-engineering

Prefer:
- simple solutions
- minimal code change
- clarity

## 6. Autonomous Bug Fixing
When debugging:

1. Identify failing behavior
2. Inspect logs/errors/tests
3. Find root cause
4. Implement fix
5. Verify correctness

Avoid requiring unnecessary user interaction.

## Core Engineering Principles

### Simplicity First
Make every change as simple as possible and minimize code impact.

### Root Cause Thinking
Never implement temporary fixes. Always resolve the underlying issue.
