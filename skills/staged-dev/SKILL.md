---
name: staged-dev
description: "Staged, human-gated development for large, complex, and domain-heavy projects."
disable-model-invocation: true
---

# Staged development

Work on a large project in stages. Each stage ends at a gate where the human
reviews and approves.

## Pipeline

### 1. Analyze and align

Read code bounded by the task goal. Present your understanding and your
approach.

Done when the human confirms the approach. Do not write code.

### 2. Implement and review

Write the code for the confirmed approach.

Done when the change is complete and the human approves the code. Do not test.

### 3. Build and test

Run the command the human confirmed.

On success, go to Done. On any error, stop and report. Do not diagnose and
fix in a loop.

### Done

Report what you ran and what passed.

## Rules

### Scope lock

Read and change only what the task goal covers. Leave unrelated problems you
see on the way.

### Sub-agents are read-only

Give a sub-agent one narrow task. A sub-agent reads only what its task covers.
It reports on that task only.

Only the main agent edits, builds, and tests. A sub-agent does not analyze
content outside its task.

### Build and test commands

Take the build and test commands from the human. Do not search for them.

Handle them only after the code review (Stage 2).

- If the human gave a command, tailor it to this change and present it. Run it
  only after the human confirms.
- If the human gave no command, ask what to run.

### Material change returns to the gate

If the approach changes in substance, return to the last gate and get a new
approval.
