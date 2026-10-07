---
name: research-recipe-audit
description: Compare a research recipe with its actual implementation, reference method and execution commands; find objective, data, checkpoint, weighting, clipping, sampler or API drift before a costly experiment or replication claim. Use for replication, ablation fidelity and training-pipeline changes, not general literature ideation.
---

# Research recipe audit

Start from the requested scientific comparison, frozen recipe and actual source/commands. Do not treat the author's summary, matching model names or passing synthetic tests as evidence of recipe equivalence.

Trace the complete path: supplied data → fitting membership → preprocessing → teacher/initializer ancestry → predictions/cache mapping → targets and weights → model outputs/loss → sampling/updates → checkpoint selection → exported predictions → evaluated/served blend.

Write a compact difference table distinguishing required matches, intentional adaptations and accidental mismatches. Check coefficient arithmetic, averaging versus summing, normalization, clipping order, row masks, teacher detachment, auxiliary-head location and loss denominator. Inspect CLI defaults and incompatible flags on the exact final commands. Profile commands must exercise the declared shape/path without accidentally enabling a rejected continuation mode.

Source identity and semantics are separate: an array hash proves bytes, not correct producer/membership; a teacher family label does not prove equivalent ancestry. Bind row/group order, prediction epoch, dtype/shape and original-label weighting. When comparing implementations, use algebra or a separately constructed tiny example; never use the production function as its own oracle.

Keep checks proportional to this arm. Do not require unused families, a new framework or every future ablation before a simpler valid comparison. A reference mismatch is not proof a variant is bad; register it as a distinct scientific attempt when useful.

Return findings first: path/location, trigger, scientific consequence, contract/equation violated, and smallest correction/check. Then return the recipe-difference table, exact commands/source revisions reviewed, checks actually executed and remaining limitations. Do not launch expensive fits unless separately authorized. Independent review requires a different implementation session.
