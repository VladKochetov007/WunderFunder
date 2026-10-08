# ADR0023: faithful method reconstruction before variants

Status: accepted, 2026-10-08. Authority: the user's latest correction about reproducing the successful teammate method with independent code.

## Decision

The primary heterogeneous-distillation branch must reconstruct the complete method: teacher architectures and fitting coverage, teacher objectives and optimizer stages, seed-specific parent ancestry, target construction, student schedule, and the submitted student blend. Matching only student architecture or delivering a single student is insufficient. Read teammate sources for methods; independently author all implementation and train our weights.

Start with independently trained seed9 and seed99 students and a fixed equal pair. Seeds54354 and7753191 extend the ensemble after the first endpoint is available. Seeds provide independent training variation; they do not replace mechanism comparisons. More students are research candidates, subject to an independently checked final serving path.

Preserve a closest-known reference branch before changing it. Where the final reference ancestry is unavailable, bind each known detail to its source and distinguish confirmed endpoint details, predecessor recipes, and unknowns. Do not claim exact replication from an approximate predecessor. Missing receipts are not a reason to abandon the mechanism or hold all runnable stages indefinitely.

Muon has existing positive project evidence: `research/campaign_2026-10-03/scores/muonfull_report.json` reports fold0 official WP0.6802049097 versus matched control0.6772058272. Our current clean reference `c20261003_combo_s3_f0` also declares Muon and OneCycle in provenance. AdamW must not silently replace this training choice. Audit optimization separately at teacher, parent, and final-student stages: a Muon-trained parent does not establish that the reference final student used Muon. Preserve completed AdamW attempts as distinct evidence rather than relabeling them.

## Comparison branches

- Closest method versus our existing best recipe, with complete served endpoints.
- Muon versus AdamW under matched data, architecture, initialization, objective, order, budget, and explicit LR group scales; compare schedules separately where matching is impossible.
- Unnormalized teacher sum with coefficient mass0.8125 versus normalization by0.8125. Keep individual family targets and clipping order explicit. Normalization changes amplitude and saturation, so it is a treatment, not harmless cleanup.
- Own baseline versus an allowed organizer baseline. Verify permitted source and preprocessing compatibility; keep teammate weights excluded.
- Reference schedule versus bounded alternatives, including a plateau/restart/noise arm after independent ideation and an explicit finite execution recipe. Do not choose fitting or checkpoint behavior using the evaluated fitting groups as held-out evidence.
- Interactions of promising components, registered with parent links and judged at the complete blended or distilled endpoint. Component weakness alone does not reject a combination.

Freeze commands and source/config/data identities per attempt. Prepare and execute stages in parallel on admitted Kaggle/Vast capacity, amortizing unchanged input verification and staging. Keep healthy current jobs intact. Local execution remains under the latest five-core morning cap and existing RAM/VRAM boundaries.

The user explicitly requests one Astra ideation role for scheduling and combinations; this is a scoped exception to the default Luna/max native-agent preference. Other native implementation roles retain Luna/max. The ideator owns a source-grounded proposal and ADR/plan artifact, not spending or self-approved fits.

## Consequences

The existing common-parent/single-student and normalized submissions remain valid historical variants, but cannot stand in for the closest reconstruction. ADR0019 remains historical authorization for the earlier bounded endpoint; this decision supersedes its priority for the next heterogeneous-distillation endpoint. Public score reproduction requires actual public evidence. The active research objective and independent-code rule remain unchanged.
