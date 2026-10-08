# H272 S4 served-head control: independent grade

Recorded: 2026-10-08T16:29:12.660522+00:00

This grade compares fixed raw 50/50 parent blends on the 468 valid-source fold-0 evaluation groups and 141 connected components. Fold 0 was previously exposed. The result is development evidence and has no public-score forecast.

## Point grades

| Endpoint | WP | t0 | t1 |
|---|---:|---:|---:|
| parent | 0.6810293773 | 0.6741514875 | 0.6879072672 |
| control_halfblend | 0.6818613275 | 0.6746870931 | 0.6890355618 |
| h272_hidden_only_halfblend | 0.6814010973 | 0.6745005708 | 0.6883016238 |
| h272_multiplicative_halfblend | 0.6810719865 | 0.6741683779 | 0.6879755951 |

| Paired comparison | Candidate WP | Reference WP | Delta WP | Component bootstrap 95% interval | Bootstrap fraction delta <= 0 |
|---|---:|---:|---:|---:|---:|
| control_halfblend_vs_parent | 0.6818613275 | 0.6810293773 | 0.0008319501 | [0.0000433563, 0.0016651434] | 0.0205 |
| h272_hidden_only_halfblend_vs_parent | 0.6814010973 | 0.6810293773 | 0.0003717200 | [-0.0002182466, 0.0009131812] | 0.0990 |
| h272_multiplicative_halfblend_vs_parent | 0.6810719865 | 0.6810293773 | 0.0000426091 | [-0.0000700882, 0.0001559199] | 0.2170 |
| h272_hidden_only_halfblend_vs_control | 0.6814010973 | 0.6818613275 | -0.0004602302 | [-0.0012896346, 0.0004294132] | 0.8410 |
| h272_multiplicative_halfblend_vs_control | 0.6810719865 | 0.6818613275 | -0.0007893410 | [-0.0015942471, -0.0000020714] | 0.9755 |

The primary endpoint is the ordinary GCorr head fixed 50/50 raw blend against its parent. The two H272 readout comparisons use the same parent and fixed 50/50 weights. The direct H272-versus-GCorr comparisons answer whether the readout adds beyond the ordinary head endpoint. No blend weights were selected here.

H272 hidden-only versus ordinary GCorr control: -0.0004602302 WP with conditional component-bootstrap interval [-0.0012896346, 0.0004294132]. H272 multiplicative/contextual versus ordinary GCorr control: -0.0007893410 WP with interval [-0.0015942471, -0.0000020714].

## Membership and limits

The control fits on 1,405 valid groups; H272's recorded readout fit uses 10,535 train plus 1,405 valid groups. Each excludes the 468 evaluation groups and their 141 components. Both bind the same parent checkpoint. H272's propensity ancestor is also checked against the source-qualified split. These different fit memberships mean the endpoint contrast does not isolate a pure method effect.

All comparisons use the same frozen official selected mask: need_prediction warmup exclusion (first 99 rows) AND is_scored, restricted to the 468 evaluation groups, totaling 808,863 scored rows. Prediction, target, split, recipe, checkpoint, provenance, sidecar, and receipt hashes are recorded in the JSON. H272's two fixed half-blend scores independently reproduce its delivered result within 1e-8.

The 2,000-resample paired component bootstrap uses seed 271828. Its intervals condition on these fixed predictions. They do not capture fit-seed variation, post-selection uncertainty, or public transfer. No training, GPU, export, package, submission, metric change, or campaign-state mutation was performed.

## Reproduction

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=4000000000 WNN_RUNTIME_MAX=1200s CUDA_VISIBLE_DEVICES= /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python /home/vlad/development/wunder/worktrees/h272-s4-control-grade-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/grades/h272_s4_control_v1/grade_h272_s4_control_v1.py --project-root /home/vlad/development/wunder/wnn_connectome_starterpack --control-worktree /home/vlad/development/wunder/worktrees/e529-s4-control-profile-fa615-20261008 --output /home/vlad/development/wunder/worktrees/h272-s4-control-grade-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/grades/h272_s4_control_v1/H272_S4_SERVED_HEAD_CONTROL_INDEPENDENT_GRADE_20261008_R1.json --markdown /home/vlad/development/wunder/worktrees/h272-s4-control-grade-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/grades/h272_s4_control_v1/H272_S4_SERVED_HEAD_CONTROL_INDEPENDENT_GRADE_20261008_R1.md

The JSON includes Python argv, resource-wrapper settings, cgroup snapshots, per-component centered moments for each endpoint and target, and complete input SHA-256 identities.
