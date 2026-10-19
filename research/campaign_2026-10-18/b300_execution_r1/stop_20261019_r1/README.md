# Requested stop and preserved results, 19 October 2026

The user requests preserving results and stopping all our training on `epanov_s@10.14.14.26`. New scientific admission is no longer authorized. **Remote stop is not confirmed.** The old SSH session returned `client_loop: send disconnect: Connection reset`. The single new `ssh epanov_s@10.14.14.26` connection was closed by the host before a banner/password prompt. No remote command or stop signal was executed in this turn. This is not evidence of an authentication/password failure or of the exact networking cause.

## Preserved scientific evidence

The [API snapshot](GITHUB_SNAPSHOT.json) preserves complete PR metadata/comments and the six most recently updated SemyonEpanov comments per PR. These are dated published observations, not a fresh filesystem/process inspection.

| Experiment | Latest published stage | Partial quality evidence |
|---|---|---|
| [PR15](https://github.com/VladKochetov007/WunderFunder/pull/15#issuecomment-6256107437), 13:40:43 UTC | 64/120 scored fits; 16 in flight including delivery; 40 unstarted | 16/30 scored cells per arm; table below |
| [R01/PR25](https://github.com/VladKochetov007/WunderFunder/pull/25#issuecomment-6257196087), recovery comment 13:02:58 UTC | 0 published paired grades; 12 rejected/incomplete publication receipt chains | Scores cannot be inferred from the rejected chains |
| [Age/PR26](https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6256781766), 13:50:04 UTC | 0/30 scored pairs; 4 active paired cells | No reported quality comparison |
| [Innovations/PR27](https://github.com/VladKochetov007/WunderFunder/pull/27#issuecomment-6256782166), 13:50:11 UTC | 0/30 scored pairs; 4 active paired cells | No reported quality comparison |

PR15 equal-fold means over available matched cells:

| Arm | Mean WP | Paired delta versus baseline | Scored cells |
|---|---:|---:|---:|
| baseline | 0.67824425 | — | 16/30 |
| moments | 0.67809108 | -0.00015317 | 16/30 |
| dictionary | 0.67756978 | -0.00067447 | 16/30 |
| combined | 0.67702747 | -0.00121678 | 16/30 |

These partial development-fold means do not establish significance or public transfer. The current table shows no positive gain over baseline; no model is promoted. Keep negative and inconclusive evidence.

The R01 recovery publisher reports missing `grade.stdout.txt` for twelve fold/seed grade chains. The sealed delivery helper writes `stdout.txt`/`stderr.txt`; this is a concrete publication filename mismatch, not proof of failed model training or proof that no grade JSON exists. Raw grades/checkpoints require server access before numerical recovery can be claimed. No refit or regrade is authorized by this stop request.

## Node action still required

With an already working shell on the node, `systemctl --user stop 'tabdata-*'` stops our tabdata service/timer/path units, including queue owners and scientific children; it does not target unrelated unit names. Preserve `/data/epanov_s/tabdata` and its artifacts. Verify all our execution units inactive, their cgroups empty, and no project-owned CUDA/training processes remain. Check foreign clients separately rather than claiming preservation without a live observation. Also disable our recurring reporting timers if the whole tabdata campaign must remain dormant.

This packet records a requested stop and the available results. It deliberately makes no claim that workloads have already stopped, that the last current epoch was saved, or that node storage was downloaded after this request.
