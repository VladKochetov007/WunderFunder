# E421 fixed categorical branch/U readout screen

Two reciprocal component partitions: 6158 sampled rows, 460 sequences, 140 components. Same frozen strong GRU states, seed271828,256updates,batch512,65,536parameter budget (within2%), fixed learning rate0.001 and fit-only moments. Unit fit-row weights, unlike the earlier full-training propensity objective. No validation checkpoint selection.

| Head | Pooled global WP |
|---|---:|
| unchanged | 0.682090227 |
| global_affine | 0.680303423 |
| wmse | 0.492579299 |
| joint_ce | 0.630341244 |
| joint_tilted_wmse | 0.580672273 |
| candidate | 0.593982835 |
| factorized_tilted_wmse | 0.548662423 |

Factorized CE minus unchanged: -0.088107392, component CI[-0.113134919,-0.066458977].

Factorized hybrid minus unchanged: -0.133427804, component CI[-0.167645119,-0.102874977].

Every fixed arm is retained. The factorized CE head also loses the joint504 CE control. Both outperform the newly fitted wMSE capacity control; that does not overcome the untouched backbone. The sparse sampled fit and frozen representation differ from a full-data/end-to-end categorical model. These results do not close that planned architectural intervention.

Raw reports are `feature_branch_results/branch_f0_d{0,1}_cpu_v1/report.json`; pooled comparisons and input hashes in `checks/e421_pooled_v1.json`. Native deployment and end-to-end training are not claimed.
