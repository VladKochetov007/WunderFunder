# Fixed error diagnosis

These are descriptive results on previously exposed OOF chains. No forecast model, ensemble weights or new candidate WP were fitted.

| Profile | WP | Deficit |
|---|---:|---:|
| current_student4 | 0.702379667615 | 0.29762033 |
| strong4 | 0.702367225515 | 0.29763277 |
| fixed3 | 0.701742429605 | 0.29825757 |

| Retrospective class | Deficit share | Metric-weight share | Mean mismatch | Amplitude mismatch | Within-class covariance shortfall |
|---|---:|---:|---:|---:|---:|
| 0 | 22.62% | 30.12% | 0.003388 | 0.003115 | 0.060818 |
| 1 | 12.65% | 16.60% | 0.001107 | 0.000989 | 0.035563 |
| 2 | 7.03% | 8.72% | 0.000195 | 0.000441 | 0.020274 |
| 3 | 57.70% | 44.55% | 0.004012 | 0.027926 | 0.139795 |

| Gradient-response probe | t0 skill | t1 skill | Mean skill | Conditional 95% interval |
|---|---:|---:|---:|---| 
| prediction_confidence | -0.7843% | -0.3445% | -0.5644% | [-0.8445%, -0.2790%] |
| plus_current211 | -2.4828% | -1.9325% | -2.2077% | [-2.7471%, -1.6991%] |
| plus_history30 | -3.0297% | -2.7823% | -2.9060% | [-3.6095%, -2.2356%] |
| plus_member_deviations10 | -3.1923% | -2.9595% | -3.0759% | [-3.8933%, -2.2922%] |

The probe uses fitting-half normalization of its response and predictors. Its skill is a heldout weighted squared-error diagnostic in post-clip gradient space, not a WP gain or deployable correction. Retrospective class is used for stratification and reporting only. Shared errors among highly dependent profiles do not establish an irreducible ceiling.
