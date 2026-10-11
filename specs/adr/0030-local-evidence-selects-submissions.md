# ADR0030: local research evidence selects submission candidates

Status: accepted, 2026-10-11. Clarifies ADR0001 and the validation policy.

The user clarified that requests for the strongest public submissions mean candidates formed from our research and valid local evidence; the public leaderboard confirms transfer. Selecting old members by public scores alone misstates that intent.

Rank recipes by distinct-seed means on the declared official metric and evaluate fixed mixtures locally. Keep fitting/teacher ancestry, exposure, checkpoint selection and serving validity explicit. Public feedback can identify transfer failures, but a public score or numerical expected-public forecast is not a packaging prerequisite. An exposed local gain remains exposed evidence; it is not a numerical public forecast.

Include strong locally evaluated runs in the submission portfolio, or record the concrete reason for omission. A fold-trained checkpoint can be a valid exploratory submission while its all-data successor trains; mark its smaller fitting membership and seed-selection exposure. Do not claim held-out performance for all-data fits. Preserve local scoring and final-training decisions as separate artifacts.

Current application: prioritize the h256 Muon-all/prop0 four-epoch recipe (two distinct seeds mean approximately0.6709) and the alpha0.9 six-epoch distilled student (0.67103, one seed), with fixed locally evaluated blends. Existing Z5/Z4 packages remain optional references. Keep KD-loss, OOD, teacher-count and other branches active in parallel.
