# Autonomous local geometry continuation

[Registered R3 plan](plan.json) resumes the local GPU loop under the user’s explicit continue instruction. It compares the same four causal feature arms for eight epochs at H128×2, three paired seeds and fixed final checkpoints. This tests a longer total optimization budget with its corresponding cosine clock, not duration alone. The same tiny fitting pool and exposed evaluation remain limitations; no quality result is claimed at registration.

Source/data/preprocessing are reused from the [frozen R2 packet](../moment_geometry_r1/README.md). Exact execution, progress, artifacts and independently checked results will be linked here as they land. The separate team experiments remain owned by the team.
