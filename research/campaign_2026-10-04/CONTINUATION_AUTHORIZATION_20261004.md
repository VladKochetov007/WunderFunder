# Continuation timing amendment

The user updated the active objective: “give new ideas, generate submissions for manual upload, run on kaggle and locally until I explicitly tell you to stop new experiments and wrap all we have”. This supersedes the earlier 20:00 Kyiv scheduling cutoff. Root stopped `wnn-e424-hard-stop.timer` and will launch the controller with an explicit `--until-user-stop` flag.

Each scientific attempt still has a sealed timeout and exact inputs, memory/CPU/GPU enforcement, monitored teardown, incident handling, and the accepted cumulative 80 device-GPU-hour ceiling. TPU time is separate. A budget exhaustion is not permission to increase the ceiling. No competition uploads are authorized.

The original execution contract/runner READY bytes remain immutable; this timing amendment changes controller scheduling only. Existing private Kaggle jobs retain their own bounded runtimes.
