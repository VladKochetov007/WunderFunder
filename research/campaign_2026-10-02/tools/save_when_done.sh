#!/usr/bin/env bash
# Wait for the running TCN fold-1 job to end, then commit its job record and log (predictions/checkpoints are saved by
# the job itself under cache/ and artifacts/runs/). Launches nothing.
cd "$(dirname "$0")/../../.."
while systemctl --user is-active -q wnn-top1-c20261002_dtcn128l12_f1.service; do sleep 30; done
research/campaign_2026-10-01/tools/commit.sh "research(arch): TCN fold-1 job finished (record and log saved; not scored)" \
  research/campaign_2026-09-29/jobs/c20261002_dtcn128l12_f1.json research/campaign_2026-09-29/logs/c20261002_dtcn128l12_f1.log \
  research/campaign_2026-10-01/jobs/gpu_queue6.log > research/campaign_2026-10-02/tools/save_when_done.log 2>&1
ls cache | grep dtcn128l12a05_s3_f1 >> research/campaign_2026-10-02/tools/save_when_done.log
