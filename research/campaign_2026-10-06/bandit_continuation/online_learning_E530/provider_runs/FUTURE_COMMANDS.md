# E530 provider retry packets

The attempts in [ACCESS.json](ACCESS.json) stopped at local session/log storage; they produced no model proposal. These commands are prepared for a future authorized host with writable native-CLI session storage. Keep each attempt isolated and sequential. Do not change the managed sandbox or redirect credential/config files to bypass its boundary.

The shared watchdog pins its process tree to CPU 0 and stops it above 1 GiB summed RSS or 600 seconds. It writes only a small resource receipt and does not capture CLI transcripts. Run commands from the project root after confirming the provider route and permitted environment.

## Grok 4.7

```sh
python research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/tree_rss_watchdog.py \
  --cwd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/grok_47 \
  --receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/grok_47/WATCHDOG_GROK_FUTURE.json \
  --timeout-seconds 600 --max-rss-kib 1048576 -- \
  /home/vlad/.grok/bin/grok \
  --cwd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/grok_47 \
  --model grok-4.7 --no-subagents --permission-mode plan --max-turns 4 \
  --prompt-file /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/grok_47/PROMPT.md \
  --output-format plain
```

## Freebuff / GLM 5.3 Flash

Freebuff has no documented prompt-file/headless argument in the inspected help. Start its actual picker in the isolated directory, confirm the displayed model name, open a fresh chat, and provide the full `freebuff_glm/PROMPT.md` content. Do not use `--trust-agents` or invent a headless flag.

```sh
python research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/tree_rss_watchdog.py \
  --cwd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/freebuff_glm \
  --receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/freebuff_glm/WATCHDOG_FREEBUFF_FUTURE.json \
  --timeout-seconds 600 --max-rss-kib 1048576 -- \
  /home/vlad/.npm-global/bin/freebuff \
  --cwd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/freebuff_glm
```

## OpenCode / Muse

```sh
python research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/tree_rss_watchdog.py \
  --cwd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/opencode_muse \
  --receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/opencode_muse/WATCHDOG_OPENCODE_MUSE_FUTURE.json \
  --timeout-seconds 600 --max-rss-kib 1048576 -- \
  /home/vlad/.opencode/bin/opencode run \
  --dir /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/opencode_muse \
  --model opencode/muse-spark-1.3-contributor-free --format default \
  --file /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/online_learning_E530/provider_runs/opencode_muse/PROMPT.md \
  'Use the attached brief as the complete task.'
```

The historical Muse route previously returned `403 FreeTierError`; this is separate from the current local read-only log failure. Retry only when the CLI reaches the route normally, and record any current provider result as new evidence.

## OpenCode / Kimi

The Kimi prompt is ready at `opencode_kimi/PROMPT.md`, but no exact Kimi `provider/model` ID is verified. Do not substitute a guessed ID. First obtain a successful read-only model listing on an authorized writable host, then record the exact ID and make one separately monitored call.
