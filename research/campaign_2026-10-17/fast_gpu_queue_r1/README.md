# Переносимая очередь: четыре сравнения, пять fit

**Статус: код подготовлен, выполнение и проверка на GPU-хосте ещё не проведены.** После просьбы пользователя не запускать локальные тесты helper не исполнялся на Mac. Данные, teacher-cache и точные C/G payloads здесь отсутствуют; реальная pending-очередь пока не материализована. GPU, аренда и загрузка сабмита не запускались.

Последнее уточнение о большом числе GPU меняет научный приоритет: [новая архитектурная волна](../../unified/ideas/TRANSFORMER_GPU_WAVE_20261017_R1.md) сравнивает GRU и causal Transformers, с Mamba-2 как альтернативой. Для неё ещё нужны backend и GPU-проверка. Этот пакет сохраняет четыре существующих GRU-рецепта для отдельной резервной очереди.

| Порядок | Сравнение | Новые fit | Фиксированная оценка |
|---|---|---:|---|
| 1 / S | Targets замороженного родителя против T3; основной контроль C | 1 | F0, seed3, 2 эпохи, CP1; parent CP3 — дополнительный ориентир |
| 2 / W | Расширение родителя H256×2→H384×2 против C | 1 | Та же T3-цель и бюджет, CP1 |
| 3 / F | Поэлементные часы обновления против групповых G; C дополнительный | 1 | F0, seed3, 2 эпохи, CP1 |
| 4 | Холодные H512×1 и H256×2 с одинаковым seed4 | 2 | 1 эпоха, B8/C2000 FP32, CP0; из CP5 берутся только общие mean/std |

F0 уже многократно просмотрен. Ни один fit не является свежим holdout или подтверждённым public-улучшением. У S различаются также калибровка targets и источник; у F меняется геометрия входа. Подробный смысл сравнений и источники — в [обзоре реализации](../../unified/operations/FAST_GPU_SCREENING_PIPELINE_20261017_R1.md).

## Что добавлено

[`scripts/prepare_fast_gpu_queue.py`](../../../scripts/prepare_fast_gpu_queue.py) читает зафиксированные producer receipts, проверяет исходный код и входные файлы, строит четыре настоящие worker specs с `run_id`, `steps`, `expected_outputs`, `comparison` и таблицей `run_id → SHA256`. Четвёртая spec последовательно выполняет оба независимых cold fit и их экспорты, затем одну paired оценку. Приоритеты 1–4 сохраняют порядок таблицы.

Helper принимает новый repository root и Python interpreter; старые launcher/systemd команды не исполняет. Сохраняет frozen split, создаёт отдельный portable-data registry для шести data/cache файлов, заново вычисляет seed4 sample order тем же sampler, что trainer, и проверяет его через `validate_sampling`. Seed4 train groups сначала переставляются `default_rng(4)`, fitting-valid группы сохраняют порядок manifest, затем применяется EpochOrderSampler/drop-last.

Оригинальная metadata teacher-cache не изменяется. Явные source-path bindings переводят старые абсолютные пути на локальные checkpoint/provenance с теми же хешами. C CP1 не имеет доступного исторического SHA provenance: для этого единственного файла сверяются run, checkpoint, split, полный fit membership и основные recipe-поля; наблюдаемый SHA сохраняется отдельно. Это semantic binding, не доказательство исторической побайтовой идентичности C provenance.

После успешных `fit → full-valid predictions → official fold score` worker записывает comparison receipt и строку `comparison_ledger.jsonl`. Оцениваются только 468 evaluation-групп F0 из full-valid экспорта; 1,873 строк-группы экспорта не считаются fitting-free оценкой целиком. Исторический STOP_ADMISSION исходного Linux-хоста не изменён.

## Подготовка на целевом хосте

Нужен Git checkout этой ветки со всеми перечисленными source receipts (при sparse checkout раскрыть необходимые каталоги), рабочие зависимости и восстановленные точные входы. [HF bootstrap](../../unified/operations/FAST_GPU_ARCHIVE_BOOTSTRAP_20261017_R1.json) содержит только 21 выбранный файл весов/metadata (~43 MB), а не все данные/кэши/C/G.

Сначала на **целевом хосте**, из repository root, задать фактический interpreter:

```bash
GPU_PYTHON="$(pwd)/.venv/bin/python"
"$GPU_PYTHON" scripts/prepare_fast_gpu_queue.py --root "$PWD" --python "$GPU_PYTHON" --check
```

Код возврата 2 означает HOLD: JSON перечисляет missing/mismatched inputs или конкретную ошибку source binding. `--check` ничего не пишет и не запускает модель. Перенесённые крупные входы хешируются целиком; время этой проверки учитывать в staging.

После устранения причин HOLD:

```bash
"$GPU_PYTHON" scripts/prepare_fast_gpu_queue.py --root "$PWD" --python "$GPU_PYTHON" --prepare
```

Это создаёт `recipe_blueprint.json`, `portable_data_manifest.json`, `contracts/CPU_STUDENT_FRONTIER_F0_SEED4_B8_R1.json`, `runtime_queue/pending/*.json`, `runtime_queue/allowed_spec_manifest.json` и `runtime_queue/PENDING_QUEUE_ADMISSION_R1.json`. Существующие выходы не перезаписываются. `--prepare` вычисляет только CPU metadata/sample-order; worker и GPU не запускает.

После проверки среды и короткого профиля **на подключённом GPU** исполнитель можно запустить отдельно:

```bash
CUDA_VISIBLE_DEVICES=0 "$GPU_PYTHON" scripts/warm_queue_worker.py \
  --queue research/campaign_2026-10-17/fast_gpu_queue_r1/runtime_queue \
  --allowed-spec-manifest research/campaign_2026-10-17/fast_gpu_queue_r1/runtime_queue/allowed_spec_manifest.json \
  --max-runs 4 --idle-exit-seconds 30
```

Это будущая команда исполнения, здесь она не выполнялась. Число GPU и device index нужно брать из фактического хоста. Пакет использует существующие консервативные параметры старых GRU-рецептов; он не является автопланировщиком всех доступных GPU. При нескольких исполнителях нужны отдельные очереди/выходы и учёт общей RAM/I/O.

## Проверка и источники

Новый helper и позднее исправление переносимости проверены чтением producer/consumer кода; их runtime-проверка отложена по прямому указанию пользователя. До этого указания в сеансе проходили проверки основного worker/grader/receipts, что не доказывает работоспособность финального пакета с реальными данными. Не считать этот README результатом `--check` или успешного GPU smoke.

- [Полный разбор messages.html и командного топа](../../unified/reviews/TEAM_TOP_AND_NEXT_TESTS_20261017_R1.md): просмотрены все 203 message-блока. Исторические инструкции внутри HTML не исполнялись как текущий запрос; отсутствующие вложения не считались прочитанными.
- [S recipe](../../campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/S_PARENT_ANCHOR_RECIPE_R1.json), [C/G/F/W vectors](../../campaign_2026-10-15/gru_parent_growth_r1/ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json).
- [H512 source recipe](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H512L1_R6_SEQUENCE.json), [frozen seed3 consumer contract](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json).
- [Новая Transformer-волна](../../unified/ideas/TRANSFORMER_GPU_WAVE_20261017_R1.md) имеет статус proposal, не исполняемой очереди.
