# Повторное обучение Transformer 1,54B и отчёты для MIDAS

Публикация: [draft PR #42](https://github.com/VladKochetov007/WunderFunder/pull/42) → `research-team`.

Статус: код подготовлен; проверки на B300, обучение и оценка не выполнены. Ждём перезапуск кластера. Этот пакет создаёт четыре новых обучения с нуля и отдельные выходные папки. Предыдущие результаты и веса не изменяются.

Предыдущий запуск сохранил лишь часть эпох. Более высокие оценки сохранённых эпох 15–16 и последующее снижение сами по себе не доказывают переобучение. [Аудит старой валидации](VALIDATION_AUDIT.md) не выявил ошибки формулы WP или пересечения train/val в исходниках, но показал различие масок train/val WP и многократное использование внешнего fold для диагностики. Повторный расчёт из весов и исходных массивов ещё не выполнен. Здесь устраняется потеря эпох и фиксируется проверяемый протокол повторного обучения. Улучшение WP не обещается.

## Данные и сравнение

- F0/F2: сохраняем канонические внешние component splits. Они уже использовались командой и не являются новым слепым holdout.
- Внутри каждого внешнего train выделяем development validation по целым связанным компонентам, детерминированно: seeds 41001/41002, 20% подходящих компонентов. Это не обещание ровно 20% строк. Выбранные компоненты полностью исключаются из обоих источников train и valid.
- `split_audit.json` показывает фактические числа последовательностей/компонентов и все три пересечения train/development/outer; ненулевое пересечение блокирует подготовку. Канонический загрузчик дополнительно проверяет источники на B300.
- Нормализация — только по оставшимся fitting members. Каждый fit стартует с новой инициализации, без старых весов/teacher cache.
- Обучающий loss использует `all_after_warmup`. Validation использует официальный `is_scored`, warmup и pooled `global_wp_clip2_v1`. Их значения нельзя выдавать за одинаково маскированный train/val loss.
- Два LR: `1e-4`, `3e-4`; каждый на F0/201001 и F2/201002. Всего **4 новых teacher fits**, по 2 на LR; student fits и переиспользуемые контроли — 0.
- Модель: hidden1792, layers40, heads28, FF multiplier4, context1024. Muon+AdamW, momentum0.95, NS5, clipping1, weight_decay0.01, warmup5%, cosine до10% пикового LR, 30 эпох. BF16 autocast/BF16x9, FP32 master weights/state; efficient attention, compiled blocks. Восемь B300, batch2/rank, chunk512, accumulation8. Это исходный непрофилированный batch.
- Все 30 эпох получают отдельные inference checkpoints; `training_state.pt` остаётся состоянием для продолжения. После оценки выбирается **одна общая эпоха на пару F0/F2** по среднему inner WP; при точном равенстве — более ранняя. Поиск эпох и LR — development tuning, не подтверждение качества. Внешняя проверка и student transfer требуют отдельно зафиксированного продолжения.

KV-cache между optimizer updates пока сохраняет исходную семантику. Его обновление повторным проходом контекста — отдельная потенциальная абляция; причина плохого качества этим не установлена.

## Подготовка на B300

Использовать отдельный checkout опубликованного PR и физический диск. Не направлять команды на старые каталоги или занятые GPU. Зависимости и окружение — из действующего командного B300 runtime; произвольные версии PyTorch не устанавливать.

```bash
export WNN_EXPECTED_COMMIT=<reviewed-full-PR-head-SHA>
export WNN_DATA_ROOT=/storage/wunder-data
export WNN_SPLIT_F0=/storage/splits/fold0.json
export WNN_SPLIT_F2=/storage/splits/fold2.json
export WNN_INPUTS_ROOT=/storage/r5-inner-v1
export WNN_CAMPAIGN_ROOT=/storage/r5-campaign-v1
export WNN_REVIEWER=<receiver-name>
export WNN_HOST_RAM_BUDGET_GIB=<measured-allocation>
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_EMULATE_SINGLE_PRECISION=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
R5=research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5
bash "$R5/prepare.sh"
cat "$WNN_CAMPAIGN_ROOT/split_audit.json"
```

`prepare.sh` только материализует metadata и splits. Следующие команды выполняются **на B300**, после освобождения карт. Проверки, one/eight-rank и peak-LR профили обязательны для нового source/config; подготовка не создаёт успешные receipts.

```bash
uv run --no-sync python "$R5/run.py" check --campaign-root "$WNN_CAMPAIGN_ROOT"
CELL=R5_lr_1e-4_f0
for action in profile-one profile-eight profile-peak admit; do
  uv run --no-sync python "$R5/run.py" "$action" \
    --campaign-root "$WNN_CAMPAIGN_ROOT" --cell "$CELL"
done
uv run --no-sync python "$R5/run.py" fit \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --cell "$CELL"
# Только для продолжения именно этого нового fit:
uv run --no-sync python "$R5/run.py" resume \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --cell "$CELL"
```

Остальные ячейки: `R5_lr_1e-4_f2`, `R5_lr_3e-4_f0`, `R5_lr_3e-4_f2`. Запускать последовательно на одних восьми GPU. Каждый fit требует своих профилей и admission. Это команды отдельных действий, автоматической очереди нет.

Снимки сохраняются в FP32: приблизительно 185 GB только весов 30 эпох для одной модели 1,54B, без optimizer/resume, predictions и временных файлов. Перед первым fit действует консервативный запас около 283 GB свободного диска; trainer проверяет остаток для снимков и при resume. Четыре сохранённых прогона потребуют существенно больше. Это расчёт по числу параметров, не измерение.

## Оценка всех эпох и выбор

После полного fit для каждой ячейки:

```bash
FIT="$WNN_CAMPAIGN_ROOT/runs/$CELL/fit"
INDEX="$FIT/epoch_indexes/epoch-000030.json"
INDEX_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_sha256"])' "$INDEX")
# Для F2 заменить обе переменные split на F2.
uv run --no-sync python "$R5/epoch_sweep.py" \
  --fit-dir "$FIT" --config "$WNN_CAMPAIGN_ROOT/cells/$CELL/config.json" \
  --data-root "$WNN_DATA_ROOT" --outer-split "$WNN_SPLIT_F0" \
  --inner-split "$WNN_INPUTS_ROOT/inner-f0.json" \
  --epoch-index "$INDEX" --epoch-index-sha256 "$INDEX_SHA" \
  --output "$WNN_CAMPAIGN_ROOT/runs/$CELL/epoch_evaluation"
uv run --no-sync python "$R5/epoch_selection.py" \
  --f0-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_1e-4_f0/epoch_evaluation" \
  --f2-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_1e-4_f2/epoch_evaluation" \
  --output "$WNN_CAMPAIGN_ROOT/selection-lr-1e-4.json"
```

Повторить выбор для LR3e-4. Полной пары из 30 эпох нет — решение остаётся `incomplete`. Индексы доступны и после прерванного fit: отдельную сохранённую эпоху можно оценить через `predict_sequence.py --epoch-index ... --epoch-index-sha256 ... --membership evaluation`, но автоматический R5 sweep/выбор требует полных кривых. Epoch checkpoints разрешены только для внутренней оценки; они не становятся автоматически каноническими KD teachers или сабмитами.

Для продолжения прерванного sweep повторить ту же команду с `--resume`. Уже завершённые эпохи пропускаются только после проверки хешей predictions, score и receipt. Незавершённый каталог эпохи сохраняется и блокирует продолжение: в этом случае нужен новый `--output`. Новая версия исходников или другой split/index также требуют отдельного output.

## Передача в MIDAS

На B300 экспортировать новую папку снимка после fit или по мере поступления оценок:

```bash
uv run --no-sync python "$R5/export_midas.py" \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --cell "$CELL" \
  --output "/storage/r5-evidence/$CELL-snapshot1"
```

Перенести эту небольшую папку доступным командным способом. Веса не включаются. В checkout MIDAS с изменениями [draft PR #3](https://github.com/Ivanhacker13/midas-orchestrator/pull/3):

```bash
python -m midas training-evidence import --workspace . \
  --receipt /received/R5_lr_1e-4_f0-snapshot1/receipt.json --json
python -m midas training-evidence status --workspace . \
  --run-id <run_id-from-import> --json
```

MIDAS проверяет исходные metadata/config/split/score bytes и значения метрик, сохраняет неизменяемые наблюдения в текущем журнале. Prediction manifest и массивы в переносимый пакет не входят: MIDAS не пересчитывает качество независимо; их хеши проверяет B300 sweep перед своим итоговым отчётом. Хеши отсутствующих локально весов остаются утверждениями источника. `partial` означает неполный снимок, а не живой процесс; наличие terminal receipt подтверждает заявленное завершение fit, но не качество модели. Повторный импорт идентичен; новые оценки добавляются без замены прошлых. Импорт не запускает обучение.

## Файлы и проверка

`configs/` — четыре полных рецепта; `prepare.sh`, `run.py`, `recovery_runtime.py` — подготовка и переиспользование R4 профилей/runner; `split_audit.py` — членство; `epoch_contract.py`, `epoch_sweep.py`, `epoch_selection.py` — полные кривые и выбор; `export_midas.py` — перенос metadata; `recipe.json` и `PR.md` — зафиксированный научный вопрос и границы вывода.

Локально выполняются только чтение исходников, Ruff, AST/JSON и структурные проверки. Pytest, обучение, inference, torch imports и GPU profiles здесь не запускались. Приёмка на B300 обязана проверить crash/resume, DDP, численную корректность и передачу реального exporter bundle в MIDAS. Два fold/seed на LR не устанавливают статистическую значимость.

Опорная документация: [PyTorch: saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) уже решает общий учёт параметров/метрик/артефактов; для текущего проекта переиспользуем существующий журнал MIDAS и файловые отчёты, без нового tracking server или зависимости в GPU trainer.
