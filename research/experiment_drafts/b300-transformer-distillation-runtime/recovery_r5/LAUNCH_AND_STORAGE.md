# R5: запуск после перезапуска B300 и хранение весов

Статус: подготовка к запуску. Проверки на B300, обучение, оценка и очистка не
выполнялись. Это продолжение [R5](README.md), без новых научных ячеек: те же четыре
обучения Transformer 1,54B, 30 эпох, F0/F2, LR `1e-4`/`3e-4`, Muon+AdamW,
BF16/BF16x9. Кластер должен быть перезапущен, восемь карт — свободны.

## Что меняется в хранении

Сначала обучаем оба fold одного LR, оцениваем **все 30 эпох**, выбираем одну общую
эпоху по среднему inner WP, экспортируем отчёты, затем явно удаляем ненужные
снимки этой завершённой пары. После этого переходим ко второму LR.
Правило выбора, метрика, precision и train/validation splits не меняются.

При 1,54B параметров FP32 inference weights занимают примерно 6,16 GB на эпоху
(десятичные GB, без накладных расходов сериализации):

| Состояние | Приблизительный объём весов |
|---|---:|
| 30 эпох одной модели | 185 GB |
| Все эпохи четырёх моделей одновременно | 740 GB |
| Пик при обработке по парам: 60 эпох + два финала текущей пары + выбранные и финальные веса предыдущей пары | 407 GB |
| После обеих очисток: выбранные + фиксированные финальные веса четырёх моделей | 49 GB |

Это **объём весов, не требуемая ёмкость диска**. Дополнительно остаются
`training_state.pt` с оптимизатором/RNG, predictions всех эпох, профили, логи,
временные файлы и запас. Сохраняется прежний консервативный admission: около
283 GB свободного места **перед каждым новым fit**. Поэтому диск на 407 GB
недостаточен для гарантированного прохождения этого handoff. `df -h` и фактические
размеры артефактов проверяются на сервере; обходить storage admission нельзя.

Хранить только индивидуальный best каждого fold нельзя при нынешнем общем
правиле: лучшая по среднему F0/F2 эпоха может не быть индивидуальным best ни
одного fold. Неполные обучения не очищаются и сохраняют обычное продолжение.

## 1. Отдельный checkout и окружение

Работать в отдельном чистом checkout. Полный SHA берётся из проверенного head
этого PR; не использовать подвижный `research-team` как launch pin. Пример ниже
содержит пути-заполнители — их заменяет принимающая сторона.

```bash
git clone https://github.com/VladKochetov007/WunderFunder.git /storage/wunder-r5-source
cd /storage/wunder-r5-source
git fetch origin <reviewed-full-PR-head-SHA>
git checkout --detach <reviewed-full-PR-head-SHA>
R5=research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5
cp "$R5/receiver.env.example" /storage/r5-receiver.env
# Заполнить /storage/r5-receiver.env реальными путями, SHA и выделенной RAM.
source /storage/r5-receiver.env
export PYTHONPATH="$PWD/src"
df -h "$WNN_DATA_ROOT" "$(dirname "$WNN_CAMPAIGN_ROOT")"
nvidia-smi
```

`UV_PROJECT_ENVIRONMENT` указывает на существующий командный B300 runtime.
Все Python-команды используют `uv run --no-sync`: зависимости не обновляются.
`WNN_INPUTS_ROOT` и `WNN_CAMPAIGN_ROOT` — новые абсолютные пути на физическом
диске, вне checkout, `/tmp`, `/dev/shm` и старых запусков. Если свободны не все
восемь карт, дальше не переходить.

## 2. Проверка данных, splits и runtime

```bash
bash "$R5/launch.sh" prepare
cat "$WNN_CAMPAIGN_ROOT/split_audit.json"
bash "$R5/launch.sh" check
```

`prepare` проверяет точные хеши канонических outer F0/F2, материализует inner
splits и выводит аудит членства. Внутри каждого fold train/development/outer
не пересекаются. Цель inner — 20% доступных valid-source последовательностей;
выбор целых компонент может увеличить фактическую долю. Их train-source члены
тоже исключаются из fitting. Нормализация только по fitting.

Пересечения inner F0 с outer F2 и наоборот показаны отдельно и допустимы:
внешние folds уже использованы командой. Это development selection, не новый
слепой holdout. `check` выполняет receiver-проверки и записывает фактическую
приёмку; один `prepare` не подтверждает корректность DDP или B300 runtime.

## 3. Первая пара: LR 1e-4

Каждый `fit-cell` последовательно делает one-rank, eight-rank и peak-LR профили,
проверяет admission, затем запускает один fit на восьми B300. Остальные команды
не стартуют автоматически. Команды ниже выполнять по одной, после успешного
завершения предыдущей.

```bash
bash "$R5/launch.sh" fit-cell R5_lr_1e-4_f0
bash "$R5/launch.sh" fit-cell R5_lr_1e-4_f2
bash "$R5/launch.sh" finish-pair 1e-4
```

`finish-pair` запускает GPU inference/score по всем 30 эпохам обоих fold,
выбирает общую эпоху, создаёт два MIDAS bundle в
`$WNN_CAMPAIGN_ROOT/midas-evidence/` и hash-bound retention plan. Это может быть
существенная отдельная стоимость; профили и fit сами по себе sweep не заменяют.
Сохранить и перенести metadata bundles **до очистки**, вместе с selection,
retention plan и последующими retention receipts. Веса в MIDAS bundle не входят.

Просмотреть `selection-lr-1e-4.json`, `retention-lr-1e-4.json`, списки `folds[].remove`
и сохранённые файлы. Применить план отдельно, передав SHA256 **точных bytes
файла**, напечатанный `finish-pair` / `sha256sum`; это не встроенный digest поля
`plan_sha256`.

```bash
sha256sum "$WNN_CAMPAIGN_ROOT/retention-lr-1e-4.json"
bash "$R5/launch.sh" cleanup-pair 1e-4 <reviewed-raw-plan-sha256>
bash "$R5/launch.sh" verify-pair 1e-4
df -h "$WNN_CAMPAIGN_ROOT"
```

Удаляются только точные индексированные `epoch_checkpoints/*.pt`, кроме общей
выбранной эпохи. Остаются выбранный snapshot, фиксированный финал `model.pt`,
`training_state.pt`, все индексы, логи, полные кривые, scores и predictions.
Перед первым удалением сохраняется intent для обоих fit; хеши/размеры и
сохранённые данные проверяются. После завершения записываются receipts.
Чужие каталоги, symlinks и незавершённые fits не допускаются.

## 4. Вторая пара: LR 3e-4

```bash
bash "$R5/launch.sh" fit-cell R5_lr_3e-4_f0
bash "$R5/launch.sh" fit-cell R5_lr_3e-4_f2
bash "$R5/launch.sh" finish-pair 3e-4
sha256sum "$WNN_CAMPAIGN_ROOT/retention-lr-3e-4.json"
bash "$R5/launch.sh" cleanup-pair 3e-4 <reviewed-raw-plan-sha256>
bash "$R5/launch.sh" verify-pair 3e-4
```

Launcher не допустит вторую пару без проверенного завершения очистки первой.
Сравнить оба выбранных mean inner WP и все fold-результаты; два fold/seed на LR
не устанавливают статистическую значимость. Внешняя оценка, KD и сабмит требуют
отдельного продолжения; этот handoff их не запускает.

## Продолжение после прерывания

- **Профили до fit:** повторный `fit-cell` столкнётся с уже созданными
  неизменяемыми profile outputs. Проверить завершённые receipts и выполнить
  только оставшиеся действия напрямую. Например, после успешного `profile-one`
  остаются `profile-eight`, `profile-peak`, `admit`, `fit`. Шаблон команды каждого
  шага — `flock -n "$WNN_CAMPAIGN_ROOT/.launch.lock" uv run --no-sync python
  "$R5/run.py" profile-eight --campaign-root "$WNN_CAMPAIGN_ROOT" --cell
  R5_lr_1e-4_f0`. Для следующих шагов заменить action; не повторять уже
  завершённые immutable producers и не обходить сохранённую ошибку.
- **Fit:** не повторять `fit-cell` поверх существующего fit. В том же source/env
  выполнить `flock -n "$WNN_CAMPAIGN_ROOT/.launch.lock" uv run --no-sync python "$R5/run.py" resume --campaign-root
  "$WNN_CAMPAIGN_ROOT" --cell R5_lr_1e-4_f0` (заменить ячейку). Исходная
  failure/admission-политика сохраняется: ошибки сначала разобрать, не удалять
  receipts для обхода. Не выполнять параллельно с другой launch-командой.
- **Sweep до selection:** `finish-pair` использует `--resume` для существующих
  sweep outputs, заново проверяя завершённые эпохи. Частично записанный каталог
  эпохи блокирует resume: сохранить его, использовать новый `--output` через
  [прямые команды sweep](README.md#оценка-всех-эпох-и-выбор) и затем передать эти
  новые пути selector/planner. Экспортировать каждую восстановленную кривую с
  `export_midas.py --evaluation-output <new-complete-sweep-output>`: без этого
  параметра exporter читает старый канонический `epoch_evaluation`. Override
  допускается только для полной проверенной кривой именно этого fit.
  Launcher не подменяет пути молча.
- **Selection/export/plan:** outputs неизменяемы. Если `finish-pair` прервался
  после selection или экспорта, повторный вызов остановится до GPU работы.
  Сохранить созданное; вручную выполнить только недостающие шаги ниже. Для
  частичного bundle использовать новый output, не стирать предыдущий.
  Если готов только selection — экспортировать оба fold; если готов также F0
  bundle — экспортировать F2; после обоих bundle — создать plan. Если plan уже
  готов, перейти к его проверке и `cleanup-pair`, повторно plan не создавать.
- **Cleanup:** повторить `cleanup-pair` с тем же планом и тем же raw SHA.
  Прерванная очистка продолжается по durable intent; отсутствие файла допустимо
  только в рамках такого intent. Проверка `verify-pair` требует завершения обоих
  fit. До этого вторую пару не запускать.

Пример недостающих metadata-шагов после успешного selection (LR и пути заменить
только на фактически связанную пару; completed bundle повторно не экспортировать):

```bash
LR=1e-4
CELL=R5_lr_1e-4_f2
uv run --no-sync python "$R5/export_midas.py" \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --cell "$CELL" \
  --output "$WNN_CAMPAIGN_ROOT/midas-evidence/$CELL-pre-retention-retry1"
uv run --no-sync python "$R5/retention_plan.py" \
  --f0-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_${LR}_f0/epoch_evaluation" \
  --f2-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_${LR}_f2/epoch_evaluation" \
  --selection "$WNN_CAMPAIGN_ROOT/selection-lr-$LR.json" \
  --output "$WNN_CAMPAIGN_ROOT/retention-lr-$LR.json"
```

После первого retention intent fit считается архивным: обычные `fit`, `resume`
и новый MIDAS export запрещены. Полный sweep/повторный selector тоже больше не
поддерживаются: удалённые веса не доступны. Оригинальные индексы остаются
историей, а plan/receipts объясняют отсутствие файлов. `training_state.pt`
сохраняется для отдельной процедуры восстановления, но не делает обычный
resume корректным после удаления префикса. Свежие MIDAS наблюдения получают
из **предварительно созданного** bundle; его remote checkpoint references
описывают состояние до очистки, а не доступность удалённых весов сейчас.

Сохранённый epoch checkpoint допускает отдельный inner inference через
`predict_sequence.py --checkpoint ... --epoch-index ... --epoch-index-sha256 ...
--membership evaluation`: этот путь проверяет индекс и запрошенные веса,
не требуя остальных snapshots. Перед ним выполнить `verify-pair`. Полные sweep
и повторный selector требуют весь набор. Epoch weights пока не являются
автоматически допущенными KD teachers или сабмитами; отдельный downstream
протокол остаётся следующим шагом.

## Передаваемые файлы и границы проверки

[`launch.sh`](launch.sh) связывает существующие runner/sweep/export команды;
[`receiver.env.example`](receiver.env.example) перечисляет все параметры среды;
`retention_plan.py` создаёт план только после повторной проверки двух полных
кривых и selection; `retention_apply.py` применяет/проверяет завершение.
`retention_contract.py`, `retention_inventory.py`, `retention_verify.py` и
`retention_receipts.py` разделяют формат плана, перечень файлов, проверку
содержимого и durable receipts. `LAUNCH_PR.md` фиксирует границы этого изменения.
Receiver-набор включает проверки отказа при неполной кривой, изменении файлов,
прерывании очистки и попытке resume/export после intent.

Локально разрешены только чтение и статические проверки исходников. Тесты,
обучение, inference, profiles и операции удаления здесь не выполнялись.
Подготовленный код ещё должен пройти приёмку на принимающем сервере.
