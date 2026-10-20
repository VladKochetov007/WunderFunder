---
kind: idea
status: proposed
hypothesis_ids: [H-CAUSAL-TRANSFORMER-20261017, H-CONTEXT-TRANSFORMER-20261017, H-SELECTIVE-SSM-20261017, H-DRAFT-CAUSAL-SMOOTHING-20261017]
experiment_ids: [E-ARCHITECTURE-CP30-HANDOFF-20261018]
evidence_paths: []
---

# Наши опубликованные пакеты для будущих GPU-экспериментов

Код сохранён в remote без перезаписи чужих веток. Это исходники для дальнейшего переноса и проверки, не готовые допущенные запуски. Здесь не выполнялись тесты, обучение, model imports или профили.

| Пакет | Опубликованный source | Содержание и ограничения |
|---|---|---|
| Четыре архитектуры | [codex/transformer-mamba-gpu-wave](https://github.com/VladKochetov007/WunderFunder/tree/codex/transformer-mamba-gpu-wave), `bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb` | GRU H256×2; causal Transformer D256×4 с окнами 256 и 2048; Mamba-2 D256×4. Реализованные исходники и host-side packet, target GPU не проверен. |
| Причинное сглаживание | [codex/causal-smoothing-executable](https://github.com/VladKochetov007/WunderFunder/tree/codex/causal-smoothing-executable), `ebc3b9244d07f107f870d5ca1154f4ecc98dbdb0` | Equal-width raw+zero против raw+EWMA/residual; полный per-cell fit → predict → score путь, 30 пар / 60 fits, 12 эпох. Доработка к [PR #9](https://github.com/VladKochetov007/WunderFunder/pull/9); опубликованный код пока находится в отдельной ветке, а не в diff PR #9. |

[Draft PR #24](https://github.com/VladKochetov007/WunderFunder/pull/24) передаёт эту четвёрку в research-team как отдельный future-CP30 handoff; исходный код сохранён в указанном remote snapshot, а его перенос в текущий trainer ещё не сделан.

## Дистилляция сильных учителей и streaming-студенты

| Draft | Проверяемый вопрос | Опубликованный snapshot |
|---|---|---|
| [PR #28](https://github.com/VladKochetov007/WunderFunder/pull/28) | L207/XL682 × GRU H256/H512: перенос качества учителя | [packet](https://github.com/VladKochetov007/WunderFunder/tree/6bbc99ad761a28659930f65931acfb452226a690/research/experiment_drafts/teacher-transfer-frontier), `6bbc99ad761a` |
| [PR #29](https://github.com/VladKochetov007/WunderFunder/pull/29) | Скрытые представления учителя: правильный hint против переставленного | [packet](https://github.com/VladKochetov007/WunderFunder/tree/98f707ebeb702dc5f282203f91429d1d5223c1fa/research/experiment_drafts/teacher-representation-hints), `98f707ebeb70` |
| [PR #30](https://github.com/VladKochetov007/WunderFunder/pull/30) | Amendment: четыре compact T128 streaming label/KD fits с повторным использованием PR28 | [current packet](https://github.com/VladKochetov007/WunderFunder/tree/experiment/streaming-student-distillation/research/experiment_drafts/streaming-student-distillation); [historical CP30 snapshot](https://github.com/VladKochetov007/WunderFunder/tree/3148fb621bdf7cd3e50110f57baf03882bafd2c4/research/experiment_drafts/streaming-student-distillation), `3148fb621bdf` |

Все три PR направлены в `research-team`. В PR30 исходный CP30 остаётся историческим; текущая prospective amendment добавляет четыре новых student fits с точным переиспользованием двух teachers/caches и четырёх GRU endpoints из PR28. Общий runtime реализован в draft [PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32), но B300-приёмка, профили, новые fits и оценки не выполнены. Остальные первичные матрицы также остаются подготовленными исследованиями; запусков по этим handoff нет и `launch_ready=false`. [Общее обоснование и первоисточники](CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md#22-select-the-best-transferable-teacher-not-the-largest-checkpoint), [CPU-инференс](../inference/TRANSFORMER_DISTILLATION_CPU_PATH.md), [свежие командные результаты и происхождение учителей](../reviews/TEAM_UPDATES_AND_TEACHER_LINEAGE_20261019.md).

## Архитектурная четвёрка

[Замороженный README и recipe](https://github.com/VladKochetov007/WunderFunder/tree/bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb/research/campaign_2026-10-17/architecture_gpu_wave_r1) содержат исторический F0/seed3/one-epoch screen. Его не переписывали под вид завершённого CP30. Он привязан к legacy trainer: модели ещё нужно подключить через текущие `wnn.training.models`, input/state/forward/checkpoint/prediction seams. Простое слияние всей старой ветки может конфликтовать с новой архитектурой trainer и общими индексами.

- [Transformer](https://github.com/VladKochetov007/WunderFunder/blob/bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb/src/wnn/causal_transformer.py) и [Mamba-2](https://github.com/VladKochetov007/WunderFunder/blob/bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb/src/wnn/mamba2_regressor.py) — переносимые исходники, не подтверждённые на GPU реализации. Mamba-2 не тождественна Mamba-3 из ranked25.
- Legacy recipe задаёт FP32, но TF32-off на пути этой волны не доказан. Future adapter должен явно закрыть autocast/scaler/TF32 и проверить фактические dtype. CUDA/Torch/Mamba build, chunk/step parity, reset/causality, состояние backward и peak memory остаются непроверенными.
- Для научного сравнения нужен новый фиксированный full-data CP30 recipe: 30 парных cells **на каждую из четырёх моделей**, всего 120 fits. Основные контрасты B−A, C−B, D−A; tuning effort и compute сравниваются явно. Это не параметрически равные модели. Исторический single-cell screen не входит в CP30.
- Старый queue runner остаётся частью замороженного source; новый queue manager не вводим. На принимающей стороне переиспользуем принятый запуск отдельных cells.
- GPU-учитель может быть больше serving лимитов. Для финала нужен отдельный matched GRU-student KD или CPU backend и замер полного ZIP на целевом CPU. Старый Docker budget checker — непроверенный инструмент, а не свидетельство прохождения 1 CPU / 16 GB / 60 min / 20 MB.

## Причинное сглаживание

[README полной реализации](https://github.com/VladKochetov007/WunderFunder/blob/ebc3b9244d07f107f870d5ca1154f4ecc98dbdb0/research/experiment_drafts/causal-smoothing/README.md) сохраняет точные команды и ограничения. Remote PR #9 на момент проверки: `a972168f5af24060fc053f569abcf5433b251de7`. Полный snapshot `ebc3b92` не является его descendant; force-push не выполнялся.

Для интеграции нужен successor от актуального research-team с переносом только feature/runner/reporting seams и разбором пересечений в `scripts/predict_run.py`, `src/wnn/training/runner.py`, ledger и README. Нельзя потерять новые team safety fixes, новые записи или заменить current-state старым snapshot. Target-host checks и ресурсный профиль обоих arms обязательны до quality fits; существующие authored tests локально не запускались.

## Источники и решение

Remote tips повторно сверены через `git ls-remote`; независимый reviewer прочитал обе source branches и их отличия от актуальной базы. Ни публикация, ни source review не доказывают numerical correctness или gain. Следующий шаг — ограниченный перенос модулей и фиксация CP30 bindings на принимающей стороне; разрешения на GPU-запуск этой задачей не добавляются. Новые [градиентные гипотезы](GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md) имеют отдельные draft PR и не подмешиваются к этой четвёрке.
