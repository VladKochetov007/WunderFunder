---
kind: review
status: evidence_synthesis
created_utc: 2026-10-18
experiment_ids: []
---

# Где сейчас теряется качество

Это разбор существующих результатов после пользовательского сообщения о public
0.65839 для `0897egiv`. Новых fits, replay или grade этот разбор не добавляет.
Здоровый F1 продолжает исходное сравнение; его замороженный план не менялся.

## Сначала разделить два разрыва

| Сравнение | Разница WP | Что установлено |
|---|---:|---|
| Лидер 0.67243 − kotiki 0.66832 | +0.00411 | Датированный снимок 17 октября, не live-рейтинг |
| Новый пакет 0.65839 − прежний team best 0.66832 | −0.00993 | Новый пользовательский public-результат против датированного best |
| Новый public 0.65839 − native F0 0.67382636 | −0.01543636 | Разные наборы данных; это не оценка единственного бага |

[Снимок лидерборда](audit-2026-10-17/official_leaderboard_20261017.json),
[новый результат](../../campaign_2026-10-18/f0_submission_r1/receipts/PUBLIC_RESULT_0897egiv_20261018.json).
Имя ZIP совпало с релизом, но SHA загруженных на портал байтов не подтверждён.
Пользователь сообщил 45m47s / 653MB. Код или рецептуры лидеров не установлены;
их конкретную архитектуру, датасеты или приёмы из таблицы scores вывести нельзя.

## Самые существенные подтверждённые пробелы

**1. Текущий архитектурный контроль не воспроизводит наш сильный полный рецепт.**
Rank1 сравнивает cold-start wide/deep, исходные supervised labels, без teacher,
initializer и propensity. Это корректный узкий вопрос о форме сети. Лучшие
записанные production-пакеты используют distillation, тёплых родителей, другие
комбинации признаков и ансамбли разных семейств/seed; финальные fits включают
train+valid. Различаются также объём fitting data и история обучения. Поэтому
победа wide над deep сама по себе не означает победу над team best.
[План](../../campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json),
[каталог и lineage](../models/CURRENT_MODEL_CATALOG.md).

**2. Текущий T3-target помог в конкретном warm-parent continuation.**
В matched warm-parent seed3 continuation на шести эпохах alpha0.9 получил
0.68115536, alpha0 — 0.66684254: разница +0.01431282. На fitting-valid alpha0
при этом выигрывал +0.012247, а на исключённых F0-группах проигрывал −0.014313.
Улучшение fitting-score здесь сопровождалось ухудшением evaluation-score.
Это сильное основание разбирать target/regularization/ancestry, но не обещание
+0.0143 для нового cold-start рецепта или public. Оба arms унаследовали одного
уже дистиллированного родителя; один exposed fold/seed, FP16 skips 23 против19,
не CP30. Сдвиг распределения и переобучение по отдельности не изолированы.
[Matched grade и follow-up fit/eval](ALPHA0_WARMPARENT_CP3_CP5_GRADE_REVIEW_20261015_R1.md).

**3. Локальная система выбора пока недостаточно предсказывает перенос.**
В старом F1-сравнении четыре local WP 0.66938 / 0.67103 / 0.67195 / 0.67329
соответствовали public 0.6639 / 0.6652 / 0.6616 / 0.6650. Это четыре зависимых
endpoint, включая смесь: не доказательство бесполезности CV, но уже реальное
расхождение ранжирования. Новый коэффициент75/25 выбран из пяти на том же
многократно исследованном F0. Локальные +0.000275 FP32 смеси над FP32 wide не закрывают
вопрос переноса. Один seed и фиксированные шесть эпох не устанавливают
сходимость. Повторы архитектурной пары не заменят matched comparison с сильной
target/initializer рецептурой.
[Исторический transfer receipt](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/PUBLIC_FOLD_AFFINITY_DIAGNOSTIC_R1.json),
[F0 selection](../../campaign_2026-10-18/f0_submission_r1/README.md).

**4. Не замкнута диагностика ошибок именно актуального сильного решения.**
Есть extensive EDA по targets, event/quiet, age, within/between-sequence
covariance и корреляции старых прогнозов. Однако у части этих caches не
подтверждены producer/checkpoint/order. Для tl3/tl4 supplied-valid scores
in-sample: модели обучались на train+valid. В изученных актуальных пакетах нет единой source-qualified
таблицы честного локального аналога incumbent против нового кандидата по
target, low-variance/tails, позиции, weighted clipping/calibration и вкладам
в pooled score. Значит, пока нельзя честно назвать конкретный режим или
target причиной public-просадки. Календарные даты и настоящие instrument IDs
из имеющихся inferred classes также не следуют.
[Предыдущая EDA](../../campaign_2026-10-17/research_reset_r2/prediction_audit/REPORT.md),
[ещё не выполненная weighted calibration диагностика](../../campaign_2026-10-17/research_reset_r1/reviews/WEIGHTED_TARGET_GEOMETRY_R3.md),
[validation scopes](../validation/VALIDATION_PROTOCOLS.md).

## Что сейчас не подтверждено как объяснение

- Q8 потерял всего0.0000245866 на полном F0, independent rescore совпал.
  Actual local ZIP прошёл468 последовательностей, reset/mask/metric checks.
  Конкретного дефекта на проверенном пути не найдено. Hidden inputs, точные
  portal bytes и официальный CPU/container этим не сертифицированы.
  [Независимый static audit](../../campaign_2026-10-18/f0_submission_r1/receipts/PUBLIC_TRANSFER_STATIC_AUDIT_20261018.json).
- Редкие clock exceptions занимают0.003172% scored target weight. Это не
  строгая граница влияния на WP, но нет свидетельства, что они объясняют большой
  разрыв. Простые output hold/EMA уже отрицательны; histogram/CE и pooled-loss
  варианты также нельзя выдавать за совсем непроверенные идеи.
  [R2 decisions](../../campaign_2026-10-17/research_reset_r2/reflection/ROUND3_EDA_DECISIONS.md),
  [история objectives](../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md).
- Обнаруженные старые preprocessing/partial-cache/NaN bugs исправлены в своих
  ветках. Четырёхгрупповой огромный эффект исправления zero-i1 не является
  улучшением public и не объясняет автоматически новый пакет.
  [Bugfix evidence](../../campaign_2026-10-17/research_reset_r1/bugfix/README.md).

## Следующее различающее исследование

1. Связать точный public incumbent с ZIP/recipe и найти его честный локальный
   аналог с исключёнными evaluation-компонентами, включая teacher/parent ancestry.
   Не сравнивать fitted valid с held-out F0. Если такого аналога нет, явно
   записать необходимость отдельного matched control вместо удобного числа.
2. На доступных source-bound прогнозах составить одну baseline-relative таблицу:
   оба target, weight mass и pooled-score contribution, режим/позиция,
   calibration и clipping. Старую EDA использовать как контекст, не как
   producer-bound доказательство о новом incumbent.
3. Сопоставить имеющиеся teacher/parent/student результаты и trajectories.
   Следующая отдельная заявленная проверка должна различать target/initializer
   benefit от capacity; сначала пригодность clean ancestry и доступность exact
   controls, затем bounded execution по действующей процедуре. Нынешний F1
   закончить без изменения frozen comparison.
4. Проверять устойчивость выбранного изменения на других folds и fitting seeds.
   Не выбирать «public-похожий» fold, не подгонять веса по единственному public
   ответу и не выдавать повторную разработку на exposed folds за новый holdout.

Организационный вывод: контроль воспроизводимости нужен, но число receipts и
точность wide/deep comparison не отвечают сами по себе на вопрос отставания
от сильного полного решения. Новые результаты должны добавляться к одной
таблице «сильный контроль → конкретное изменение → где выигрыш/потеря», с
явной границей между установленным эффектом и ещё не проверенной причиной.

Независимое factual review агента local_history не нашло существенных ошибок;
его уточнения о механизме T3, FP32 delta и границах изученного корпуса внесены.
Это review интерпретации существующих источников, не независимая ML репликация.
