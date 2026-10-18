---
kind: review
status: complete
hypothesis_ids: [H-RR17-01, H-RR17-19, H-055]
experiment_ids: [E-RR18-01-LABELS, E-F0-SUBMISSION-20261018-R1]
evidence_paths: [research/campaign_2026-10-18/priority25_execution_r4/receipts/RANK01_F0_S1001_RESULT.json, research/campaign_2026-10-18/f0_submission_r1/receipts/RELEASE.json]
---

# Что добавилось после обзора 17 октября

Первый проход: `research-team@1d57ed815c18eb278bcdd0207369b200970b28fb`; повторное чтение результатов: `e35220969ca6f74f42aec6296e82d1b16ca66efa`. Открытые PR проверены через GitHub API. Это чтение сохранённых результатов и исходников: локальные тесты, обучение и обращение к работающим GPU jobs не выполнялись. Более позднее состояние смотреть в [current-state](../../../specs/current-state.md).

## Новые результаты при повторном чтении Git

- [F1/S1001 завершён](../../campaign_2026-10-18/priority25_execution_r4/receipts/F1_COMPLETION_20261018T1500Z.md): wide **0.66132918806**, deep **0.66169478858**, delta wide-minus-deep **−0.00036560051**, conditional component 95% CI **[−0.00265684406,+0.00192301662]**. 812 167 scored rows, 457 групп, 149 компонент. F0 и F1 дают разные направления; это **2/30 пар**, а не устойчивый выигрыш ширины.
- [Восстановленный Main T3](../../campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r4/records/RESULTS.md) на тех же 808 863 F0 строках: **0.68115415663** против Q8 **0.67382635767**, delta **+0.00732779896**, conditional CI **[+0.00238446,+0.01352521]**. Это сильный historical recipe, не точная реплика `tl3.zip` и не изоляция эффекта KD/архитектуры. Потеря Q8 против своей FP32 смеси — лишь **0.00002458657**, поэтому эти данные не поддерживают quantization как основную причину локального отставания.
- [Записанный результат портала](../../campaign_2026-10-18/f0_submission_r1/receipts/PUBLIC_RESULT_0897egiv_20261018.json): для F0 ZIP пользователь команды сообщил **0.65839, 45m47s, 653 MB**. Это более позднее наблюдение заменяет прежнюю оценку времени для данного сабмита; браузером и SHA загруженного payload оно независимо не подтверждено. Сам этот агент ничего не загружал.
- [Trace прежнего лучшего `tl3.zip`](../../campaign_2026-10-18/quality_gap_r1/public_submission_trace_r1/README.md): в пользовательской таблице **0.66832**, ID `1lmehfi3`, 39m12s. Это смесь distilled и tied-lead GRU, а не Transformer. Production members обучены на train+valid; их нельзя объявлять чистыми F0 контролями. Восстановление их полной чистой цепочки ещё не завершено.
- В `current-state@e35220969` F3/S1001 допущен и имеет сохранённый прогресс без grade. Наши карточки и публикация не трогают worker ownership.

Практический вывод: сначала закрыть чистое происхождение сильного baseline и делать новые парные сравнения относительно точно объявленного контроля. Два наших loss/gradient варианта пока сохраняются как предложения; эти новые наблюдения не превращают их в приоритетные запуски.

## Первый проход: сохранённые результаты и ограничения

**Первое парное сравнение ширины и глубины завершено.** [F0/S1001 receipt](../../campaign_2026-10-18/priority25_execution_r4/receipts/RANK01_F0_S1001_RESULT.json): H512×1 = **0.67357575099 WP**, H384×2 = **0.67210876732**, delta **+0.00146698367**; фиксированный epoch5, по 89 520 успешных optimizer updates, ноль skips/nonfinite. Оценены 808 863 строки / 468 последовательностей / 141 компонент. Conditional component-bootstrap 95% CI **[−0.00041169457,+0.00358981964]**. Это одна из 30 пар, exposed F0, не доказательство преимущества по seeds/folds и не CP30.

**Подготовлен F0 ZIP, но CPU-дедлайн официального сервера не доказан.** [Release receipt](../../campaign_2026-10-18/f0_submission_r1/receipts/RELEASE.json): выбранная на exposed F0 смесь 75% wide / 25% deep, actual native Q8 WP **0.67382635767** против FP32 **0.67385094424**, размер **17 480 575 байт**. Это лучший из пяти локально просмотренных коэффициентов, а не независимое подтверждение. SHA256 `c4cb51bf20e7639f0fa607489b911b551f5296158db3bd913429e95ce7c139e6`. **53.9 минуты — условная оценка по историческому runtime**, не замер на сервере соревнования; в receipt отдельно есть 137-минутная абсолютная экстраполяция Beleriand и объяснение несопоставимой quota. Загрузка в соревнование не проводилась. Q8 относится к отдельной submission-работе; наши новые научные сравнения FP32.

**F1 ещё без grade в первом проверенном дереве 1d57ed815.** [Сохранённый heartbeat](../../campaign_2026-10-18/priority25_execution_r4/receipts/HEARTBEAT_20261018T1115Z.json) сообщает продолжение wide epoch4 в 11:17 UTC. Это snapshot producer, не наша live-проверка. F1 score здесь не приписываем. Новые гипотезы не меняют распределение этих задач.

## Пять новых draft PR

| PR | Вопрос | Evidence boundary |
|---|---|---|
| [#15](https://github.com/VladKochetov007/WunderFunder/pull/15), `e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e` | Причинная геометрия и дискретные признаки. | 60 признаков / предложенные B300 сравнения; отдельный небольшой geometry pilot не является результатом этого плана. |
| [#17](https://github.com/VladKochetov007/WunderFunder/pull/17), `588b93f9c2ad7e53aac7b1e2a39f30c32047ea22` | Raw-retaining Kalman local-trend учитель и замена одной семьи в student cache. | Fixed-GRU replay на 16 выбранных exposed-F0 группах: +0.000338 WP, CI пересекает ноль; это не trained-teacher/student gain. |
| [#18](https://github.com/VladKochetov007/WunderFunder/pull/18), `b5b0ea0e87318ad7feba149e56f328809a7e9057` | Предсказание следующей строки входов как auxiliary GRU. | Предложение с матрицей; результатов fit/grade нет. |
| [#20](https://github.com/VladKochetov007/WunderFunder/pull/20), `96a98a6963af14b88187f713e593cc1421a60859` | Input-only online adaptation с meta-training. | Trainer integration и B300-профиль ещё отсутствуют; нет измеренного student результата. |
| [#21](https://github.com/VladKochetov007/WunderFunder/pull/21), `28ff7499c7e1363a92f7599c6c68d58de966d8c2` | Group-masked предобучение с пересчётом event-признаков после маски. | Три matched arms предложены, B300 profile/fit/grade отсутствуют. |

До нашей публикации API показывал 19 открытых draft PR к research-team. Часть source notes находится только на experiment-ветках: наличие PR не означает merge в research-team или main. Новые общие изменения также включают FP32 trainer enforcement и модульные seams; это инженерные изменения, не качество модели.

## Следующее исследование

Предлагаем [два узких продолжения градиентных гипотез](../ideas/GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md) на основании [первичных статей](../sources/GRADIENT_ESTIMATOR_LITERATURE_20261018.md), сохранив историю отрицательных результатов. Их приоритет относительно ranked25 не меняли; публикация документов не запускает обучение.

## Независимая проверка наших новых карточек

Независимый source-only review выявил три расхождения: epsilon-ветви официального scorer, 95% против 97.5% интервалов между packet/card, и неопределённые дополнительные веса WMSE. Координатор исправил документы: нулевые ветви W<1e-8 / std≤1e-8, единая two-contrast family, одинаковые z/w/masks без propensity/class/age/auxiliary weights. Сохраняется ограничение: WP replay сравнивается при своём macrostep, не доказывает преимущество перед обычным trainer cadence. Ни reviewer, ни координатор не запускали модель, тесты или обучение; исправления проверены чтением текста и JSON.
