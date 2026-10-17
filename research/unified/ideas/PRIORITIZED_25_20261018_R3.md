---
kind: idea
status: proposed
created_utc: 2026-10-17T21:07:13.765241+00:00
updated_utc: 2026-10-17T21:07:13.765241+00:00
hypothesis_ids: [H-RR17-01, H-RR17-06, H-RR17-02, H-RR17-04, H-RR17-10, H-RR17-22, H-RR17-05, H-RR17-16, H-RR17-11, H-RR17-14, H-RR17-15, H-RR17-20, H-RR17-08, H-RR17-03, H-RR17-21, H-RR17-12, H-RR17-09, H-RR17-17, H-RR17-18, H-RR17-07, H-RR17-19, H-RR17-13, H-RR17-23, H-RR17-24, H-RR17-25]
experiment_ids: []
evidence_paths: [research/campaign_2026-10-17/research_reset_r2, research/campaign_2026-10-17/moment_geometry_r1]
---

# Приоритеты 25 исследовательских идей на 18 октября 2026

Первые четыре направления: широкий компактный GRU, причинные статистики с сохранением исходного уровня, сильный teacher и передача его промежуточного состояния. Рейтинг учитывает потенциал конечной модели, местные результаты, стоимость проверки и ценность информации. Близкие позиции — суждение об очередности, а не измеренная вероятность успеха.

Исходные H-RR17-ID сохранены. Этот список уточняет порядок после R2, не заменяет факты и не создаёт 25 новых экспериментов. Обучение, inference и подбор коэффициентов в этой задаче не запускались. Численные приоры ожидаемого прироста остаются неизвестными.

## Что изменило приоритеты

- Редкие clock-исключения занимают 0,003172% веса оцениваемых valid-меток и слабо мотивируют отдельную архитектуру.
- Отрицательные output-hold/EMA результаты не проверяют изменение hidden state; oracle +0,03 использует будущее.
- Корреляция внутриучастковых колебаний около 0,96 относится к трём связанным массивам с неполной producer-привязкой. Она мотивирует поиск разных механизмов, но не доказывает бесполезность любого ансамбля.
- Tiny geometry-пилоты дают соседнее основание для moment/scale и dictionary/dwell. Они не подтверждают strict normalization, equivariance или выигрыш полного student.
- Attention, Mamba и xLSTM подняты относительно clock-ориентированного списка R2. Наше EDA не было отрицательным экспериментом над этими архитектурами.

## Упорядоченный портфель

| № | Исходный ID | Идея и основание приоритета | Решающее сравнение | Будущая стоимость |
|---:|---|---|---|---|
| 1 | H-RR17-01 | **Широкий однослойный GRU до сходимости.** Самая прямая ставка на конечную модель; старый H512×1 screen лучше H256×2, но это один seed и короткий бюджет. | Cold wide/shallow против deep/narrow с согласованными данными, бюджетами и checkpoint rule; измерить качество и CPU latency. | Средняя |
| 2 | H-RR17-06 | **Причинные статистики и нормализация с raw bypass.** Есть соседние moment/scale positives; абсолютный уровень нельзя без проверки стирать. | Raw; raw + prefix statistics; те же statistics + normalized path + raw bypass. Для строки t нормализующие статистики через t−1. | Низкая–средняя |
| 3 | H-RR17-02 | **Сильный causal teacher → компактный student.** KD уже имеет локальные основания; важен выигрыш student, а не teacher-only рекорд. | Один teacher и одинаковый контекст: labels-only против output KD. Разнородный cohort — следующий отдельный контроль. | Высокая |
| 4 | H-RR17-04 | **Промежуточные hints учителя.** Два выхода могут терять полезное состояние, но произвольные hidden coordinates могут мешать. | При одном teacher/student: output KD против output + projected hidden hints. Semantic queries — отдельный вариант после prior audit. | Средняя поверх KD |
| 5 | H-RR17-10 | **Разная ёмкость общей и контрастной компонент.** Реальное нелинейное разделение задач, а не эквивалентная замена координат. | Independent heads против c+d / c−d с согласованной ёмкостью; официально оценивать восстановленные два выхода. | Средняя |
| 6 | H-RR17-22 | **Общие преобразования компонентов с сохранением ID.** Альтернатива плоскому encoder; симметрия таргета пока не установлена. | Shared encoder внутри допустимых групп + ID/escape против обычного encoder. Не считать индекс физической глубиной. | Средняя |
| 7 | H-RR17-05 | **Надёжность i1 и межинструментные лаги.** Исправление preprocessing позволяет пересмотреть затронутые старые результаты; само исправление не доказывает пользу i1. | No-i1, обычный i1 и один fixed lag basis; input-only gate проверять отдельным следующим шагом. | Средняя |
| 8 | H-RR17-16 | **Локальное causal attention + recurrence.** Другая форма чтения недавних событий при сохранении длинного состояния. Clock-EDA её не проверяло. | Один фиксированный left-causal window против того же backbone без attention; chunk/step parity и конечный CPU student. | Средняя–высокая |
| 9 | H-RR17-11 | **Ковариационная ориентация групп признаков.** Может сохранить информацию о совместном движении, которую теряет scalar activity. | Trace/activity-only против trace + orientation одной и той же малой потоковой ковариации. | Низкая–средняя |
| 10 | H-RR17-14 | **Mamba-3 teacher, MIMO против SISO.** Отдельная ставка на оператор памяти; языковые результаты не являются рыночными. | Сначала stepwise/reset/causal parity; затем согласованные ресурсы MIMO/SISO и один и тот же student. | Высокая |
| 11 | H-RR17-15 | **xLSTM teacher с матричной памятью.** Возможность хранить ассоциации иначе, чем GRU; наша EDA преимущества не доказала. | Scalar/matrix memory при контроле ёмкости, streaming parity и одинаковом student-transfer. | Высокая |
| 12 | H-RR17-20 | **Нелинейная калибровка под weighted-метрику.** Дешёвая проверка clipping/нелинейности; обычный positive affine transform до clipping Pearson не улучшает. | Предзаданные weighted bins; будущую карту fitting-only проверять на отдельных компонентах целиком. | Низкая |
| 13 | H-RR17-08 | **Новые observable future queries.** Обычные next-mark/waiting-time уже были в Belief; нужен конкретный дополнительный query bank. | Точная разница с Belief/aux_quote/future moments, censoring, student с одним новым auxiliary и без него. | Средняя |
| 14 | H-RR17-03 | **Мягкая запись в основное hidden state.** Выход остаётся плотным; output-hold не тестировал этот механизм. Generic DualClock — уже повтор. | Сначала source-bound replay для разведения новых наблюдений и recurrent settling; затем soft-write против той же dense recurrence. | Средняя |
| 15 | H-RR17-21 | **Ограниченная group-robust поправка objective.** Может уменьшить систематические провалы; переразвесовка способна ухудшить pooled WP. | Группы и их массу определить на fitting-данных; один bounded objective против исходного. Группы не входят в serving. | Низкая–средняя |
| 16 | H-RR17-12 | **CUSUM для скорости адаптации медленной памяти.** Только observable innovations, без истинных классов. Age/class EDA не установило нужного порога. | CUSUM против event-age/EW activity на fixed feature-shift controls; затем soft gate без hard reset. | Низкая–средняя |
| 17 | H-RR17-09 | **Признак порядка межпоточных событий.** Low-order path signature близка лаговому базису и должна превзойти именно его. | Одна antisymmetric area против fixed cross-lags и count/sum-preserving shuffles. | Низкая |
| 18 | H-RR17-17 | **Восстановление скрытой группы текущих признаков.** Предобучение encoder на межгрупповых связях; риск простого копирования. | Маска до всех delta/normalization/skip путей; один student с reconstruction-pretraining и без него. | Средняя |
| 19 | H-RR17-18 | **CPC будущих входных состояний.** Может выучить динамику, но часто решается через persistence, activity или sequence identity. | Same-sequence matched negatives, persistence control и тот же supervised-only student; auxiliary loss не заменяет WP. | Средняя–высокая |
| 20 | H-RR17-07 | **KD распределения и зависимости.** Небольшой набор квантилей/связей может дополнять два средних; histogram/CE negatives снижают приоритет. | Один teacher/student: output-only KD против output + fixed distribution summary; прежние два serving-выхода. | Средняя |
| 21 | H-RR17-19 | **Компенсированное усечение градиента ARTBP.** Другой способ обучать длинную память, но компенсация может резко увеличить дисперсию. | После разрешения на градиентные проверки — toy exact BPTT; затем ARTBP против fixed truncation с тем же forward. Сейчас не выполняется. | Высокая при обучении |
| 22 | H-RR17-13 | **Поиск прошлых состояний по содержимому.** Снижают приоритет нестабильные старые retrieval-результаты и стоимость lookup. | Bounded current-sequence cache против recency/fixed taps/shuffled keys; без меток и межпоследовательной памяти. | Средняя–высокая |
| 23 | H-RR17-23 | **Топология переходов и циклов входных кодов.** Может отличать одинаковую occupancy; dictionary/dwell pilot подтверждает только необходимость простого контроля. | Fixed training-only discretization; циклы против occupancy/first-order transitions и shuffle порядка. | Средняя |
| 24 | H-RR17-24 | **Адаптирующаяся fast memory только у teacher.** Смелая ставка; student может не суметь воспроизвести адаптацию. | Input-only self-supervised teacher с sequence reset; конечный fixed student против KD от обычного teacher. | Очень высокая |
| 25 | H-RR17-25 | **World-model входов и внутренние rollout учителя.** Длинная цепочка ошибок; воображаемое будущее не добавляет наблюдений. | Multi-step dynamics сверх persistence/one-step, затем отдельный student-transfer. Никаких реальных будущих входов в serving. | Очень высокая |

## Практическая очередность

Основная линия — позиции 1–4 с отдельными контролями. Attention — первая независимая архитектурная ветка; Mamba и xLSTM сохраняют место для исследования вне GRU-линии. Калибровка может пройти раньше по календарю как дешёвая проверка, хотя занимает 12-е место по совокупной ценности.

Это не очередь из 25 разрешённых запусков. В research-reset scope обучение остаётся запрещено. Перед state-replay требуется точная связь checkpoint/preprocessing/предсказаний. Полные сравнения качества после разрешения следуют [CP30](../../campaign_2026-10-17/research_reset_r1/PROTOCOL.json): 30 парных fold×seed ячеек на arm, strict FP32, проверенная ancestry и неизменная метрика. Короткий pilot — отдельный уровень доказательности; имеющиеся folds остаются exposed.

## Критика и принятые уточнения

`priority25_critic_r3`, gpt-6-luna/max, провёл read-only критику. Черновой порядок root пришёл до фиксации независимого top8, поэтому это informed critique, не blind review. Принято предложение опустить H-RR17-08 из top8 из-за пересечения с Belief и поднять H-RR17-16. Калибровка оставлена дешёвой параллельной проверкой. Уточнены границы geometry → normalization/equivariance и preprocessing repair → i1 benefit. Согласие агентов не является подтверждением качества.

## Источники

- [Исходные 25 карточек](../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md) — механизмы, пересечения, контроли и литература.
- [R2 решения и prior-code attack](../../campaign_2026-10-17/research_reset_r2/reflection/ROUND3_EDA_DECISIONS.md) — Belief/DualClock, fixed-output EDA и границы выводов.
- [Ширина и глубина student](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md) — короткий exposed-F0 screen: H512×1 0,67323776 против H256×2 0,66977764; не доказанное converged превосходство.
- [Геометрия и дискретные признаки](../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) — отдельные 33 tiny-endpoint результата, fold/seed heterogeneity и fixed-mixture проверки, без production/KD-выигрыша.
- [Текущий статус](../../../specs/current-state.md) — отдельные границы разрешений.
- [FitNets](https://arxiv.org/abs/1412.6550), [Mamba-3](https://arxiv.org/abs/2603.15569), [xLSTM](https://arxiv.org/abs/2405.04517) — первичные abstracts повторно проверены в этой задаче для intermediate hints, MIMO и scalar/matrix memory. Их внешние результаты не являются свидетельством местного выигрыша.
