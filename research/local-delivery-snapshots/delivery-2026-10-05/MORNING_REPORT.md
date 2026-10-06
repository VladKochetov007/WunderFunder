# Утренний отчёт 5 октября (черновик, дополняется ночью)

## Итог в одном абзаце
(заполняется в конце ночи)

## Результаты по 18 пунктам
| # | ветка | результат | вердикт |
|---|---|---|---|
| 1 | WiSE α 0.5 / 1.5 (смесь весов родителя и полного FT) | α .5: valid −.0003/+.0003, train-HO +.0023/+.0020; α 1.5: −.0012 / −.0039 | α 0.5 — кандидат |
| 2 | FT только вход+голова (lr 1e-4) | valid −.0002/+.0003, train-HO +.0008/+.0008 | слабый плюс |
| 3 | FT с полом маски λ .02 | valid −.0004/+.0002, train-HO +.0008/+.0007 | слабый плюс |
| 4 | since-weight в FT | valid 0/+.0003, train-HO 0/+.0003 | ноль |
| 5 | ffA lr 2e-4 / ffB 4 эп | ffA: valid −.0002, train-HO −.0016 | ffA не слать |
| 6 | сглаживание/заморозка выхода внутри сегмента | заморозка −0.28 на c3, любое сглаживание < 0 | закрыто |
| 7 | потолок тихого класса 3 по декодированному состоянию | lookup по ценовому состоянию: WP .017 на тихом (GRU .413) | закрыто: байес-движок по цепи бессмыслен |
| 8 | rank по сплитам | маргиналы train/valid различаются → преобразование общее | закрыто |
| 9 | int8 vs fp32 | −0.00005 WP | закрыто |
| 10 | зеркало bid↔ask | точной симметрии графа нет, но граф неполный | не решено, отложено |
| 11 | аудит псевдометок i1 | a-признаки не виноваты (.181 vs .174) | закрыто |
| 12 | бейзлайн организаторов в смеси | composite: valid +.0005/+.0010, train-HO +.0026/+.0026 (w .25); на OOF valid ≈0 | кандидат (с риском) |
| 13–17 | GPU-эксперименты | идут | — |

## Кандидаты на LB (проверены: паритет, детерминизм, время)
(финализируется утром)

## Кандидаты на LB 5 октября (порядок отправки)
Время на скорере ≈ 0.49 × локальной проекции (калибровка по посылке Влада: 75 мин локально → 37 мин public).
| # | файл | состав | локально / оценка public | основание |
|---|---|---|---|---|
| 1 | blend_rc.zip | d256 combo s4 0.6 + rawcorr s3/s4 0.15 + бейзлайн 0.0625 | 83 / ~41 мин | база 0.6660 + разнообразие на сырых признаках |
| 2 | blend_rw.zip | d256 combo s4 0.5 + raw-wMSE s3/s4 0.25 + бейзлайн 0.0625 | 80 / ~39 мин | raw-wMSE сильнее rawcorr на train-HO (+.0011 vs +.0005 в смеси) |
| 3 | pair_fullft_bw.zip | наша пара fullft 0.5 + raw-wMSE 0.25 + бейзлайн | 62 / ~30 мин | composite: valid +.0007/+.0018, train-HO +.0033/+.0035 к pair_fullft (0.6649) |
| 4 | blend_pair.zip | d256 combo 0.5 + наша пара fullft 0.5 | 61 / ~30 мин | два разных рецепта |
| 5 | pair_boot_fullft.zip | d256 Muon+boot8 пара + полный FT | 58 / ~29 мин | наш вариант рецепта Влада |
Запасные: pair_fullft_br, pair_wise05_br, pair_wise05.
Не слать: pair_ffA_lr2e4, pair_ffB_4ep (минус на composite), blend_pair_rc (~51 мин, риск таймаута).

## LB 5 октября
| время | посылка | public | время на скорере |
|---|---|---|---|
| 03:11 | blend_rw.zip | **0.6664** (лучший) | 32m02s (локально 79.7 → коэффициент 0.40) |
| 03:56 | blend_pair_rw.zip | **0.6666** (лучший) | 37m51s (локально 93.8 → 0.40) |

## Продолжение 5 октября: пересмотр отказов и 15 идей

Карточки 15 экспериментов, приоритеты, первые проверки и грубые бюджеты: `revival/ideas.json`. Ни одна новая идея не объявлена выигрышем public. Текущий балл лидера live не получен.

На beleriand собраны sufficient statistics фиксированных смесей; DataSphere job `bt1je9p2d21o87q81dik` успешно выполнил 2000 парных bootstrap-пересэмплирований по цепочкам. Результаты: `revival/report.json`, входы и коэффициенты: `revival/inputs.json`. Это локальный composite-аналог смеси, не точный ансамбль `blend_pair_rw`.

Выявлено пересечение цепочек в прежнем composite train-holdout: из 2000 последовательностей только 345 не имеют членов той же цепочки в известных обучающих данных composite-родителя; на valid остаётся 436/467. Это соседство, не доказательство прямого использования отложенных меток. Фильтрация не отменяет адаптивного использования валидации и не сертифицирует неизвестную историю обучения baseline организаторов.

| Замена raw-wMSE компонента при фиксированной доле 0.2 | Δ valid без пересечения цепочек | Δ train-HO без пересечения цепочек |
|---|---:|---:|
| two-stream, seed 3 | +0.000685 | +0.000931 |
| LSTM, среднее seed 3/4 | +0.000454 | +0.000510 |
| dual-clock, seed 3 | +0.000459 | +0.000021 |

У two-stream условные 95% интервалы: valid [+0.000339, +0.001079], train-HO [+0.000436, +0.001430]. Интервалы относятся только к фиксированным моделям; не включают вариативность обучения, подбор идей или сдвиг public. LSTM-пара и одиночный raw-reference имеют разное число моделей. Проверки двух seed и CPU-стоимости ещё нужны.

Запущены на beleriand: online normalization seed 3/4 (GPU 2/3), two-stream seed 4 (GPU 6), matched raw-wMSE seed 4 (GPU 0). Online-norm ранее падал из-за лимита 3.9 ГБ при свободной карте; теперь лимит 0.65 GPU. Первый перезапуск остановился при завершении user systemd после logout: включён linger собственного пользователя, повторные запуски имеют новые имена `onorm65r2`.

Supervisor `wunder-revival-finish` ждёт четыре задания, проверяет наличие предсказаний и выполняет тот же chain-bootstrap с двухseedовым raw-контролем. Выход на beleriand: `~/WunderFunder-creative/research/iterations/2026-10-05-revival/followup/report.json`; журнал `finish.log`. Для заданий установлен лимит 2 часа. На момент записи обучение продолжается, новой посылки на leaderboard нет.

## Завершённая серия wave2 — 5 октября, 12:27 UTC

Все задания этой серии на beleriand завершены успешно. DataSphere job `bt16bgpsdmqg48gs6hd7` подтвердил native-инференс в Python 3.11 / Bookworm. Первоначальный стандартный образ DataSphere оказался несовместим с GLIBC 2.36 у уже существующих бинарников; конфигурация образа исправлена, бинарники не изменены.

Two-stream реплицировался на seed 3/4 против raw-пары: Δ valid-clean +0.000679, Δ прежний train-clean +0.000606. На новом отложенном train-наборе (282 последовательности, 191 цепочка) Δ +0.000670. Это сравнение локальных аналогов смесей на адаптивно использованных данных, не измерение public и не точная копия полного public-ансамбля.

| Новый эксперимент, среднее предсказаний двух seed | Δ valid-clean (436 seq) | Δ новый train-HO (282 seq) | Решение |
|---|---:|---:|---|
| Sentinel indicator+age против контроля с теми же 6 дополнительными нулевыми входами | −0.000038 | +0.000204 | Нет устойчивого преимущества; не включать |
| Residual-FT против обычного FT, оба внутри фиксированной смеси | −0.000114 | +0.000634 | Смешанный результат; не включать |
| Residual-FT смесь против исходной смеси | +0.000089 | +0.001149 | Гипотеза остаётся исследовательской; преимущество над FT на valid не показано |

Для residual-FT выделены 598 последовательностей / 389 цепочек, исключённых из известных обучающих цепочек учителей и из valid fold0. Новые 282 последовательности оставлены для проверки; ни одна их цепочка не вошла в residual-fit. История обучения baseline организаторов независимо не сертифицирована. Bootstrap-интервалы в `wave2/wave-report.json` условны на обученных моделях и не учитывают адаптивный подбор.

Исправлена карточка идеи №4: `--distill-boot-h` уже проверяет, что `next_label_change(y) > t+h`. Предложенный исходный тест дублировал действующий механизм; повторный запуск не делался.

### Подготовленные ZIP

| ZIP в `../submissions/` | Состав | Размер | Оценка public по парному замеру |
|---|---|---:|---:|
| `blend_twostream.zip` | прежний blend_pair_rw, raw-пара заменена full-data two-stream seed3 с весом .25 | 9,559,618 B | ~43.4 мин |
| `blend_twostream_lite.zip` | состав blend_rw: bootstrap .5 + two-stream .25 + baseline .0625 | 7,630,117 B | ~36.3 мин |

Время оценено относительно исходной посылки с записанным временем 37m51s, в одинаковом тесте: full ×1.145, lite ×0.958. Консервативная старая калибровка даёт 45.8–56.1 и 38.2–46.8 минут соответственно; у lite больше запас. Это не измерение новых ZIP на скорере.

Проверки: размер <20 MB, CRC и SHA256 архивов; 100,000 строк / 5 последовательностей, конечные выходы, детерминизм, корректный warm-up/reset и совпадение с взвешенной суммой участников. Two-stream преобразован в обычную GRU256 с блочно-диагональными весами без изменения функции (финальная PyTorch-погрешность 1.25e−6). Проверены INT8 и FP32, включая native v2 fallback и DataSphere. Полные отчёты и хэши: `wave2/`.

Сабмиты ещё не отправлены. Лучший записанный public остаётся 0.6666.

## Autonomous continuation, 2026-10-05 14:36 UTC

User target: top-1 = 0.6711. Recorded public remains 0.6666. No new submissions.

All 15 directions received an experiment or feasibility check, not exhaustive validation of every implementation. Exact mirror failed full feature-map verification; future/race/retrieval started as limited prototypes. Current registry: AUTONOMOUS_STATUS.json.

Main result: heterogeneous distillation into h256, two seeds. On 425 true-mask valid sequences / 123 chains excluding ALL train chains, student pair 0.7181503 vs teacher 0.7153250 (+0.0028253, conditional CI [0.0019575, 0.0038363]); single seeds 0.7176578 / 0.7174401. Two-stream teacher 0.7161344. Adding that teacher lowers the student pair score. Train-disjoint composite replication and final all-data training are active. These are NOT public/private score estimates.

Retrieval revived: 65536-row chain-balanced bank, GRU-only PCA64, k128, correction .05. Stride32 pair: train +0.0002378, valid +0.0002250; conditional CIs positive. Causal MLP student retains about +0.00016 on both, but intervals cross zero. Full-stream replay active.

Negative followups: capped32 bootstrap loses guarded h8 by .000524. Event8 adaptive compute loses matched all-update control. Warmup summary, quiet adapter, moment-ratio head, SSL pretrain and joint objective show no reliable gain. H512 distillation beats h128 but loses h256.

Baseline: old FP32 path kept cuDNN TF32 enabled; +16 shifts impaired optimizer conditioning. Corrected FT trains, but head FT loses .00133 on train diagnostic, removing baseline loses .00235. Keep baseline in robust candidate rather than select on valid alone.

Native: skip exactly zero GRU blocks while preserving global int8 activation scale. Bit-exact on all scalar/AVX2/VNNI backends and v2/avx2/v3 builds; reset/step parity pass. Relative projected public runtime: sparse single two-stream package 36.8 min, with linear int8 baseline 31.3 min; two-stream pair ~40.7 min. Not actual scorer time. Final seed4 training completed.

## Public result reported by user

`distilled_h256_pair.zip`, submission `26n9pg4y`, displayed 22:44 2026-10-05: **0.6674**, **24m57s**, **640 MB**. New best, +0.0008 vs .6666; user-provided leader .6711, gap .0037. Daily counter 4/5 at report. Train-disjoint compression replication also positive: valid +.0017429, train diagnostic +.0016844 vs teacher.
