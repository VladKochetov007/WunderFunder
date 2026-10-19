# Статус25карточек после R1–R13

Текущий [синтез свидетельств](../../unified/ideas/IDEA_SCREEN_SYNTHESIS_20261019_R1.md) отделяет положительные условные leads, закрытые exact variants и действительно NOT_TESTED механизмы. Число screens не означает, что все25механизмов проверены; новая R14-модель или очередь не допущены.

Дополнение R10: [target-free joint-support screen](../idea_screens_r10/RESULTS.md) обнаружил редкие distinct cases beyond marginal ranges на фиксированных raw0/52 и256строках.52компоненты из64,26в каждой заранее заданной половине; без меток/модели/grade. Это не доказательство предсказательной пользы или выбранного направления.

Дополнение R9: [прямая интервенция в recurrent carry](../idea_screens_r9/RESULTS.md) на одном неизменном GRU,8fit-компонентах и16окнах. Замена полной истории состоянием, восстановленным за256строк, меняет следующие128прогнозов на19.354% pooled relative RMS; все16окон выше1% описательного ориентира. Causal input features и их история одинаковы. Это зависимость прогнозов от старого state, не польза для меток, conditional-information test или выбранная архитектура;0обновлений/0backward/0grade.

Дополнение R8: [learned constant-variance control](../idea_screens_r8/RESULTS.md) не воспроизвёл преимущество R7 input-dependent objective: WP.501394 против stored.514297; primary+.012903, secondary кWMSE−.000528. Это одна новая CPU-arm с теми же756updates/init, без replayR7. Удалён также variance-to-trunkgradient, поэтому истинныйnoise/expressivity не изолированы.

Дополнение R7: [matched mean/variance MLP](../idea_screens_r7/RESULTS.md) даёт +0.012376 WP относительно своего WMSE-контроля на той же R3-панели, при одинаковых инициализации/756updates/12epochs. Это положительная bounded CPU проверка карточки22; более высокий stored ridge .536004, один seed, exposedpanel и связанное изменение gradient scaling/globalclip ограничивают вывод. Истинный источник выигрыша и перенос на GRU не установлены.


Дополнение R6: [реальные градиенты двух целей](../idea_screens_r6/RESULTS.md) измерены на одном фиксированном GRU checkpoint,8fit-компонентах и16заранее заданных окнах без обновления весов. Общий input/GRU trunk:12положительных/4отрицательных скалярных произведения, pooledcos+0.333540; shared hidden readout:+0.026854,3отрицательных окна. Это узкий фактический тест карточки20, а не доказательство training-wide вреда/пользы PCGrad или отдельных сетей. Остальные gradient/state механизмы не подменяются этой пробой.

Дополнение R5: [моментная проба карточки12](../idea_screens_r5/RESULTS.md) сравнила два декодера одних и тех же шести линейных голов. Деление на предсказанную амплитуду проиграло постоянному знаменателю на0.025555WP; все32drop-one-контраста отрицательны. Постоянный вариант дал+0.013658 к сохранённой R3 WMSE-рецептуре, но этот вторичный контраст меняет objective/functional form. Learned moment heads и совместное end-to-end обучение не проверены. Другие NOT_TESTED остаются в силе.

Дополнение R4: [четыре фиксированные nonlinear-пробы](../idea_screens_r4/RESULTS.md) не дали selected-выигрыша к identity-контролю на raw/истории; fixedalpha1 история сохраняет+0.002015WP. Это проверка конкретных случайных признаков/регуляризации на той же панели, не learned MLP, moment heads, gradient conflict или recurrent-state механизм. NOT_TESTED ниже остаются в силе.

Дополнение R3: [девять фиксированных probes](../idea_screens_r3/RESULTS.md) расширили проверку карточек2/11/19 до32evaluation-компонент, разделили i1/tail и сравнили nested32/64fit. На полном бюджете raw почти равенi0+tail; selected signed EMA+0.01046WP имеет положительные32/32drop-one-контрасты. Fixed-alpha1 absolute EMA также положителен, но fitting-CV выбирает противоположный по outer результат. Это линейные summaries/ablation, а не state matching, Jordan mechanism или gradient conflict. Остальные NOT_TESTED ниже не заменяются этими результатами.

Дополнение: [R2](../idea_screens_r2/RESULTS.md) завершил fit-only выбор регуляризации и разложение EMA; это всё та же exposed-панель. Ниже сохранены конкретные R1 результаты, а не универсальные выводы о семьях. Лаги при выбранной alpha почти равны raw; joint EMA также почти равна raw, отдельные alpha1 EMA и selected endpoints расходятся по ранжированию. i0-subset остаётся лучше, но не изолирует i1.

Источник гипотез: [IDEAS_25.md](../no_fit_ideation_r1/IDEAS_25.md). Результаты: [RESULTS.md](RESULTS.md); предварительная подробная карта тестов сохранена в [SCREEN_MAP.md](SCREEN_MAP.md). Все измерения относятся только к текущей exposed-панели; положительный proxy не доказывает механизм.

| № | Полученное свидетельство | Следующий различающий тест / граница |
|---:|---|---|
| 1 | Измерены10fit-only future-delta proxies; стабильной формулы таргета не установлено | Другой заранее заданный оператор и корректная анонимная шкала; не расширять поиск горизонтов по выигрышу |
| 2 | PARTIAL: R9 actual carry-state intervention даёт19.354% pooled relative prediction RMS при recent256 vs full; EMA/lag результаты сохранены | Зависимость fixed checkpoint от старого carry при одинаковых causal features; target-free matching и полезность старой истории для меток не измерены |
| 3 | Положительная композиция почти нигде не определена; отдельно названный proxy− .00088 | Сначала восстановить допустимый смысл преобразования; не log-ratio signed-чисел |
| 4 | Row-shape view− .01937 | Fit-only выбор регуляризации/контроль размера; физическая инвариантность не установлена |
| 5 | NOT_TESTED: нет обоснованного полного отражения112каналов и таргетов | Не создавать mirror labels без полной спецификации |
| 6 | Rolling-beta view− .00036; почти нулевой общий screen | Проверить fit-only регуляризацию; это не тест экономической beta или causal identification |
| 7 | Есть описательные масштабы trade/quote; общие физические единицы не доказаны | Target-blind проверка допустимости конкретных отношений; predictive cloud-view не запускался |
| 8 | PARTIAL: R10 fixed2D hull даёт274joint-only queries/31552 (0.8684%),52/64fit components | Это геометрическое отличие от same-window box; target usefulness, full-dimensional support и matched outlier/volatility controls NOT_TESTED |
| 9 | NOT_TESTED: sparse factor model не обучался | Sparse-vs-dense matched bottleneck; low-rank/PCA не считать подтверждением |
| 10 | NOT_TESTED: conditional innovation graph отсутствует | Input-only fit-component stability grouped VAR/residual associations |
| 11 | PARTIAL: [R3](../idea_screens_r3/RESULTS.md) selected signed EMA+.010459WP vs raw, все32drop-one+; fixedalpha1 absoluteEMA+.029060, но selected−.004323 | Линейная causal history representation; не isolated Jordan-split. Fit-CV/outer ranking mismatch не разрешён выбором alpha по outer |
| 12 | PARTIAL: R5 shared linear moment heads; ratio−constant−0.025555WP, все32drop-one отрицательны | Learned heads/joint objective остаются NOT_TESTED; вторичный constant-vs-R3+0.013658 не изолирует механизм |
| 13 | NOT_TESTED: prior/correction recurrent model отсутствует | Сначала доказать отличие от имеющегося gated update; затем tiny recurrent comparison |
| 14 | PARTIAL: [R13](../idea_screens_r13/RESULTS.md) immutable mean99 context WP .462646 vs current-bypass .499683/zero .507387, primary Δ−.037037; both warmup contrasts/all32drop-one negative | One exposed CPU MLP seed/756updates per arm closes this fixed prefix-mean recipe. Does not test ordered/learned context, GRU memory or general contextual usefulness; R9 remains separate checkpoint-conditional carry evidence |
| 15 | R9 измерил actual state-distance snapshots при carry reset; energy regularizer не применялся | NOT_TESTED actual recurrent energy bound; descriptive state norms не доказывают пользу штрафа |
| 16 | PARTIAL: [R12](../idea_screens_r12/RESULTS.md) full-fit pooled-correlation objective WP .405817 vs matched WMSE .490743, Δ−.084925; обе цели/all32drop-one отрицательны | Один seed/128full-fit updates на exposed R3. Оба loss используют global moments; закрыт этот exact objective/AdamW вариант, не проверены global-vs-local и другие schedules/GRU/full recipe |
| 17 | PARTIAL: [R11](../idea_screens_r11/RESULTS.md) fixed initial-gradient sampling WP .454911 vs uniform .488192, Δ−.033282; все32drop-one отрицательны | Один seed/756updates на exposed R3; initial norm proxy + exact inverse-probability correction. Этот вариант закрыт; equal-time/dynamic sampling/full-recipe эффект не проверены |
| 18 | NOT_TESTED: ортонормальный basis не реализован | EMA/lag результаты не тестируют matched-pole-span conditioning |
| 19 | [R3](../idea_screens_r3/RESULTS.md) full64fit: presence i1−.000496 безtail/−.000044 сtail; drop-one меняетзнак | Нет устойчивого основания удалятьi1; cross-fit common/residual mechanism NOT_TESTED. Обычная линейная репараметризация может сохранить span |
| 20 | PARTIAL: R6 actual gradients, один frozenGRU/8fit-компонент/16окон; trunk pooledcos+0.333540,4negative; hiddenreadout+0.026854,3negative | Не training-wide prevalence/causalharm/PCGradbenefit; optimizer intervention и другие checkpoints не проверены |
| 21 | NOT_TESTED: gradient-covariance preconditioner отсутствует | Нужны реальные gradients и matched optimizer, не covariance coefficients |
| 22 | PARTIAL: R7 NLLmean .514297 vs WMSE.501921; R8 constantvariance .501394, primarygap+.012903,32/32drop-one+ | Этот globalscalingcontrol не повторил gain; variance-to-trunkgradient также меняется, истинныйnoise/expressivity не изолированы; одинseed/exposedpanel/GRUнепроверен |
| 23 | NOT_TESTED: old/current-weight recurrent state replay не делался | Законный один и тот же prefix, сравнение state при фиксированной gradient horizon |
| 24 | NOT_TESTED: mixed-curvature penalty отсутствует | Обычная линейная модель его не реализует; допустимые in-support perturbations |
| 25 | NOT_TESTED, зависит от16 | Нельзя заменить uncertainty of training moments оценкой разброса evaluation scores |

Условные положительные leads — causal signed EMA и coupled mean/variance objective; constant-denominator moment recipe остаётся вторичным descriptive lead. Они не выбраны как новая архитектура/очередь. Следующий дешёвый тест должен соответствовать заявленному механизму и менять решение, а не повторять автоматическую серию слабых proxies. Полный training campaign или candidate submission не допущены этим статусом.
