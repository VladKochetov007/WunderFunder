# Статус всех25 после первого bounded screen

Дополнение: [R2](../idea_screens_r2/RESULTS.md) завершил fit-only выбор регуляризации и разложение EMA; это всё та же exposed-панель. Ниже сохранены конкретные R1 результаты, а не универсальные выводы о семьях. Лаги при выбранной alpha почти равны raw; joint EMA также почти равна raw, отдельные alpha1 EMA и selected endpoints расходятся по ранжированию. i0-subset остаётся лучше, но не изолирует i1.

Источник гипотез: [IDEAS_25.md](../no_fit_ideation_r1/IDEAS_25.md). Результаты: [RESULTS.md](RESULTS.md); предварительная подробная карта тестов сохранена в [SCREEN_MAP.md](SCREEN_MAP.md). Все измерения относятся только к текущей exposed-панели; положительный proxy не доказывает механизм.

| № | Полученное свидетельство | Следующий различающий тест / граница |
|---:|---|---|
| 1 | Измерены10fit-only future-delta proxies; стабильной формулы таргета не установлено | Другой заранее заданный оператор и корректная анонимная шкала; не расширять поиск горизонтов по выигрышу |
| 2 | История в разных fixed views имеет разные эффекты: EMA+ .01119, lag-view− .03206 | Это линейная ценность конкретных summaries; target-free state matching ещё не выполнен |
| 3 | Положительная композиция почти нигде не определена; отдельно названный proxy− .00088 | Сначала восстановить допустимый смысл преобразования; не log-ratio signed-чисел |
| 4 | Row-shape view− .01937 | Fit-only выбор регуляризации/контроль размера; физическая инвариантность не установлена |
| 5 | NOT_TESTED: нет обоснованного полного отражения112каналов и таргетов | Не создавать mirror labels без полной спецификации |
| 6 | Rolling-beta view− .00036; почти нулевой общий screen | Проверить fit-only регуляризацию; это не тест экономической beta или causal identification |
| 7 | Есть описательные масштабы trade/quote; общие физические единицы не доказаны | Target-blind проверка допустимости конкретных отношений; predictive cloud-view не запускался |
| 8 | NOT_TESTED: joint-support estimator отсутствует | Сравнить фиксированную совместную поддержку с axis-aligned bounds на fit |
| 9 | NOT_TESTED: sparse factor model не обучался | Sparse-vs-dense matched bottleneck; low-rank/PCA не считать подтверждением |
| 10 | NOT_TESTED: conditional innovation graph отсутствует | Input-only fit-component stability grouped VAR/residual associations |
| 11 | Joint signed/absolute-innovation EMA+ .01119, но выигрыш меняет знак без компоненты995 | Разделить signed и absolute части; нынешний screen не изолирует Jordan/variation механизм |
| 12 | NOT_TESTED: moment/sign-amplitude heads отсутствуют | Единая мера и правильный moment ratio, matched direct-WMSE head |
| 13 | NOT_TESTED: prior/correction recurrent model отсутствует | Сначала доказать отличие от имеющегося gated update; затем tiny recurrent comparison |
| 14 | Warmup99 constant-view− .04337, две цели реагируют разнонаправленно | Этот конкретный linear-context recipe не прошёл; не объявлять все initializer/context методы плохими |
| 15 | Только input-innovation descriptors; hidden state не измерялся | NOT_TESTED actual recurrent energy bound, не подменять его нормой входа |
| 16 | Согласованность scorer проверена; objective-gradient comparison не выполнялся | NOT_TESTED optimizer treatment; population algebra не равняет finite fits |
| 17 | NOT_TESTED: importance sampling не применялся | Fixed stride40 не является этим методом; нужен inverse-probability и matched time |
| 18 | NOT_TESTED: ортонормальный basis не реализован | EMA/lag результаты не тестируют matched-pole-span conditioning |
| 19 | i1-only=.36115, i0-only лучше joint raw в этом fit | i0-only также исключает8tailканалов: i1-effect не изолирован. Cross-fit common/residual mechanism NOT_TESTED; линейная репараметризация может иметь тот же span |
| 20 | NOT_TESTED: target gradients не собирались | Нельзя выводить gradient conflict из разнонаправленных score deltas |
| 21 | NOT_TESTED: gradient-covariance preconditioner отсутствует | Нужны реальные gradients и matched optimizer, не covariance coefficients |
| 22 | NOT_TESTED: heteroskedastic mean/variance objective отсутствует | Остаток ridge смешивает noise и misspecification; отдельная variance head нужна для проверки |
| 23 | NOT_TESTED: old/current-weight recurrent state replay не делался | Законный один и тот же prefix, сравнение state при фиксированной gradient horizon |
| 24 | NOT_TESTED: mixed-curvature penalty отсутствует | Обычная линейная модель его не реализует; допустимые in-support perturbations |
| 25 | NOT_TESTED, зависит от16 | Нельзя заменить uncertainty of training moments оценкой разброса evaluation scores |

Первые пригодные для следующей дешёвой волны вопросы: regularization/data-size confounding у i0-vs-i0+i1 и разложение joint EMA. Никакого решения о полном training campaign или candidate submission этот статус не содержит.
