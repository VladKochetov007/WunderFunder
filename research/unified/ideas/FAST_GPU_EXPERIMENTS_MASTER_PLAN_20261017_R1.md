---
kind: method
status: proposed
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015, H-STUDENT-PARENT-GROWTH-20261015, H-087, H-138, H-STUDENT-LADDER-20261014, H-CAUSAL-TRANSFORMER-20261017, H-CONTEXT-TRANSFORMER-20261017, H-SELECTIVE-SSM-20261017, H-TEACHER-FAMILY-KD-20261017, H-CAUSAL-CIRCLE-FEATURE-20261017, H-PRETRAINED-TS-TEACHER-20261017]
experiment_ids: [E-PORTABLE4-PARENT-ANCHOR-F0-S3-R1, E-PORTABLE4-GRU-GROWTH-F0-S3-R1, E-PORTABLE4-FIELD-CLOCK-F0-S3-R1, E-PORTABLE4-H512-SEED4-CONFIRM-F0-R1, E-ARCH-GPU-A-F0-S3-R1, E-ARCH-GPU-B-F0-S3-R1, E-ARCH-GPU-C-F0-S3-R1, E-ARCH-GPU-D-F0-S3-R1]
evidence_paths:
  - research/campaign_2026-10-17/fast_gpu_queue_r1/README.md
  - research/campaign_2026-10-17/architecture_gpu_wave_r1/README.md
  - research/unified/reviews/TEAM_TOP_AND_NEXT_TESTS_20261017_R1.md
  - research/unified/reviews/CPU_STUDENT_FRONTIER_H512L1_RESULT_REVIEW_20261015_R1.md
---

# Полные GPU-эксперименты: draft PR, повторения и список гипотез

Основной режим — полные fitting-данные,
заранее заданный полный горизонт, 30 парных fold/seed-ячеек на arm, strict FP32 и
эксперименты в draft PR. Обязательная лестница 10%/30%, разработка менеджера очереди
и квантование во время исследований исключены из текущего плана.

GitHub API подтвердил 14 открытых draft PR с base `research-team` и ссылку на
workflow в `main/AGENTS.md`. Проверенные revisions: main
`0edd432271991250f4718f9b68f33340eff46214`, research-team
`db3f37b50e69f39b16410aa2ad5c2a14f6aaf224`. Локальная ветка
`codex/transformer-mamba-gpu-wave` с исходным планом `c3fb8472` — другой checkout
с более ранними trainer/пакетами; их готовность нельзя переносить на новый main.

Переписка и удалённые документы описывают протокол. Доступ к B300, профили и
выполнение новых 30-ячеечных матриц здесь не проверены.

## Что хотим получить

Сохранить существующие этапы **fit → predict → score**, чтобы
каждая законченная проверка давала сопоставимый результат, затраты и следующий
шаг. Основная единица прогресса — оценённое сравнение с контролем, а не число
запущенных процессов или занятых GPU. Новый менеджер очереди пока не разрабатываем.
Каждый самостоятельный механизм оформляется draft PR в `research-team`.

Доступ к GitHub был восстановлен, исходники и сохранённые командные материалы
разобраны. Ссылки на рецепты ниже ведут к конкретным источникам; наличие исходников
не означает, что все данные, веса и teacher-cache уже восстановлены на GPU-хосте.

## Формат эксперимента и источники истины

Использовать уже опубликованные
[workflow](https://github.com/VladKochetov007/WunderFunder/blob/0edd432271991250f4718f9b68f33340eff46214/research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md)
и [English PR template](https://github.com/VladKochetov007/WunderFunder/blob/0edd432271991250f4718f9b68f33340eff46214/research/unified/templates/EXPERIMENT_DRAFT_PR.md).
Общие правила уже добавлены в main; не создавать параллельный стандарт.
Этот MD — русскоязычная сводка и навигация, а точный контракт берётся из workflow.

Маршрут: `main → research-team → experiment/<id>`, draft PR направлен в
`research-team`. Полезные положительные, отрицательные и неопределённые результаты
сохраняются вместе с источниками; перенос согласованной совокупности изменений в
main — отдельная интеграция. Сам факт merge не делает модель победителем.

В каждом draft PR должны быть ответы:

1. **Линия и владелец:** teacher, student или связь teacher→student; H/E IDs,
   родительская гипотеза, source SHA и фактическая стадия реализации.
2. **Почему должно работать:** недостаток текущей системы, механизм, ожидаемое
   наблюдение, альтернативное объяснение и различающий их контроль.
3. **Где дополнительный compute покупает качество:** размер/контекст teacher,
   длительность оптимизации, teacher diversity или student capacity. Указать
   бюджет и альтернативную стоимость, не обещать pay-to-win без результата.
4. **Точный опыт:** candidate/control, полный fit, targets, precision, optimizer,
   scheduler, batch/accumulation, updates/epochs, reset и checkpoint policy.
5. **Валидация:** матрица 30 ячеек на arm, ancestry, основной эффект, интервалы,
   множественные сравнения, ограничения мощности и репрезентативности.
6. **Результат, меняющий решение:** практически полезный эффект с обоснованием,
   что означает успех, неопределённость и отрицательное наблюдение.
7. **Исполнение и возврат:** реальные команды, недостающие адаптеры, профили,
   resume, логи, predictions, оценки и отдельные архивы evidence/weights.

## Полный цикл исследования

```text
Подготовить данные и источники один раз на хост
    ↓
Draft PR: гипотеза, контроль, полная fold/seed-матрица и бюджет
    ↓
Проверить реализацию и память на целевом GPU
    ↓
Полный fit в каждой ячейке → predict → official score → paired comparison
    ↓
Сохранить результат, время, память и происхождение артефактов
    ↓
Синтез всех ячеек → решение / следующий механизм / более крупный teacher
```

### 1. Подготовка один раз

- Разместить исходные данные и необходимые кэши на диске GPU-хоста. Не скачивать
  и не распаковывать один набор отдельно для каждого worker.
- Проверить revision исходников, хеши файлов, split и membership обучающих групп.
  Восстанавливать только нужные checkpoints и predictions; большой HF-архив
  целиком не является обязательной зависимостью.
- Подготовить teacher-targets для тех опытов, которым нужна дистилляция. Кэш
  переиспользуется только при совпадении teacher checkpoint, нормализации,
  признаков, порядка строк, масок и допустимого происхождения обучения.
- Подготовить контрольные predictions. Для нового seed, fold, membership или бюджета
  нужен соответствующий контроль; старый сильный endpoint может быть ориентиром,
  но не автоматически становится парным контролем.
- Зарезервировать общий RAM/CPU/I/O-бюджет всех workers и место под результаты.
  Число GPU само по себе не определяет, сколько одновременных загрузчиков данных
  выдержит хост.

### 2. Полные данные и 30 парных ячеек

Использовать всю разрешённую **fitting-часть каждого outer fold** с первого
научного опыта, полные последовательности и заранее объявленный горизонт.
«Все данные» не означает включить оценочные компоненты в fit. Final all-data
training для сабмита — отдельный этап без held-out оценки на его обучающих данных.
Малые synthetic/parity и real-shape resource checks остаются инженерными
проверками; они не заменяют полный научный опыт и не считаются репликациями.

Точная матрица из опубликованного workflow:

| Outer fold | Fitting seeds | Ячеек на arm |
|---|---|---:|
| F0 | 1001–1008 | 8 |
| F1 | 1001–1008 | 8 |
| F2 | 1001–1007 | 7 |
| F3 | 1001–1007 | 7 |
| **Всего** | | **30** |

Candidate и control проходят одни и те же ячейки: **60 fit для двух arms,
120 для четырёх**. Teacher fits, построение caches и student fits считаются
отдельно. Уже существующие controls можно переиспользовать только при точном
совпадении нового контракта; старые seed3/4 не входят в seeds1001–1008 незаметно.

Сохраняем целые компоненты, sequence reset, исходные scored/prediction masks и
одинаковую fitting/evaluation membership у пары. Для каждого fold нужны чистые
teacher, initializer, normalizer и propensity ancestors. Изменение CLI `--fold`
не очищает родителя, обученного на оценочных группах другого fold.

Полная evaluation-панель не сокращается. У F0 это 468 групп; full-valid export
содержит 1,873 группы, включая fitting-valid, поэтому весь export нельзя объявлять
held-out оценкой. Для F1–F3 берутся их собственные проверенные manifests.

### 3. Полная точность

Strict FP32 — weights, activations, states, gradients и optimizer state в FP32;
autocast, GradScaler и TF32 отключены. BF16/FP16/FP8 и Q8/Q4 не участвуют в научной
волне. Флаг `fp32` сам по себе не доказывает отключение TF32: настройки и реальные
операторы нужно проверить на целевом GPU. Поддержку FP32 attention kernels нельзя
вывести из рекламной BF16/FP8 производительности B300.

Перед долгим fit — ограниченный профиль реального shape, forward/backward и
state/reset parity. При нехватке памяти разрешённые checkpointing, DDP, sharding
или batch/accumulation меняются явно; precision незаметно не снижается.

### 4. Результат каждой ячейки и всей матрицы

Сохраняем checkpoint/provenance, полный нужный prediction panel, official
`global_wp_clip2_v1`, оценки по двум targets, разницу с контролем и paired
неопределённость по evaluation-компонентам. Отдельно записываем время подготовки,
fit, predict, score; GPU peak memory, доступную RAM-телеметрию и число завершённых
optimizer updates.

Успешный exit обучения без predictions и оценки не считается законченным
сравнением. OOM, ошибка сборки или несовместимость кэша — технический результат,
а не доказательство плохой гипотезы. Не скрываем неудавшиеся arms из общей таблицы.

Назначение следующей ячейки делает принимающая сторона через свои существующие
средства; разработка scheduler не входит в текущую работу. Небольшие независимые
GRU fit распределяются по картам; большой teacher может использовать DDP/sharding.
Фактические B300, topology, CUDA и профиль подтверждает исполнитель. Starter или
`run_plan.py`, печатающий матрицу, не является готовым trainer.

### 5. Статистика и практическое решение

Основной эффект workflow — средняя paired разница WP по seeds внутри каждого
fold, затем среднее четырёх fold-эффектов с равными весами 1/4:
`D = (1/4) Σ_f mean_s[WP(candidate_f,s) − WP(control_f,s)]`.
WP считается отдельно для каждой модели на всей её outer evaluation. Усреднение
predictions по seeds создаёт другой endpoint и не подменяет эту статистику.

30 запусков не являются 30 независимыми датасетами: fitting-наборы перекрываются,
source components повторяются, seed identities используются в нескольких folds.
Разделяем три вида неопределённости:

- **Обучение:** workflow задаёт 10,000 bootstrap draws восьми seed identities с
  одинаковыми multiplicities во всех folds и нормировкой по доступным ячейкам.
  Это приближённый 95%-интервал, условный на этих данных, четырёх folds и ancestry;
  точный алгоритм, включая отсутствующий seed1008 в F2/F3, берётся из workflow.
- **Оценочные данные:** paired resampling source-qualified components при
  фиксированных моделях, одинаковый для обоих arms и всех seeds. Такой интервал
  не включает неопределённость самого обучения.
- **Выбор гипотез и перенос:** показать все arms, failures/exclusions и fold means.
  Для заявления о значимости заранее определить null, primary comparison family
  и поправку за множественность, например Holm. Нужна оценка мощности по paired
  вариации; число 30 само по себе не гарантирует значимость или точное покрытие.

Порог практически полезного выигрыша задаётся в PR до результатов с учётом
compute и роли teacher/student. Успех — полезный устойчивый эффект; пересечение
интервала с нулём допускает целевой следующий опыт; отрицательный результат
ограничивает конкретный рецепт. Ни четыре exposed folds, ни новый seed не дают
свежего временного holdout. Валидация требует анализа source/regime coverage,
дубликатов и drift; component disjointness сама по себе не доказывает временной
репрезентативности. Локальную delta нельзя прибавлять к public score.

## Уже опубликованные draft PR

Все ссылки ниже проверены через GitHub API: open, draft, base `research-team`.
Это реестр предложений и кода, а не подтверждение выполненной 30-ячеечной матрицы.

| PR | Механизм | Ветка идей | Подтверждённая стадия / что остаётся |
|---|---|---|---|
| [#1](https://github.com/VladKochetov007/WunderFunder/pull/1) | Causal Transformer с output AR и отдельным auxiliary прогнозом изменений признаков | Teacher | Архитектурная постановка и starter; B300 profile и полные fit отсутствуют |
| [#2](https://github.com/VladKochetov007/WunderFunder/pull/2) | GRU: ширина против глубины при matched compute | Student | Есть старые one-seed F0 screens; новой полной матрицы и scaling law нет |
| [#3](https://github.com/VladKochetov007/WunderFunder/pull/3) | GRU/Transformer teachers для одинакового GRU student | Связь | Нужны teacher fits, fold-clean caches и полная trainer/export интеграция |
| [#4](https://github.com/VladKochetov007/WunderFunder/pull/4) | Per-field age против семи групповых clocks | Student / связь | C/G fitted; F имел 50-update profile, полного F fit нет |
| [#5](https://github.com/VladKochetov007/WunderFunder/pull/5) | Function-preserving widening | Student | Initializer parity/profile есть; полный W fit не выполнен |
| [#6](https://github.com/VladKochetov007/WunderFunder/pull/6) | Frozen-parent output anchor | Связь | S profile есть; полного S fit нет |
| [#7](https://github.com/VladKochetov007/WunderFunder/pull/7) | Learned causal initial state | Student | Статические рецепты; validation/profile/fit новой матрицы впереди |
| [#8](https://github.com/VladKochetov007/WunderFunder/pull/8) | Grouped Transformer input encoding | Teacher | Starter, результата нет |
| [#9](https://github.com/VladKochetov007/WunderFunder/pull/9) | Past-only smoothing с сохранением raw inputs | Teacher / student | Causal helper, fit не выполнен |
| [#10](https://github.com/VladKochetov007/WunderFunder/pull/10) | Trailing geometry / circle residual | Teacher / student | Диагностический helper; смысл координат не подтверждён |
| [#11](https://github.com/VladKochetov007/WunderFunder/pull/11) | Chronos/TimesFM teacher transfer | Teacher / связь | Нет готового адаптера; checkpoint, условия weights и input semantics не закрыты |
| [#12](https://github.com/VladKochetov007/WunderFunder/pull/12) | Расширение clean recurrent teacher cohort | Teacher / связь | Один clean F3 teacher и FP32 exports есть; cohort неполон |
| [#13](https://github.com/VladKochetov007/WunderFunder/pull/13) | Fixed checkpoint trajectory averaging | Student | Два F0 seed-average артефакта построены; export/grade ожидаются |
| [#14](https://github.com/VladKochetov007/WunderFunder/pull/14) | Innovations сверх moment features | Teacher / student | Предыдущая пара остановилась на cache schema до fit; это execution failure |

Источник статусов — [опубликованный inventory](https://github.com/VladKochetov007/WunderFunder/blob/0edd432271991250f4718f9b68f33340eff46214/research/unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md).
Заголовок PR, наличие helper или план запуска не подтверждают готовность к B300.

## Исторический переносимый GRU-пакет: использовать исходные рецепты

В прежнем one-seed пакете это **четыре сравнения и пять новых fit** при доступных точных C/G controls.
Если controls отсутствуют, их восстановление или повторное получение тоже
потребует работы. [Подготовленный пакет](../../campaign_2026-10-17/fast_gpu_queue_r1/README.md)
привязан к исходникам `685d5331`; запускать его нужно из этой версии, не подменяя
хеши новым trainer. Этот расчёт числа fit относится к старому пакету. Для текущего
протокола mechanisms переносятся в соответствующие draft PR и получают 30 ячеек
на каждый arm; один старый seed не является их полным исполнением.

| Опыт | Что выясняем | Candidate и контроль | Статус и ограничения |
|---|---|---|---|
| S: замороженный родитель вместо T3-цели | Польза KD приходит от дополнительной информации T3 или от сохранения функции родителя? | Student с targets фиксированного parent против C с T3; те же parent, seed3, F0, две эпохи, оценка CP1 | Рецепт подготовлен. Один новый fit. Калибровка targets и их источник меняются вместе; это не идеальное доказательство единственной причины. |
| W: расширение H256×2 → H384×2 | Помогает ли дополнительная ёмкость при сохранении исходной функции? | Увеличенный parent против C, одинаковые T3-targets и бюджет | Рецепт подготовлен. Один новый fit. Проверяем качество после обучения и стоимость CPU, а не только начальную parity. |
| F: индивидуальные часы признаков | Полезнее ли время с последнего обновления каждого признака, чем групповые часы? | F против G; C — дополнительный ориентир | Рецепт подготовлен. Один новый fit. Меняется также входное представление; часы считаются причинно, будущие обновления недоступны. |
| H512×1: повтор на новом seed, затем fold | Повторяется ли ранее найденный прирост около +0,00346? | Холодные H512×1 и H256×2 с общими T3-targets на одном seed4/F0; одна эпоха, B8/C2000, FP32, CP0 | Подготовленная пара требует двух fit. Она меняет seed, но не fold. Другой fold — отдельная следующая пара; исходный прирост не является public-прогнозом. |

Источник числа +0,00346 — [сохранённая оценка H512 на F0/seed3](../reviews/CPU_STUDENT_FRONTIER_H512L1_RESULT_REVIEW_20261015_R1.md):
0,67323776 против matched H256×2 0,66977764; paired delta +0,00346012,
условный 95%-интервал по 141 компоненте [+0,00200307; +0,00490448]. Это cold
student с KD, а не labels-only: цель смешивает 0,1 labels и 0,9 clean-T3 target,
используются propensity weights, от parent берётся только общая нормализация.
Интервал относится к фиксированным predictions и не включает вариацию seed,
teachers, выбора модели или перенос на другой fold/public. Native Q8, финальный
ZIP и CPU-время этим результатом не подтверждены.

S/W/F и архитектурные направления могут выполняться независимо при выделенных
ресурсах. Точные данные, teacher-cache, C/G и исходные receipts проверяет
исполнитель; они не считаются автоматически скачанными на новый хост.

## Ранее подготовленная архитектурная четвёрка и новый teacher-план

[Код и команды запуска](../../campaign_2026-10-17/architecture_gpu_wave_r1/README.md)
подготовлены в локальной ветке до нового workflow: **четыре модели, четыре one-seed fit**. Общие условия — scratch,
labels-only, seed3/F0, одна эпоха, batch8/chunk2000, FP32, seeded order, одинаковые
fit membership и нормализация, фиксированный CP0. Это сравнение при одинаковом
объёме обучения; число параметров и время вычислений не объявляются одинаковыми.
Эта четвёрка не подменяет 120 fit текущей четырёх-arm матрицы и не доказывает
совместимость с новым модульным trainer в main. Перед переносом нужны адаптация,
полный precision audit, fold/seed bindings и профиль на фактических GPU.

| Arm | Модель | Проверяемый вопрос |
|---|---|---|
| A | GRU H256×2 | Общий контроль для новых архитектур |
| B | Causal Transformer, width256, четыре слоя, восемь heads, окно256 | Даёт ли attention полезный сигнал сверх GRU? |
| C | Такой же Transformer, окно2048 | Даёт ли более длинный контекст выигрыш против B и A? |
| D | Mamba-2, width256, четыре слоя, state64 | Полезна ли selective state-space память по качеству и стоимости? |

До полного fit runner требует GPU-проверки причинности, state/reset, chunk parity,
градиентов и короткий реальный batch. Он проверяет конечность updates и GPU
allocator peak не выше 80% памяти устройства. Сборка Mamba и все эти проверки
на целевом оборудовании пока не выполнены.

После fit оцениваются отдельные модели и заранее заданные смеси **90% A + 10%
B/C/D**. Это три дополнительных оценки без новых обучений. Более слабая модель
может быть полезна в смеси; выигрыш teacher ещё не доказывает перенос в student.

Основной новый teacher-дизайн описан в
[CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300](https://github.com/VladKochetov007/WunderFunder/blob/0edd432271991250f4718f9b68f33340eff46214/research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md):
отдельные AR-readout и future-observation auxiliary arms, большие модели,
длинный контекст и полное обучение. В документе предложены примерно
86M/207M/682M/1.65B параметров и горизонты 6/12/24/48 эпох; это проектные варианты,
не измеренные B300 возможности или обязательство запустить весь grid.
Для каждого выбранного контраста заранее фиксируются бюджет и 30 ячеек на arm.
Mamba остаётся отдельной альтернативой из локального пакета; среди проверенных
14 PR самостоятельный Mamba draft не найден.

## Где compute может дать преимущество и как проверять scaling

Дополнительные карты покупают возможность обучать более сильного teacher,
увеличивать контекст/оптимизационный горизонт, получать teacher diversity и
сравнивать перенос в student. Число GPU и параметров сами по себе не дают прироста.
Разделять улучшение teacher score, фактический transfer и итоговый student score.

Для GRU depth-versus-width взять несколько заранее заданных shapes и общую
полную fold/seed-матрицу. Провести **два явно разных сравнения**: одинаковый полный
горизонт данных/updates и одинаковый заявленный вычислительный бюджет. На B300
сначала измерить стоимость shapes; epochs/updates и допуск на расхождение compute
зафиксировать до quality results. Одинаковое число параметров или эпох не означает
одинаковые FLOPs, GPU-часы или скорость. Показать WP против параметров, processed
rows, training compute и wall time; loader/compile/export учитывать отдельно.
Серьёзным contenders дать одинаковый ограниченный бюджет настройки LR/schedule.

Работы [Kaplan et al.](https://arxiv.org/abs/2001.08361) и
[Hoffmann et al.](https://arxiv.org/abs/2203.15556) описывают эмпирические scaling
зависимости для language-model loss и распределения compute. **Они не дают готовой
формулы «глубже или шире GRU» для нашего weighted Pearson.** Здесь scaling law пока
не установлена. Нужны несколько масштабов и compute budgets с повторениями,
контролем optimization и проверкой предсказаний зависимости на неиспользованном
масштабе. Нельзя выдать две точки H256/H512 или степенную кривую по шумному WP за
универсальный закон и экстраполировать public score.

## Другие опыты: teacher, student и ансамбли

| Опыт | Минимальное содержательное сравнение | Что нужно и как трактовать |
|---|---|---|
| GRU-учитель против Transformer-учителя | Один и тот же GRU-студент: labels-only / KD от GRU / KD от заранее выбранного Transformer | Три student arms × 30 ячеек = 90 student fit, плюс отдельный teacher budget; одинаковые initialization, seed, порядок, бюджет и KD coefficient для двух KD arms. Нужны fitting-group teacher predictions: full-valid export для оценки их не заменяет. Полная generic KD-cache интеграция ещё не подтверждена. |
| Смешанный GRU + Transformer teacher | Тот же student с фиксированной смесью двух teachers против лучшего заранее определённого одиночного teacher | Следующий interaction arm. Коэффициенты задаём до оценки; не предполагаем, что эффекты складываются. |
| Mamba как teacher | Тот же student и бюджет, Mamba-targets против выбранного recurrent/attention teacher | Имеет смысл после причинно корректного teacher fit и оценки его вклада; нужен тот же протокол cache/ancestry. |
| Командный ансамбль → один или два student | Нынешние T3-targets против фиксированного разнообразного командного teacher | Нужны fold-qualified teachers. Final weights, обученные на validation, не подходят для заявления о чистом CV. Поздний FTT recipe/ancestry остаётся неполным. |
| Вклад участников топового ансамбля | Фиксированная полная смесь против удаления одного участника с перенормировкой весов | При наличии predictions новые fit не нужны. Измеряем вклад и стоимость полного endpoint. Final-data диагностику не выдаём за held-out качество. |
| Перекрёстные teacher-targets | Один student получает in-sample либо cross-fitted targets от сопоставимых teachers | Дороже из-за teacher fits. Проверять после S/KD-аудита, если остаётся вопрос о переобучении учителя. In-sample KD на student-fit данных сам по себе не запрещён; оценочные группы исключаются из ancestry всегда. |

Везде применяется новая полная матрица. Старый единственный A/B checkpoint из F0
нельзя использовать как чистого teacher для других folds. Семейство teacher и
правило выбора его checkpoint фиксируются заранее; cache привязан к конкретному
fold, teacher seed/ancestry и точному student-fit membership.

Сохранённый командный `pair_ftt2` имеет public-запись 0.6682 и состоит из четырёх
GRU H256×2; FTT в имени не доказывает Transformer teacher. Это исторический
receipt, не заново проверенная позиция живого leaderboard. Подробности и
ограничения — в [разборе командного топа](../reviews/TEAM_TOP_AND_NEXT_TESTS_20261017_R1.md).

## Другие опыты: признаки и состояние

| Опыт | Контроль | Что фиксируем перед запуском |
|---|---|---|
| Принадлежность окружности ±ε | Тот же GRU без геометрической фичи | Координаты пока не названы, фото отсутствует. Нужны формула, единицы ε и warm-up. Геометрия строится по fit-данным либо уже наблюдённому префиксу, без будущих точек. |
| Непрерывный радиальный residual | Бинарный индикатор принадлежности либо заранее выбранный baseline | Проверяет, полезна ли степень отклонения, а не только порог. Выбор ε/масштаба не подгоняется по итоговому validation score. |
| Нормированные innovations | Raw features + moment-control против raw features + innovations | Одинаковые width, targets и бюджет. Старый R1 остановился на несовместимости cache до fit; исправленный R2 подготовлен. Это ещё не отрицательный результат гипотезы. |
| Инициализация состояния по началу sequence | Постоянное состояние / первая строка / разрешённый warm-up prefix | Использовать только доступный к прогнозу префикс, одинаковый student и reset. Рецепт initializer family уже есть; нового quality result нет. |
| Длина градиентной памяти | Detach500 против detach2000 при одинаковом update-window2000 | Отделяет влияние backpropagation history от частоты обновления optimizer. Не путать длину forward memory с длиной градиентного графа. |
| Причинное сглаживание входов, PR #9 | Raw inputs против raw + trailing EMA/rolling signal | Сохраняем raw ветку, чтобы не потерять резкие изменения. Только текущие и прошлые строки; centered filters, backward fill и full-sequence statistics недопустимы. Одинаковый backbone; изменение input width явно учитывается. |
| Разные input encodings Transformer, PR #8 | Один row-token с общим MLP против group-specific encoders с общей проекцией | Порядок колонок и подтверждённые feature groups фиксированы; width/context/backbone общие. Feature identity и временная position — разные вещи. Индекс столбца не объявляется уровнем стакана без основания. |

Предложение «JEV обучаем» пока не превращается в конкретный recipe: в переданной
переписке и проверенных canonical-документах нет определения JEV, objective,
входов и контроля. Не подменять его предположением о JEPA или другой архитектуре;
нужна расшифровка автора. Факт уже идущего обучения также не подтверждён receipt.

Для AR Transformer feedback — предыдущий **собственный prediction**, а не
`y[t-1]`: target описывает будущее, его доступность к следующей строке не задана.
Future observations допустимы как отдельно объявленные auxiliary targets при
обучении; они не становятся входом текущего representation.

Источники этих ветвей и ограничения существующих реализаций перечислены в
[индексе гипотез](README.md) и [архитектурном дополнении команды](TRANSFORMER_GPU_WAVE_20261017_R1.md).

## Chronos, TimesFM и другие pretrained-учителя

Это отдельная исследовательская ветка, пока без готового адаптера к trainer.
Сначала инженерно подтвердить API, row-time alignment и FP32 профиль.
Выбранный научный контраст затем получает полный fit и 30 парных ячеек на arm;
короткий API-профиль не заменяет сравнение качества.

- **Chronos-2:** причинный адаптер observable feature histories → representation
  или feature forecast → readout двух targets. Сравнить с простым readout на тех
  же данных, чтобы отделить пользу pretraining от новой головы/представления.
- **TimesFM:** тот же вопрос и evaluation protocol, отдельный source/version pin.
  Поддержка covariates зависит от версии. Для весов 3.0 отдельно проверить условия
  использования; наличие открытого кода не означает одинаковые лицензии weights.
- **После полезного teacher signal:** тот же фиксированный GRU-student с/без
  новых targets. Итог исследования — полное FP32-качество студента, а не benchmark модели
  на чужой задаче.

Стандартный forecast с историей настоящего target не является готовым решением
нашего callback `X[<=t] → y[t]`: прошлые true labels в нём не доступны. Причинными
должны быть также normalizer, patches и embeddings. Вычисление embedding по всей
20k sequence может передать будущие признаки в текущий прогноз.

Официальные источники:
[Chronos-2](https://huggingface.co/amazon/chronos-2),
[TimesFM](https://github.com/google-research/timesfm),
[TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch).
Польза этих pretrained-моделей на наших анонимных признаках пока не измерена.

## Только после исследований: минимальный пакет и сабмит

Финальные условия: **Linux/Python3.11, один CPU, RAM16GB, 60 минут на весь тест,
ZIP20MB, без GPU и интернета**. Исследовательский teacher может быть больше;
готовый submission endpoint обязан пройти отдельную проверку. **Квантование не
входит в текущие эксперименты, их primary score или критерий выбора teacher.**
Оно возвращается только при подготовке конкретного сабмита с измерением потери
качества относительно выбранного FP32 endpoint.

Командное «попробовать минимальную версию запаковать» сохраняется как отдельная
инженерная задача: реальный offline ZIP, Python3.11, causal state/reset, размер,
RAM и время. Это не новая научная гипотеза и не подтверждение, что большой teacher
напрямую подходит для CPU. Новый Transformer/Mamba CPU exporter ещё не готов;
возможен путь через GRU student. Пакет не должен задерживать FP32 quality research.

Уже добавлен [CPU budget checker](../../../scripts/check_submission_budget.py)
и [инструкция с Docker image](../../campaign_2026-10-17/architecture_gpu_wave_r1/README.md#final-zip-check-on-one-cpu).
Он работает с реальным ZIP, закрепляет одно ядро, ограничивает RAM без swap,
отключает сеть и останавливает прогон через час. Результат привязывается к ZIP,
image и входной панели по хешам и точному числу строк.

Пока код только подготовлен. Статус будущего успешного замера
`passed_for_supplied_dataset` означает прохождение предоставленной панели на
конкретном CPU; это не доказательство времени на неизвестном hidden test и другом
железе. Наличие checker не подтверждает ни запуск, ни прохождение лимитов.

## Как распределить следующую работу

1. Использовать уже существующие draft PR; у каждого определить владельца,
   недостающий код/inputs, точный primary contrast и 30-ячеечный контракт.
2. Принимающая сторона адаптирует runnable code к реальным B300 и модульному
   trainer, подтверждает strict FP32, causality/reset и реальный memory profile.
3. В каждой выбранной ячейке сразу выполнить полный fitting-набор и полный
   заявленный горизонт. Отдельно учитывать teacher/cache/student стоимость.
4. Собрать все cells, failures, predictions, grades, progress и ancestry;
   выполнить статистический протокол и независимую проверку результата.
5. Следующий масштаб/механизм выбирать по наблюдаемому полезному эффекту,
   неопределённости и роли teacher/student/link. Не заменять исследование
   инфраструктурой очереди или обещанием автоматического pay-to-win.
6. К упаковке и precision reduction перейти при выборе конкретного submission
   endpoint. Локальные модели, тесты, GPU fits и публикации этот MD не запускает.

## Минимальная карточка результата

```text
hypothesis / experiment / candidate / control
source revision + input/checkpoint/cache hashes
all fitting component IDs + outer evaluation IDs for each cell
teacher / initializer / normalization ancestry
seed + architecture + precision + loss + optimizer + update budget
fixed checkpoint policy
strict FP32 / TF32-off proof + fixed fold/seed matrix
official WP, per-target WP, primary D + seed/component intervals + multiplicity
fit / predict / score / total seconds; completed optimizer updates
GPU memory peak; scope of RAM measurement
prediction / receipt / log paths
decision: continue / increase scale / test interaction / defer exact recipe
```

## Источники и границы документа

- [Текущий workflow и статистический контракт](https://github.com/VladKochetov007/WunderFunder/blob/0edd432271991250f4718f9b68f33340eff46214/research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md).
- [main/AGENTS.md с гайдом](https://github.com/VladKochetov007/WunderFunder/blob/0edd432271991250f4718f9b68f33340eff46214/AGENTS.md),
  [открытые draft PR](https://github.com/VladKochetov007/WunderFunder/pulls?q=is%3Apr+is%3Aopen+base%3Aresearch-team).
- [Прежний протокол 10%/30%](../../../specs/plans/2026-10-08-small-data-screening.md)
  остаётся историческим источником; в данной командной волне это не основной путь.
- [Реализация fit/predict/score и GRU-пакета](../operations/FAST_GPU_SCREENING_PIPELINE_20261017_R1.md).
- [Четыре GRU-сравнения](../../campaign_2026-10-17/fast_gpu_queue_r1/README.md),
  [четыре архитектурных fit и CPU gate](../../campaign_2026-10-17/architecture_gpu_wave_r1/README.md).
- [Все 203 message-блока и командный топ](../reviews/TEAM_TOP_AND_NEXT_TESTS_20261017_R1.md).
- [Transformer/Mamba и новые сообщения команды](TRANSFORMER_GPU_WAVE_20261017_R1.md).
- [Гипотезы](../../hypotheses.jsonl), [эксперименты](../../experiments.jsonl).

Это консолидация уже обсуждённых направлений. Запись идеи не означает её
реализацию, допуск к запуску, измеренный выигрыш или загрузку сабмита.
