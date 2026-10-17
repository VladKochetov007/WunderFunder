---
kind: proposal
status: proposed_not_executable_no_gpu_runs
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: [H-CAUSAL-TRANSFORMER-20261017, H-CONTEXT-TRANSFORMER-20261017, H-SELECTIVE-SSM-20261017]
experiment_ids: []
evidence_paths:
  - src/wnn/block_memory_attention.py
  - research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_FOLD0_REFERENCE_BLEND_20261008_R1.json
  - research/campaign_2026-10-13/team_models_r1/INVENTORY.json
---

# GPU-волна: проверить Transformer рядом с GRU

Implementation follow-up: the user subsequently requested all four arms. The
[implementation and GPU launch packet](../../campaign_2026-10-17/architecture_gpu_wave_r1/README.md)
now supplies the new backends and target-host commands; runtime validation and
fits remain pending. The proposal below is preserved as the original rationale,
including its then-current implementation gaps.

## Командное дополнение 17 октября, 13:15–13:35 UTC+3

Источник — сообщения Vlad Koch и Ivan Vorfolomeev, переданные пользователем в
этом чате: принадлежность последовательности окружности с допуском, сравнение
GRU/Transformer как учителей, Chronos/TimesFM как pretrained teachers. Это новые
гипотезы, не результаты и не разрешение запускать расчёты локально. Подготовленная
четвёрка A–D остаётся без изменений; следующие сравнения используют её артефакты.

**Первый следующий эксперимент — один и тот же GRU-студент с разными учителями.**
Три условия: labels-only, KD от GRU A, KD от заранее выбранного Transformer B.
Фиксируем student initialization, seed, fit groups, нормализацию, порядок строк,
loss/KD coefficient, optimizer и число updates; оцениваем один и тот же epoch.
Выбор B заранее отделяет смену семейства от подбора победителя B/C на evaluation.
Смесь GRU+Transformer — последующий interaction arm, если отдельные teachers
покажут переносимый сигнал. Главное наблюдение — качество **студента** после
одинакового обучения и после Q8, а не только score или размер teacher.

Teacher, его preprocessing и адаптация не должны обучаться на оценочных groups.
In-sample teacher predictions на student fitting groups допустимы как явно
обозначенный первый KD-протокол у обоих teachers; whole-component OOF targets
можно проверить отдельно, не добавляя K переобучений к первому сравнению.
Full-valid predictions из A–D достаточны для grading, но **недостаточны для KD**:
ещё нужны row-aligned predictions для fitting train+valid groups, с масками,
checkpoint/source hashes и ancestry. Текущий `--distill` принимает teacher cache,
а существующий `build_distill_cache.py` ориентирован на T3 — универсальный
экспорт этих двух teachers пока не реализован. Это proposal, не runnable KD queue.

**Окружность — отдельная короткая feature ablation.** Координаты, рисунок и единицы
epsilon в переданном сообщении не определены; соответствие анонимных столбцов
геометрии не угадываем. После уточнения сравнить тот же GRU без/с фичей. Если
центр и радиус оцениваются по последовательности, для строки t они используют
только её префикс до t−1; текущая точка даёт расстояние до окружности. Если
геометрия общая, её параметры обучаются только на fitting groups и замораживаются.
Помимо `abs(distance - radius) <= epsilon` полезно заранее проверить непрерывный
радиальный residual: он сохраняет информацию рядом с порогом. Нужны определённые
warm-up и degenerate-fit значения, reset между sequences и epsilon, выбранный на
fit данных. Полная последовательность, включая будущие точки, не может определять
текущую фичу. Время обновления геометрии тоже входит в one-CPU budget.

**Chronos/TimesFM — отдельная проверка переноса pretrained representations.**
Официальная [карточка Chronos-2](https://huggingface.co/amazon/chronos-2), проверенная
17 октября 2026, описывает 120M параметров, multivariate/covariate forecasting,
контекст до 8,192 и Apache-2.0. Это разумный первый кандидат для GPU teacher:
одни лишь 4-битные dense weights дают примерно 60 MB до служебных данных, поэтому
для ZIP20MB основной исследуемый путь — дистилляция, а не обещание прямого сабмита.
Это расчёт по числу параметров, не замер архивного размера или CPU скорости.

[Официальный TimesFM](https://github.com/google-research/timesfm) теперь описывает
3.0 с native multivariate и past/future covariates. Код и веса до 2.5 отмечены
Apache-2.0, а [веса 3.0](https://huggingface.co/google/timesfm-3.0-pytorch) имеют
отдельную non-commercial лицензию. Допустимость конкретных внешних weights и их
использования в этой competition здесь не установлена; версия и источник должны
быть определены до включения в рецепт.

Оба forecasting API не являются готовой заменой нашей регрессии `X[<=t] -> y[t]`.
Обычный пример с настоящей историей target нельзя выдавать за deployable teacher:
в callback доступны признаки, но не прошлые true labels. Первая техническая
проверка должна использовать только доступные на момент t признаки: например,
frozen representations/прогноз наблюдаемых feature channels плюс обученный на fit
groups readout к двум targets. Это **предлагаемая адаптация**, не подтверждённый
API или работающий backend. Нормализация, патчи и извлечение representations также
должны быть причинными; один bidirectional embedding всей 20k sequence протечёт
в будущее. Не переносить рекламные benchmark scores на наши данные. Сначала
проверить row-time contract и signal на ограниченном GPU-пилоте, потом выделять
полный teacher budget.

Связанные новые гипотезы: H-TEACHER-FAMILY-KD-20261017,
H-CAUSAL-CIRCLE-FEATURE-20261017, H-PRETRAINED-TS-TEACHER-20261017.
Ни один новый KD/geometry/pretrained fit не запущен.

## Question or claim

Последнее уточнение пользователя: доступно много GPU; на этом Mac не запускать тесты; рассмотреть переход с GRU на Transformer. Новые локальные проверки и модельные расчёты остановлены. Эта записка меняет рекомендуемый научный приоритет, а не запускает оборудование. Адреса GPU, их число, память и допустимый бюджет ещё не установлены. Старая четвёрка S/W/F/H512 остаётся отдельным резервным пакетом; она не должна задерживать архитектурную ветку.

**Рекомендация: начать отдельное сравнение архитектур, сохранив GRU как контроль.** GPU позволяют проверить более ёмкую модель и более длинный контекст; из одного наличия GPU не следует выигрыш в качестве. Практический выход — либо самостоятельный компактный Transformer, либо новый teacher и его вклад в student/фиксированный ансамбль.

## Mechanism and method

### Четыре параллельных направления

Это предлагаемые карточки, **не исполняемые worker specs**. Новый sliding-window Transformer и Mamba-2 пока не подключены к trainer/exporter. Ни одного нового GPU fit не было.

| Слот | Предлагаемая модель | Что различаем | Состояние реализации |
|---|---|---|---|
| A | GRU H256×2 с нуля | Общий контроль для новой архитектурной волны | Существующий trainer; отдельный согласованный рецепт нужен до запуска |
| B | Causal Transformer: width 256, 4 слоя, 8 heads, FFN×4, окно 256 | Может ли attention извлечь сигнал, который теряет компактное recurrent state? | Нужен sliding-window backend; существующий H218 — другой, block-memory/GRU hybrid |
| C | Та же модель, окно 2048; остальные настройки как B | Даёт ли длинный контекст выигрыш, оправдывающий стоимость? | Тот же новый backend; это контекстный контраст B→C, не одновременная смена ширины/targets |
| D | Mamba-2, начальная width 256 / 4 слоя; размеры state зафиксировать перед запуском | Альтернативная сжимаемая память длинной последовательности | Новый backend, зависимости, exporter и state/reset adapter; готового runnable arm нет |

Размеры — стартовые предложения, не замеренные оптимумы. Число параметров A–D не объявляется одинаковым. Более крупный Transformer width512/8 слоёв — отдельный следующий этап после фактических learning curves; его нельзя скрыто подменить вместо C.

### Как получить содержательное сравнение

1. Один frozen split и одинаковые fit components, входные признаки, нормализация и scored mask. Начать с scratch labels-only у всех A–D; сравнение scratch Transformer с warm KD-GRU не изолирует архитектуру. Старые C и лучший командный пакет остаются внешними ориентирующими endpoints. Нормализация обучается только на общих fitting groups.
2. Сначала один общий seed для проверки направления. Одинаковый порядок последовательностей и одинаковое число обучающих строк; записывать attempted/completed optimizer updates, effective batch, режим precision, время GPU и загрузки. Сравнить качество как при одинаковом количестве данных, так и на фиксированных временных срезах. Один общий LR не считается одинаково качественной настройкой: затем дать каждой семье одинаковое число заранее ограниченных LR/warmup вариантов, а не настроенному GRU противопоставлять единственный произвольный Transformer.
3. Новый GPU сначала выполняет минимальные проверки causal mask, state/reset, streaming-versus-batch и конечности градиентов; затем короткий профиль. Эти проверки перенесены на GPU-хост по просьбе пользователя. Перенос не означает, что неподтверждённый код уже проверен.
4. Не сбрасывать память на случайных внутренних границах последовательности. При обучении окнами нужен левый контекст без двойного учёта loss; допустимые future keys исключены на каждом слое. Patch/conv tokenizer должен использовать только уже пришедшие строки. Нормализация всего окна с будущими строками также нарушает причинность.
5. При большом числе GPU сначала независимые arms/настройки/seed-fold пары; DDP подключать, если одна модель или полезный batch требуют нескольких устройств. Данные и кэши подготовить один раз на хост, не размножать распаковку на каждый GPU. Один занятый GPU не доказывает высокий throughput: считать законченные и оценённые сравнения на час с учётом I/O.
6. Выгружать одинаковые full-fold predictions, оценивать official Global WP и заранее фиксированную смесь с контрольным GRU. Положительное дополнение смеси полезно даже при более низком standalone score. Затем другой seed и source-qualified fold; существующие folds уже просмотрены. Интервал, пересекающий ноль, не закрывает архитектуру автоматически.
7. Если teacher приносит сигнал, сформировать его targets только из допустимого fold-qualified обучения и сравнить два одинаковых student: прежние targets versus новая смесь teachers. Отдельно проверить конечный CPU endpoint. Не использовать обученные на validation final-team weights как источник якобы чистого CV.

### Почему это ещё имеет смысл после прошлых attention-опытов

H218 уже дал 0.66240365 против 0.68094121 у GRU-ориентира на exposed F0. Фиксированная добавка attention 10% ухудшила WP на 0.00057181. Фактически обучался width128 / 2 слоя / 4 heads / block256, четыре эпохи, один seed; зафиксированы 5,338 секунд и около 0.793 GB peak VRAM на прежнем оборудовании. Это реальное отрицательное наблюдение для того endpoint. Но receipt прямо отмечает: scratch block-memory сравнивался с GRU с since-teacher/propensity ancestry; это не matched architecture/compute ablation. В коде H218 каждая attention layer сжимает завершённый блок в GRUCell memory, а не хранит длинный sliding KV context. Нельзя назвать этот результат доказательством бесполезности всех Transformers или повторять тот же H218 без нового отличия.

H264 не добавил новых fits: фиксированная смесь старого attention с TCN также проиграла. В H280 старому attention достался нулевой вес в component-CV на уже просмотренной панели. E467/E469 — ещё более узкий опыт: width32, 8-мерная recurrent kernel-memory, 24 обучающие последовательности и 16 exposed evaluation components; это не полноценный Transformer. Их исходные результаты сохранены, но не служат основанием закрыть новую архитектурную ветку.

Лучший сохранённый public команды `pair_ftt2` = 0.6682 состоит из четырёх GRU H256×2. FTT в имени файла не доказывает наличие Transformer teacher. Поздний исходный FTT-рецепт и его ancestry пока не найдены. Это ограничение сведений о команде, не ограничение выбора нового направления.

## Evidence and interpretation

- [Официальные данные](https://wundernn.io/connectome/docs/data_overview): последовательности по 20,000 строк, 112 признаков, прогноз двух targets по текущей и прошлым строкам. Именно потоковая причинная постановка определяет маски и сброс state.
- [Официальный submission guide](https://wundernn.io/connectome/docs/submission_guide), проверен онлайн 17 октября: 1 vCPU, 16 GB RAM, 60 минут, ZIP20 MB, без GPU. Это финальный endpoint, не размер разрешённого исследовательского teacher.
- [FlashAttention, Dao et al.](https://arxiv.org/abs/2205.14135): IO-aware вычисление точного attention сокращает трафик/память на GPU. Оно не превращает dense attention в линейный алгоритм и само не решает стоимость однопоточного CPU callback. Конкретный fused kernel зависит от GPU, dtype и маски; его выбор нужно наблюдать на целевом хосте.
- [Mamba-2 / SSD, Dao & Gu](https://arxiv.org/abs/2405.21060): основание проверить selective state-space память и эффективное GPU-обучение. Результаты авторов в language modeling не являются результатом на наших признаках или гарантией быстрого CPU runtime.
- [HFformer, Barez et al.](https://arxiv.org/abs/2302.13850): первичная работа о Transformer для Bitcoin-USDT log-return forecasting. Это смежная задача с другой метрикой и экспериментом; не установлено, что именно эта статья была отсутствующим вложением HFTTransformers в `messages.html`.

## Limits and validity

Здесь нет результатов нового архитектурного сравнения, измеренного ускорения, стоимости часа или public-прогноза. GPU topology и адреса не проверены. Отсутствие конечного CPU-экспорта не блокирует изучение teacher, но блокирует заявление о готовом сабмите. Новые зависимости и streaming backend требуют проверки на GPU-хосте; локальные тесты после уточнения пользователя не запускались.

## Decision and next step

Новая архитектурная волна имеет приоритет перед последовательным перебором мелких GRU-доработок. Подготовленная S/W/F/H512 очередь сохраняется как отдельный пакет для свободных исполнителей. Следующий шаг архитектурной ветки — подключить causal-window backend и общий scratch control к существующим trainer/predict/grade, затем проверить и обучить на предоставленном GPU-хосте. Это конкретный implementation gap; не маркировать эту записку как готовую GPU-очередь.

## Source records

- [H218 paired grade](../../campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_FOLD0_REFERENCE_BLEND_20261008_R1.json), [реализация H218](../../../src/wnn/block_memory_attention.py).
- [H264 fixed-mixture grade](../../campaign_2026-10-06/bandit_continuation/interaction_attention_tcn_H264/H264_FIXED_EQUAL_MIXTURE_RESULT_20261008_R1.json), [H280 conditional architecture analysis](../../campaign_2026-10-06/bandit_continuation/blend_lab_H274/BLEND_LAB_RESULT_20261008.md), [малые sequence-components: три seed](../../campaign_2026-10-05/sequence_components/THREE_SEED_REPORT.md). Отсутствовавшие в sparse checkout записи прочитаны из Git; это исторические результаты, не новые вычисления.
- [Командные методы и источники](../../campaign_2026-10-13/team_models_r1/TEAM_METHODS_AND_EVIDENCE_R1.md), [топ сабмит и гипотезы](../reviews/TEAM_TOP_AND_NEXT_TESTS_20261017_R1.md).
- [Переносимый пакет существующих GRU-рецептов](../../campaign_2026-10-17/fast_gpu_queue_r1/README.md).
