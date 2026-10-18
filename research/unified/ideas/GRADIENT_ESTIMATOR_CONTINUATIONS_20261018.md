---
kind: idea
status: proposed
hypothesis_ids: [H-RR17-19, H-055]
experiment_ids: [E-ADAPTIVE-TBPTT-20261018, E-CURRENT-MOMENT-WP-20261018]
evidence_paths: [research/unified/sources/GRADIENT_ESTIMATOR_LITERATURE_20261018.md]
---

# Два продолжения гипотез о градиентах GRU

## Что предлагаем

| Вариант | Какой недостаток проверяет | Контроль | Решение, которое может изменить результат |
|---|---|---|---|
| **Adaptive TBPTT**, продолжение H-RR17-19 | Фиксированный длинный backward может тратить вычисления на почти исчезнувший градиент. | Fixed detach=2000 против адаптивного выбора 500/1000/2000 при одинаковом forward и optimizer clock. | Если время полного fit, включая probes, уменьшится минимум на 15% при допустимой потере WP не более 0.0005, можно тратить бюджет на больше независимых fits или большего учителя. |
| **Current-moment WP replay**, продолжение H-055 | Chunk-wise correlation не видит общие смещения/масштабы; detached EMA-моменты устаревают и не передают градиент прошлым строкам. | Label-only WMSE против общей clipped WP по тем же 64 полным последовательностям на один optimizer step. | Средний прирост WP не менее 0.0005 при времени не более 2× контроля оправдает дальнейшую проверку loss; иначе оценим, полезен ли механизм относительно его стоимости. |

Пороги — предложенные критерии практической ценности, а не измеренные эффекты или универсальные scaling laws. Оба варианта меняют обучение учителя/студента, но не архитектуру forward; заявлять перенос улучшения учителя в студента можно только после отдельного matched KD сравнения. Текущие ranked25 и работающие ячейки не переназначаются. Здесь подготовлены документы, **не готовые GPU-запуски**.

## Почему это отличается от уже проверенного

**Adaptive TBPTT.** [Aicher et al.](../sources/GRADIENT_ESTIMATOR_LITERATURE_20261018.md) оценивают затухание hidden-cotangent по лагам и выбирают минимальную глубину, удовлетворяющую оценке bias budget. В нашем варианте microchunk=500 и update_rows=2000 сохраняются; меняется только stop-gradient внутри update window. Состояние forward переносится по всей последовательности. Нельзя подменить это изменением optimizer cadence или переносить авторский BPTT(2K,K) loop целиком. Нужна явная спецификация границ для всех rowwise losses: K — интервал detach, не обещание одинаковой глубины для каждой строки. Controller ещё не реализован.

Это ветка внутри H-RR17-19, но не ARTBP: unbiasedness и компенсация здесь не заявлены. Отсутствие устойчивого decay, недостижимый допуск, некорректный знаменатель относительной ошибки, почти постоянный K=2000 или слишком дорогие probes ослабят объяснение. На подозрительном хвосте выбирается Kmax и пишется причина; это fallback, **не подтверждение допустимого bias**. Синтетический oracle должен сопоставлять оценку с точным градиентом на контролируемых графах и обнаруживать нарушения предпосылок. Для реального AdamW/EMA процесса формальной гарантии нет.

**Current-moment WP replay.** Общие моменты вычисляются при одном frozen θ на выбранных 64 последовательностях. После graphless forward сохраняются производные loss по выходам, RNG и detached states на TBPTT-границах; replay восстанавливает те же выходы и накапливает gradient до единственного update. Нет EMA-моментов, обновления весов между проходами или best-chunk selection. Градиент точен только для определённого супербатча и графа с заданными detach, с учётом численной точности; это **не полный train-fold gradient и не full BPTT**. Текущий общий macrostep также используется WMSE-контролем: сравнение не объясняет, лучше ли macrostep обычного trainer schedule.

Для одной головы: z=clip(y,±2), q=clip(p,±2), w=|z|; μ — взвешенные средние, Syy и Spp — **ненормированные** суммы w·(z−μy)² и w·(q−μp)². Тогда при ненулевых дисперсиях производная −r по p равна

`clip'(p) * w * [r*(q-μp)/Spp - (z-μy)/sqrt(Syy*Spp)]`.

Для среднего двух голов множитель 1/2. Если использовать дисперсии, нормированные на W, нужен дополнительный 1/W. При W<1e-8 или любой взвешенной стандартной девиации ≤1e-8 вклад определяется как нулевой по контракту scorer; число таких случаев обязательно логируется. Вне clipping-диапазона градиент обнуляется, а на ±2 нужна фиксированная subgradient convention. Gate сравнивает replay с прямым графом с **теми же detach**, отдельно проверяет маски, RNG, reset, clipping и почти постоянные цели. Gate здесь не запускался.

WMSE-контроль использует те же z=clip(y,±2), w=|z| и fitting masks: среднее двух `sum(w*(raw_p-z)^2)/W`; при W<1e-8 вклад нулевой. Propensity/class/age/episode/auxiliary/importance веса и target shifts отключены в обоих arms. Разница включает clipping предсказания внутри Pearson; это сравнение loss при фиксированном macrostep, а не изоляция replay как причины улучшения.

Отрицательная история обязательна: E-023 refuted H-018; E-291 отклонил прежний H-055 pooled-memory вариант (two-stage delta −0.001521, conditional 95% CI [−0.002868,−0.000161]). [Подробный разбор](EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md) включает также неубедительный head-only результат. Новый вариант не отменяет их и не повторяет clipping-only reopening.

## Общий протокол будущего сравнения

- Полные fitting memberships, два arms × 30 парных ячеек = **60 fits на вариант**. F0/F1: seeds1001–1008; F2/F3: seeds1001–1007. Существующие короткие profiles/fit-результаты не входят в эту матрицу. На каждом fold оба arms используют одинаковые исходные веса, порядок данных и параметры, кроме заявленного вмешательства. Для cold run нет учителя; любой warm parent потребует отдельной source-clean registry и новой фиксации recipe.
- FP32 weights/states/activations/gradients/optimizer, autocast/GradScaler/TF32 off; float32 matmul precision highest. Код scorer `global_wp_clip2_v1` и scored mask остаются прежними. Никакой quantization в этих опытах.
- Фиксированный последний checkpoint после шести полных эпох; диагностические промежуточные grades не выбирают endpoint. Shared seed используется как crossed factor, а не как 30 независимых holdouts. Полные последовательности и связанные компоненты не пересекают fit/evaluation границы. Нормализация только на fitting membership; данные будущего и validation labels не попадают во входы или gradients. Обе головы и оценочная маска не меняются.
- Основная величина `D = mean_f(mean_s(WP(candidate_f,s)-WP(control_f,s)))`, равные веса 1/4 для folds, n=8/8/7/7. Сначала score каждого endpoint по всем оцениваемым строкам; усреднение predictions по seeds — другой, вторичный endpoint.
- Как в [методике](../methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md), 10 000 bootstrap draws восьми seed IDs с одинаковыми multiplicities во всех folds; в fold нормируем на имеющиеся seeds и повторяем draw при нулевой массе. Отдельно — component bootstrap при фиксированных fits, одни multiplicities на оба arms и все seeds. RNG seed=20261018. Приводим marginal 95% интервалы, все fold means и failed cells. Оба интервала условны на уже виденные folds; не покрывают selection, новые режимы и public transfer.
- Для двух заранее выбранных primary contrasts дополнительно даём консервативные 97.5% marginal интервалы (Bonferroni family of two). Для Adaptive TBPTT quality gate: нижняя граница seed- и component-интервалов выше −0.0005, измеренное среднее парное сокращение полного времени ≥15% и отсутствие систематического провала по folds. Для WP replay: D≥0.0005 и нижние границы выше нуля при стоимости ≤2× контроля. Из-за восьми seed IDs это приближённая оценка, не обещание точного покрытия или мощности. Не заявляем calibrated statistical significance только на основании числа 30. CI, пересекающий ноль/порог, означает неопределённость, а не закрытие всей семьи.
- Все измерения обеих сторон парной ячейки выполняются на одном provider/runtime, с явным владением ячейкой и учётом расходов. Дополнительный диагностический проход не выдаётся за matched extra training. Неполные пары не заменяются молча; публикуется coverage и причина каждого исключения.

## Пакеты и готовность

| Experiment | Ветка для draft PR | Что отсутствует |
|---|---|---|
| E-ADAPTIVE-TBPTT-20261018 | [PR #22](https://github.com/VladKochetov007/WunderFunder/pull/22), `experiment/adaptive-tbptt-bias-budget` | Controller/probes, checkpoint/resume состояния schedule, доказательство сохранения update cadence, чистая registry CP30, профиль на выбранной GPU. |
| E-CURRENT-MOMENT-WP-20261018 | [PR #23](https://github.com/VladKochetov007/WunderFunder/pull/23), `experiment/current-moment-wp-replay` | Two-pass objective/replay adapter, gradient oracle, равный macrostep WMSE, чистая registry CP30, фактический профиль времени/памяти. |

Перед fit требуется отдельная фиксация реализационного source SHA, data/split/initializer/config hashes, optimizer settings, env и путей. Данные контроллера и replay входят в resume contract. Не добавлять логику в legacy trainer: использовать [модульную карту](../../../src/wnn/training/README.md). `scripts/experiment_draft_plan.py` только печатает матрицу, не обучает. В этой задаче нет тестов, обучения, model imports или профилей; scope текущей просьбы — анализ и публикация гипотез.

## Источники и след

- [Проверенные статьи и пересечения](../sources/GRADIENT_ESTIMATOR_LITERATURE_20261018.md).
- [Что добавилось в репозитории](../reviews/REPOSITORY_DELTA_20261018.md), source `1d57ed815c18eb278bcdd0207369b200970b28fb`.
- [H ledger](../../hypotheses.jsonl) хранит новые continuation records; исходные H-RR17-19/H-055 и отрицательные E-023/E-291 не переписаны. [E ledger](../../experiments.jsonl) регистрирует два unrun proposals. Draft PR #22 и #23 опубликованы к research-team; head SHAs и статус draft проверены через GitHub API. Публикация не является fit admission.
