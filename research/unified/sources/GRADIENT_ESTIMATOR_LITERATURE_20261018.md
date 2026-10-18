---
kind: review
status: complete
hypothesis_ids: [H-RR17-19, H-055]
experiment_ids: [E-ADAPTIVE-TBPTT-20261018, E-CURRENT-MOMENT-WP-20261018]
evidence_paths: [research/unified/ideas/GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md]
---

# Литература о градиентах: что переносим в WunderFunder

## Вопрос

Можно ли улучшить стоимость обучения GRU и соответствие loss официальной метрике, сохранив причинный forward и размер CPU-модели? Это литературные основания для двух **продолжений существующих гипотез**, а не доказательства прироста на Challenge2. Исходное дерево проекта: `1d57ed815c18eb278bcdd0207369b200970b28fb`.

## Первичные источники и глубина чтения

| Источник | Что проверено | Что следует и чего не следует |
|---|---|---|
| Aicher, Foti, Fox, [Adaptively Truncating Backpropagation Through Time to Control Gradient Bias](https://proceedings.mlr.press/v115/aicher20a/aicher20a.pdf), UAI 2019 / PMLR 2020; [код авторов](https://github.com/aicherc/adaptive_tbptt/blob/master/tbptt/adaptive_truncation.py) | Разделы 3–5: предположение о геометрическом хвосте, оценка bias, выбор K; scout прочитал реализацию оценки decay. | Предлагается выбирать длину обратного прохода по допустимой оценке смещения. Это **biased adaptive TBPTT**, не unbiased ARTBP. Теория SGD при предпосылках не гарантирует качество GRU с AdamW, clipping и EMA. Норма hidden-cotangent может переоценивать вклад отдельных координат; эмпирическая оценка decay не является сертифицированной границей. |
| Gao et al., [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/html/2101.06983v2), RepL4NLP 2021; [Gradient Cache](https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py) | Разделы 3.2–3.3: graphless forward, производные по представлениям, повтор подбатчей; scout проверил replay RNG в авторском коде. | Chain-rule replay позволяет накапливать градиент общей функции выходов без хранения всех activation graphs. Перенос на pooled WP и stateful GRU — **наша гипотеза**. Статья про contrastive retrieval; она не реализует наши TBPTT/reset/mask и не доказывает локальный speedup. |
| Wang, Fang, Liu, [Stochastic Compositional Gradient Descent](https://arxiv.org/abs/1411.3803) | Только abstract: композиция ожиданий, вспомогательное отслеживание внутренних величин, двухмасштабная схема. | Полезный контекст: нелинейная функция моментов не сводится к среднему chunk-loss. Не основание повторять EMA pooled moments и не гарантия для clipped WP. Полный метод этой работы здесь не воспроизводили. |

Численные выигрыши из внешних работ не переносим в прогноз нашего score или времени. Внешние данные и веса не импортировали.

## Дополнительный раунд: μP и перенос настройки между ширинами

Проверены [Tensor Programs V, §§4, 6.1 и D.1](https://arxiv.org/html/2203.03466v2) и [официальный microsoft/mup](https://github.com/microsoft/mup). μP согласованно меняет параметризацию, инициализацию и learning rates разных типов параметров; это не просто перенос одного LR. Показанные авторами модели — Transformer/ResNet и MLP-примеры; GRU-специфичного подтверждения в прочитанном материале нет. Перенос по глубине имеет отдельные оговорки, в том числе pre-LN против post-LN.

Наша гипотеза переноса: width/depth результат может зависеть от несогласованных масштабов обновления. Но fused GRU gate matrices и recurrent/readout weights требуют явной разметки base shapes и coordinate checks до применения библиотеки. **Не включать μP в текущую CP30-матрицу:** смешаются архитектура и параметризация. Возможное отдельное продолжение — 2×2 standard/μP × wide/deep при одинаковом tuning budget, после проверки механизма на нескольких ширинах. Сейчас это отложенный кандидат, без кода, fit или зарегистрированного запуска. Muon и μP — разные методы. Это решение не отвергает μP в causal Transformer.

## Дедупликация и отрицательная история

- `H-RR17-19` уже предлагает [ARTBP](https://arxiv.org/abs/1705.08209). Новый вариант — адаптивный **смещённый** estimator, а не переименование ARTBP. [S14](../../campaign_2026-10-13/strategy_execution_r1/gradient_credit_f0_s3_r1/README.md) содержит профили фиксированных detach=500/2000, но не завершённое сравнение качества. Эти профили не засчитываются в новую матрицу.
- `H-018 / E-023`: per-chunk correlation провалилась на постоянных целях. `H-055 / E-291`: decayed pooled moments `.995` уже дали отрицательное сравнение; старый `queued` в H-055 не означает, что опыт не проводился. [Разбор с интервалами и code-path ограничениями](../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md) сохраняет этот результат. Новый вариант использует текущие параметры и все строки заранее выбранного супербатча, без отложенной памяти моментов.
- Новые team PR по Kalman, next-input auxiliary, meta-TTA и group masking решают другие вопросы. Не дублируем их карточки и не считаем их публикацию результатом обучения; [сводка проверенных изменений](../reviews/REPOSITORY_DELTA_20261018.md).

## Решение

Сохранить две узкие проверки в [общей карточке](../ideas/GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md). Сначала требуется реализация и проверка математического контракта в разрешённой среде; затем ресурсный профиль и парное сравнение. В этой задаче не запускались тесты, обучение или model execution. Не меняем действующую очередность ranked25.
