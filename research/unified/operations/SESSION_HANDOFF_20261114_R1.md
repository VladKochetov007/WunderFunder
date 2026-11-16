---
kind: result
status: complete
created_utc: 2026-11-14T21:47:23Z
updated_utc: 2026-11-14T21:47:23Z
hypothesis_ids: [H-Q14-TEACHER-POOL-R3]
experiment_ids: [E-Q14-TWO-TEACHER-R1, E-Q14-STRICT-LOCAL-PAIR-R2]
---

# Передача текущего состояния Wunder — 14 ноября 2026

Самодостаточная точка входа для следующей сессии. Последняя просьба пользователя — сохранить состояние; диагностика Kaggle сейчас не приоритет. Ничего не перезапускать только на основании этого файла.

## Главное

- Цель: максимизировать качество сабмитов Wunder Connectome Challenge 2, пробовать новые механизмы и использовать командные GitHub-артефакты. Challenge 3 исключён.
- **Локальный GPU оставлен PHCapital по явному указанию пользователя.** Маркер `research/campaign_2026-11-14/quality_r2/LOCAL_GPU_HOLD` остаётся; не удалять и не возобновлять локальные GPU-задачи автоматически. Разрешены ограниченные CPU/read-only проверки. Даже отсутствие процесса PHCapital не отменяет указание.
- При проверке 2026-11-14 21:40 UTC работающих `wnn-*` systemd-сервисов нет. Все делегированные агенты закрыты.
- **Kaggle уже ERROR, а не QUEUED.** Это новое наблюдение при подготовке передачи. Обученных моделей и оценок от этой панели нет. Подробности ниже; исправление отложено по просьбе пользователя.
- Лучший новый диагностический результат: фиксированная смесь 50% FBC +50% TLead-s5 даёт **0.692260448** против **0.691420480** текущего одиночного учителя, **+0.000839968**, обе цели улучшились. Это многократно просмотренный valid; улучшение студента и public не установлены.
- Новый более сильный ZIP в продолжении не получен. Три ранее проверенных ZIP готовы; два из них пользователь уже отправил.
- Никаких платных ресурсов, пополнений, новых B300-запусков или автоматической отправки в соревнование.

## Рабочие каталоги, ветки, коммиты

| Назначение | Путь / состояние |
| --- | --- |
| Основной репозиторий | `/home/vlad/development/wunder/wnn_connectome_starterpack` |
| Основная ветка | `experiment/next-cycle-recovery-20261021` |
| Последний опубликованный основной коммит | `9b619f5a877e8a9bc249166605f29972b35d742e` |
| Общий draft PR98 | https://github.com/VladKochetov007/WunderFunder/pull/98 |
| Worktree PR98 | `/home/vlad/development/wunder/research_tmp/wt_local1114` |
| Ветка PR98 | `experiment/local-gpu-kd-since-20261114` |
| Опубликованный HEAD PR98 | `3dc7830ceb54a75bdcf9f06a1378b94eb695450e` |
| Замороженный исполнявшийся strict source | `/home/vlad/development/wunder/research_tmp/q14_strict_source` |
| Его commit | `48776fba8af26b16b7e6813def7ccf571f56c1ac` |
| Данные, модели и кэши текущих опытов | `/home/vlad/development/wunder/research_tmp/chrono_root` |
| Командные архивы | `/home/vlad/development/wunder/research_tmp/q14_team_artifacts_r3` |
| Свежескачанные логи Kaggle | `/home/vlad/development/wunder/research_tmp/q14_handoff_20261114` |

Предыдущие результаты и код запушены. Этот файл и добавленные при передаче ссылки/уточнения статуса сохраняются локально; отдельного commit/push для передачи не делали. PR98 остаётся снимком предыдущей публикации и ещё говорит про очередь Kaggle.

До передачи в основной рабочей копии были две **посторонние удаления**, не трогать и не включать автоматически в коммит:

```text
cache/pred_clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1_cp5_full_valid_r1_valid.json
cache/pred_clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1_valid.json
```

Не копировать целиком основной `AGENTS.md`, `current-state.md`, unified README или ledgers в worktree PR98: там другая интеграционная история команды. Переносить только нужный пакет и точечные индексные записи. Замороженный source48776 не менять на месте.

## Сабмиты и public-обратная связь

Каталог готовых ZIP:
`/home/vlad/development/wunder/wnn_connectome_starterpack/submissions/local_20261114/`

| ZIP | Размер | Состояние |
| --- | ---: | --- |
| `final_kd50_s3s4_pair.zip` | 8,257,503 B | train+valid, два seed, веса .5/.5; пользователь отправил: **0.66510**, 25m12s, 641MB |
| `final_kd50_s3_single.zip` | 4,645,911 B | train+valid, seed3; проверенный быстрый fallback, отправка не сообщалась |
| `kd90_s3_single.zip` | 4,631,365 B | train-only, alpha .9, seed3; пользователь отправил: **0.66193**, 15m31s, 628MB |

SHA256 в том же порядке:

```text
b6634ddc776c4f3b4b8d0a5fbc23cc9c08c8ba319ec4c4cd9ddce44a06ff0482
ba871f492f8bd5d6d5214564a77bb6c79d84fbacc0066720bc4d96fc193d7890
a63ce606d7f7c040721950d553f74ed4499cec5ee359d0aa832260581a057541
```

Последний пользовательский снимок: **3/5 today, 112 total**. Это датированный снимок, не актуально проверенная квота.

| Portal ID | Вариант | Public |
| --- | --- | ---: |
| `2publky5` | KD90 s3 single | .66193 |
| `0b8njz9v` | final KD50 s3/s4 pair | .66510 |
| `1qn2bx0q` | старый train-only KD50 s3 | .66108 |
| `1y4rw9o7` | teammate `tabdata-final-fullmix-20261112-r2.zip` | .66543 |
| `1h7kp3oq` | teammate `tabdata-w18-final-20261103-r1.zip` | .66453 |
| `00kq8dw5` | tiedlead-s5 dynamic INT8 | .66246 |

Final KD50 pair ниже указанного fullmix на .00033. KD90 single выше прежнего KD50 single на .00085. Прирост final-pair к train-only-single смешивает эффект ансамбля и обучения на дополнительных данных. Исторический public .66832 существует в старых записях, но свежий live-best не проверен. Portal не сообщает hash: сопоставление с локальным ZIP сделано по имени.

Все три ZIP прошли CRC, native ABI Python3.11, конечность, точный A-B-A reset и parity на64 последовательностях. Tensor export соответствует fixed ep3. Полный native serving replay для этих трёх ZIP не повторялся. Final-data fit использовал12,479 последовательностей, причина пропуска одной группы не установлена; его valid не является holdout. Независимый аудит первоначальных ZIP не дал завершённого отчёта — не заявлять независимое одобрение.

[Подробный handoff и причины отбора](../../campaign_2026-11-14/local_r1/SUBMISSIONS_R1.md), [манифесты/проверки](../../campaign_2026-11-14/local_r1/submission_checks/README.md), [public receipts](../../campaign_2026-11-14/local_r1/PUBLIC_RECEIPTS_R2.json).

## Что уже исследовано

Первоначальная локальная очередь из18 fits закончена. На train-only/full-valid:

| Вариант | WP / результат |
| --- | ---: |
| legacy KD50 s3 | .690466215 |
| legacy KD50 s4 | .690481734 |
| фиксированная KD50 s3/s4 pair | .691422906 |
| legacy KD90 s3 | .691442250 |
| KD90 strict s3 `q14s_local_kd90_s3` | .691544385307459 |
| KD100 strict s3 `q14s_local_kd100_s3` | .6915276969552111 |
| mean weights ep2/ep3 KD90 | .6915238703355132 |
| mean weights ep2/ep3 KD100 | .691563618656678 |

KD поверх since32 улучшает два seed на +.003756/+.002299. Взаимодействие since×KD меняет знак: +.001587/−.001786; синергия не воспроизвелась. Pure teacher относительно strict KD90: −.00001669. Усреднение весов: −.00002051/+.00003592; полезного эффекта нет, seed5-confirmation не запускали.

Не повторять без новой причины: row-dependent alpha .691117; шесть эпох .688870; без auxiliary .688568; второй раунд KD .689705; H384 .690212; since8 .689727. Эксперименты с50%/75% данных сильно регрессировали (.621511/.662904) и не установили закон масштабирования. Старые TabM-mini GRU, NorMuon, column-view masks также screen_failed на проверенном рецепте; это не универсальное опровержение семейств.

Важная поправка precision: legacy Muon использовал **FP32 model kernels, но BF16 Newton–Schulz**. Старые scores/ZIP наблюдения сохраняются с этой маркировкой. Исправленный source48776 пробрасывает strict flag через обе обычные factory branches; optimizer-switch не входит в этот fix. Bug-specific CUDA checks,120-update profile и сохранённые FP32 optimizer states прошли. Старую автоматическую доставку/подтверждение не возобновлять.

[Строгая пара](../../campaign_2026-11-14/quality_r2/STRICT_LOCAL_PAIR_R2.json), [четыре полных endpoint](../../campaign_2026-11-14/quality_r2/trajectory_mean_r1/LOCAL_ENDPOINTS_R1.json), [precision fix](../../campaign_2026-11-14/quality_r2/precision_fix_r1/README.md).

Дополнительные завершённые проверки:

- Фиксированная teacher/student mean: .691898 против singles около .691442/.691420; универсальное отклонение ансамблей по residual correlation неверно.
- Оригинальный native TL3C: .692335900. Подмена .34 FBC-слота на KD90 дала proxy .692363415, всего +.000027515; component CI пересекает0. Actual native candidate replay остановлен из-за нагрева/малой ожидаемой отдачи, partial не оценивать и ZIP не считать улучшением.
- Original FBC native single: .691322088,1269.8s,555.9MB, reset passed. Это другая численная реализация, чем Torch teacher-score ниже.
- WMSE и WP gradients на одинаковых scored coordinates имеют cosine .997901. Низкий all-row cosine был эффектом маски, не доказательством плохого objective. Increment-KD fit не запускали.

[Blend audit](../../campaign_2026-11-14/quality_r2/blend_audit_r1/README.md), [target geometry](../../campaign_2026-11-14/quality_r2/target_geometry_r1/README.md), [native slot probe](../../campaign_2026-11-14/quality_r2/original_fbc_r1/README.md).

## Сильнейший подготовленный следующий опыт: качество учителя

Импортированы исходные PR49 FBC/TLead5/TLead6.16 ancestor runs проверены:10,325 train-only fitting sequences,0 valid. Модельные tensors сохранены без изменений. Переносимые metadata views имеют новые имена/hash; исходные архивы и provenance сохранены. Producer/local split отличаются timestamps/cache serialization; membership, exclusions, footer/size/index content совпадают. Независимый аудит подтвердил tensor storage и12 parent bindings.

| Полный Torch endpoint | WP |
| --- | ---: |
| `q14t_fbc_s5` | .6913549829253516 |
| `q14t_tlead_s5` | .6914152787485164 |
| текущий `tt29rep_fbc1k_s5` | .6914204798653247 |
| предобъявленная .5/.5 mixture первых двух | **.6922604480837262** |

Дельта +.0008399682184015; t0 +.0009050715, t1 +.0007748650. Оценены1,873 valid sequences/3,528,273 scored rows, неизменная official metric + reference assertion. Это одна дополнительная предобъявленная смесь на exposed valid,0 новых fits. Независимый review проверил код/receipt, но не пересчитывал массивы самостоятельно. Target digest проверяется в коде; grade JSON не содержит отдельного hash scorer/source и target cache — сохранять receipt вместе с исходником.

**Это не выполненный трёхучительский gate.** В grade `target_quality_trigger=false` потому, что выполнена отдельная two-teacher diagnostic; `diagnostic_delta_at_least_0005=true`. Студент не обучен. Уже существующий complete native TL3C локально немного выше — нового serving winner нет.

Замороженный следующий рецепт: .34 FBC +.33 TLead5 +.33 TLead6, alpha .9; четыре эпохи, fixed ep3, H256×2, since32, WMSE и прежние auxiliaries. Три новых student fits:

| Run | Seed | Target |
| --- | ---: | --- |
| `q14p_pool_s3` | 3 | fixed three-teacher pool |
| `q14p_control_s5` | 5 | текущий single FBC |
| `q14p_pool_s5` | 5 | fixed three-teacher pool |

Переиспользуется exact control `q14s_local_kd90_s3`;0 новых teacher fits. Полезный student effect: mean paired delta≥.0005 и оба seed positive, плюс сравнение полных фиксированных endpoint. Предобъявлены equal pool-student s3/s5 pair и equal pool-student-s3/legacy-KD50-s3 blend.

**Что отсутствует:** полный S6-valid export, все три train prediction caches, actual canonical train-target check и measured resource admission. Ничего из трёх student fits не запущено. Loader делит сумму teachers на3, поэтому правильный CLI:

```text
q14t_fbc_s5*1.02,q14t_tlead_s5*0.99,q14t_tlead_s6*0.99
```

Передача `.34,.33,.33` напрямую ошибочно уменьшит target втрое. Сначала смешать raw teachers, потом clip pooled target в[-2,2]; unchanged loss: .1 clipped-label +.9 clipped-pool.

S6 сначала OOM после появления PHCapital с~4.1GiB VRAM; batch4 recovery занимал~428MiB, затем остановлен пользователем. Сохранены `pred_q14t_tlead_s6_valid.npy.partial.oom_r1` и `.partial.user_gpu_hold_r1`; **не оценивать**. Локальные wrapper launches проверяют hold marker.

[Основной packet](../../campaign_2026-11-14/quality_r2/teacher_pool_r3/README.md), [результат](../../campaign_2026-11-14/quality_r2/teacher_pool_r3/TWO_TEACHER_RESULT_R1.md), [grade](../../campaign_2026-11-14/quality_r2/teacher_pool_r3/TWO_TEACHER_GRADE_R1.json), [рецепт](../../campaign_2026-11-14/quality_r2/teacher_pool_r3/RECIPE_R3.json), [allocation](../../campaign_2026-11-14/quality_r2/teacher_pool_r3/ALLOCATION_R3.json).

## Kaggle: свежий терминальный статус, ремонт не приоритет

Kernel: `vladkochetov07/wnn-q14-strict-q14s-20261114-r2-runner`, version1, ID144995259.
[Ссылка](https://www.kaggle.com/code/vladkochetov07/wnn-q14-strict-q14s-20261114-r2-runner).

Статус на21:40 UTC: `KernelWorkerStatus.ERROR`. Старые QUEUED записи и последний PR98 описывают прежний снимок. Фильтрованно скачаны JSON/log outputs, без новых запусков и без скачивания больших input/model payloads:

```text
/home/vlad/development/wunder/research_tmp/q14_handoff_20261114/
  wnn-q14-strict-q14s-20261114-r2-runner.log
  results/summary.json
  results/q14s_a50_s3/result.json
  results/q14s_a50_s3/worker.log
  results/q14s_a90_s3/failure.json
  results/q14s_a90_s3/worker.log
```

Оба стартовавших worker logs заканчиваются:

```text
ValueError: chronological parquet identity changed: /kaggle/working/experiment/datasets/train.parquet
```

Точка: frozen `src/wnn/chronological_splits.py`, `_verify_source`, строка130, при `load_lineage`, до обучения. a50-s3: returncode1, wall156.217s, peakRAM628,498,432B, peakGPU3MiB. a90-s3 имеет failure receipt при остановке общей панели; terminal wall в нём отсутствует. Четыре run не стартовали: a50-s4,a90-s4,boot8-s3,boot8-s4. Checkpoint/history/training-state отсутствуют. Это ошибка исполнения/идентичности источника, **не отрицательный научный результат** alpha или boot8.

Точная причина расхождения source identity не разобрана. В коде после portable verification сравнивается целый `_footer_identity(source)` с сохранённым dict; следующий инженерный разбор может проверить переносимость metadata, не отключая содержательные проверки. Не утверждать, что parquet повреждён или что это лишь timestamp, пока не проверено. Пользователь сказал: **«просто текущий статус опиши, там проверим, не приоритет»**. Повторный запуск не делали.

Изначальная панель: alpha .5/.9/.9+boot8 ×seeds3/4,6 fits, exact source48776, private1.58GB bundle. Kernel timeout13,500s, worker7200s,2 T4 lanes,8device-hour ceiling. Quota/billing после ошибки повторно не сверены; при продолжении обновить остаток и учесть startup/failure costs.

`collect_results.py` и `endpoint_grade_r2/grade_endpoints.py` готовы, но требуют всех6 complete outputs/identity checks. Сейчас нет данных для quality grading или выбора final fits. Не запускать graders на отсутствующих/частичных outputs.

[Пакет Kaggle](../../campaign_2026-11-14/quality_r2/kaggle_r2/README.md), [инструкция collection](../../campaign_2026-11-14/quality_r2/kaggle_r2/COLLECTION_README.md), [endpoint grade](../../campaign_2026-11-14/quality_r2/endpoint_grade_r2/README.md).

## Что есть у команды на GitHub

- **PR99**, Ivanhacker13, head `320ebd6ca38cf5cd16a708dd1a088c5f06f23e48`: structured causal perturbations для Transformer. Stale i1 bursts vs matched IID; old attention-key bursts vs IID с защитой current/recent32 keys.10 предложенных fits,2 seeds, Transformer24×768/context1024,2B300,BF16/BF16x9. Только implementation/design и static AST/Ruff:0 model tests/profiles/fits/weights/results. Не автоматически переносить BF16/B300 authority в root. На вечернем списке PR head тот же.
- **PR79 K16 stem:** paired gain около+.0013 на другом small-AdamW recipe (mean .682824435 vs .681489392). Это не доказательство превосходства над нашим .6915. Скачиваемых weight/prediction arrays на GitHub не было, только receipts/hashes.
- **PR95 since32:** paired +.002904 mean, уже в текущем recipe.
- **PR96 KD:** +.0018458 mean, но additive WMSE/objective отличается от нашего convex alpha; прямое сравнение alpha неверно.
- **PR97 head count:** подготовленный опыт, результатов не было.
- **PR49/52/54:** реальные выпущенные teacher/final artifacts. PR52 clean W18 .6914165, fixed W18/TLead pool .692377425. Малые GALAM/LSTM blend gains около .69250 — exposed, интервалы пересекают0; не новая уверенная победа.
- **PR54 final FBC:**43files/13roles проверены. Full-data fit12,480 groups; source/parents отличаются от clean reproduction, holdout score отсутствует. Core recipe/EMA endpoint совпадают, но historical strict IEEE cuDNN RNN не сертифицирован. Не переносить clean score на final teacher.
- PR91 bilinear, PR80 K64/width/dropout и PR60/61 варианты не дали поддержанного улучшения в проверенных рецептах.

[Обзор команды и provenance](../../campaign_2026-11-14/quality_r2/team_refresh_r3/README.md), [PR99 readout](../../campaign_2026-11-14/quality_r2/team_refresh_r3/PR99_READOUT_R1.md), [final teacher audit](../../campaign_2026-11-14/quality_r2/team_refresh_r3/FINAL_TEACHER_AUDIT_R1.md).

Командные архивы лежат в `research_tmp/q14_team_artifacts_r3`:

```text
tabdata-pr49-checkpoints-20261029-r1.zip
SHA256 02899081e9576bd86a0197cb3eae5bd1ccc8a4b1c26e1669cd1e10a62efd31fa

tabdata-w18-final-13-role-evidence-20261103-r1.zip
SHA256 2dc00d8a16f316775ddb979382312ee8a1d60b3c3555128726b8c4a44786c29e
```

Оригинальные/адаптированные checkpoint hashes,16 ancestor memberships, provenance и manifest equivalence: `quality_r2/teacher_pool_r3/IMPORT_R1.json`. Полные teacher predictions: `chrono_root/cache/pred_q14t_fbc_s5_valid.npy`, `pred_q14t_tlead_s5_valid.npy`; текущий teacher `pred_tt29rep_fbc1k_s5_valid.npy`. Модели: `chrono_root/artifacts/runs/<run>/`.

## Ограничения, бюджет и проверки

- Official metric: `global_wp_clip2_v1`, fixed mask, clip±2, weights по original clipped-label magnitude. Не менять scorer ради улучшения.
- Train-only comparison:10,325 train-source groups;282 linked train groups excluded; valid1,873 groups/3,528,273 rows. Всё valid уже exposed; новые seed/agents/разбиения не превращают его в fresh holdout.
- Final train+valid models разрешены, но fitted-valid scores не выбирать как heldout качество. Team assets разрешены с точной ancestry.
- Общая November14 allocation:8local-GPU-hours +8cloud-device-hours, максимум2final fits. Teacher continuation сохраняет хотя бы1final fit плюс packaging/parity reserve. Нельзя считать время ожидания/паузы бесплатным, если оно включено в admission.
- Зафиксированные local charges: interrupted legacy1840.938s; strict pair5757.969s; four exports621.101s; conservative profiles/startup600s; initial teacher export413.513s; stopped recovery508.186s. Это не новая allocation и не полный новый расчёт доступного остатка; сверять receipts и production reserve перед продолжением.
- Один local GPU worker при разрешённом возобновлении, strict FP32/autocast off/TF32 off, CPU/RAM по текущему bounded admission, swap0. Scratch на физическом диске `research_tmp`, не `/tmp`/`/dev/shm`.
- Были измеренные92–96°C и thermal pauses; старый R2 watcher не стартовал из-за relative path. Исправленные R4-based guards управляли owned jobs; сейчас все остановлены. Не запускать горячие replay/training одновременно с чужой нагрузкой.
- Пройдены Ruff/bаsh syntax для изменённых helpers; tree build/check на опубликованном9b619f5a8:1765nodes,1924preserved ledger rows, acyclic/current. После этой документационной передачи каталог не перестраивался; это не новый scientific check.
- Сохранены отрицательные результаты, OOM и partial artifacts. Не считать source/receipt review независимой тренировочной репликацией или fresh-CV подтверждением.

## Продолжение в новой сессии

1. Прочитать этот файл и актуальный `AGENTS.md`, затем `specs/current-state.md` и `quality_r2/FRONTIER.md`. Поздние указания пользователя важнее старых running/stop записей.
2. Сохранить посторонние изменения; проверить фактические jobs/resources/quota. **Local GPU hold остаётся**, пока пользователь явно не изменит распределение GPU.
3. По желанию пользователя продолжить лёгкие CPU/read-only проверки. Ошибка Kaggle сохранена для будущего разбора, не является текущим приоритетом и не требует немедленного retry.
4. Если GPU снова разрешён: сначала закончить S6 valid export и actual three-teacher grade; затем оценить бюджет train exports + complete matched student panel + final/package reserve. Не заменить frozen3teacher gate результатом two-teacher probe молча.
5. До student fit получить все3 train caches, выполнить `check_train_target.py`, зафиксировать input hashes. Использовать правильные teacher scales1.02/.99/.99 и unchanged strict source/recipe.
6. После полных matched результатов выбирать лучший deterministic endpoint по корректному локальному сравнению; проверить serving/native parity и только затем готовить сильнейшие ZIP. Публичный результат — отдельное подтверждение, не сумма прежнего public и local delta.
7. Любые новые полезные outputs индексировать в `research/unified/*/README.md`, H/E ledger вести append-only. Не запускать заново закрытые pure/SWA/weak-native ветки без новой гипотезы.

Короткий ввод для следующей сессии:

> Прочитай `research/unified/operations/SESSION_HANDOFF_20261114_R1.md`. Продолжай Wunder Challenge2 ради качества. GPU пока оставлен PHCapital; никаких автоматических local GPU starts. Сильный новый факт — equal FBC/TLead teacher +.000840 на exposed valid, student transfer ещё не проверен. Kaggle R2 ERROR на source-identity preflight; логи сохранены, разбор не приоритет. Сначала учитывай текущие указания и сохраняй существующие артефакты.
