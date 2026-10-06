# Восстановление материалов продолжения

Эта папка создана отдельно от закрытой `delivery-2026-10-01`. Готовность всей выдачи определяется итоговым `manifest.json`; точный состав архива — `research-archive.manifest.json`.

Для самой посылки достаточно одного ZIP: код, веса и нативные библиотеки находятся внутри. Загружать следует неизменённый архив. `student_fixed4.zip` — однослойный GRU256 и три прежних GRU128; `strong4_int8.zip` — прежние четыре сильные модели в INT8. Фактическое время и статус проверки каждого файла указаны в итоговом README и манифесте. Исполнение рассчитано на Linux/x86-64, CPython 3.11; измеренное время относится к пути v3/VNNI на beleriand. Время на другом оборудовании не гарантируется.

Исходники и новые обученные модели восстанавливаются так:

```sh
git clone -b research/creative-iteration-20260930 research-source.bundle restored
tar -xzf research-continuation.tar.gz -C restored
tar -xzf error-analysis.tar.gz -C restored
git -C restored rev-parse HEAD
```

Bundle сохраняет Git-историю с зафиксированными исходниками продолжения. Архив добавляет отчёты, результаты, нативные артефакты и все промежуточные/конечные checkpoints новых обучений. Точный состав и SHA256 файлов перечислены в архивном манифесте. Старые эксперименты остаются в предыдущей выдаче и историческом архиве; они не переписываются.

Организаторские Parquet-файлы, большие teacher-кэши и OOF-массивы в эту переносимую копию не входят. Они сохранены на beleriand; отчёты содержат их пути, идентификаторы и хэши. Для повторного обучения понадобятся эти входы и окружение проекта. Точная эпоха генерации старых teacher-кэшей исторически не зафиксирована, поэтому отсутствие кэшей нельзя исправить молчаливой заменой учителя или обещанием побитового повторения обучения. Для запуска готового ZIP эти внешние файлы не нужны.

Новые материалы не опубликованы на GitHub и не отправлены в соревнование.

Expected HEAD: `3e8f1855afaa847f95d23a327bcff402a7ebfede`. The source bundle has verified complete history. The separate error-analysis archive restores `research/iterations/2026-10-01-error-analysis`, including derived diagnostic arrays; original large OOF/teacher caches and organizer datasets are excluded. Both archives were fully read back and member hashes checked. Exact file inventories are in `research-archive.manifest.json` and `error-analysis.manifest.json`.
