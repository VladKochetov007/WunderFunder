# Восстановление ночного исследования

`research-overnight.tar.gz` — архив исследования; для соревнования предназначены отдельные маленькие ZIP. Архив содержит все локальные материалы ночной кампании, все 52 обученных запуска (224 файла), точный Git bundle исходников и три неизменные посылки. Байтовая опись находится в `archive-manifest.json` внутри архива и `research-archive.manifest.json` рядом с ним; SHA256 самого архива — в `research-archive.receipt.json`.

Извлеките архив в новую пустую папку. Его `WunderFunder-creative/` содержит материалы и веса; `delivery-2026-10-01/` содержит посылки и пояснения. Для восстановления полноценной рабочей копии:

```sh
git init restored
git -C restored fetch ../research-source.bundle HEAD
git -C restored checkout -b research-restored FETCH_HEAD
cp -R WunderFunder-creative/research/iterations/2026-10-01-overnight restored/research/iterations/
mkdir -p restored/artifacts/runs
cp -R WunderFunder-creative/artifacts/runs/. restored/artifacts/runs/
```

Git bundle содержит историю исходников до указанного в описи commit. Дополнительные численные артефакты, native-библиотеки, квитанции и веса восстанавливаются из соответствующих папок архива. Прогнозные NPY-кэши большого размера и тела датасетов не включены. Они сохраняются на `beleriand:/home/grinchenkoeg/WunderFunder-creative`; входные и OOF-квитанции в исследовании фиксируют пути и SHA256. Старые исторические 268 запусков относятся к прежнему архиву, указанному в `research/CURRENT_STATE.md`; этот пакет добавляет именно 52 запуска ночной кампании.

Архив не воспроизводит серверное Python/CUDA-окружение автоматически. Полные исторические планы имеют истёкшие временные ограничения и неизменяемые output-пути: не запускайте их поверх сохранённых результатов. Новое обучение требует отдельного output-каталога и нового плана. `state-refresh-preparation` — только будущий helper и проект диагностики, без готового тренера или разрешённого запуска.
