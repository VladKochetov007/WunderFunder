from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
b=Path('research/iterations/2026-10-01-ensemble-backbone')
def read(p):return json.loads((b/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 p=b/name;assert not p.exists();p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
r=read('evaluation-01/report.json');rc=read('evaluation-01/receipt.json');p=read('completed-local-preservation.json');fit=read('fit-local-preservation.json');a=read('cache-local-archive.json')
assert r['status']=='COMPLETE_FIXED_ENSEMBLE_BACKBONE_COMPARISONS'
for name,expected in rc['outputs'].items():assert sha(b/'evaluation-01'/name)==expected['sha256']
remote_paths={x['path'] for x in p['records']}
assert {x['original_path'] for x in a['records']}<=remote_paths
assert {path for job in fit['records'] for path in job['outputs']}<=remote_paths
assert len([x for x in p['records'] if x['representation']=='gzip_lossless'])==6
primary=r['comparisons']['primary']['scopes']['all'];op=r['comparisons']['operational']['scopes']['all'];quiet=r['comparisons']['primary']['scopes']['class3_quiet64']
gates={k:dict(pooled_at_least_0003=v['scopes']['all']['delta_WP']>=.0003,positive_both_folds=all(f['delta_WP']>0 for f in v['scopes']['all']['folds'].values())) for k,v in r['comparisons'].items()}
passed=all(all(x.values()) for x in gates.values());assert not passed
hashes=r['provenance']['input_sha256'];oldroot='/home/grinchenkoeg/WunderFunder-creative/'
identity={str(f):hashes[oldroot+str(b/f'cache/pred_eb_control_s3_h2_f{f}_valid.npy')]==hashes[oldroot+f'research/iterations/2026-10-01-continuation/cache/pred_cont_sh256_s3h2_f{f}_valid.npy'] for f in [0,1]};assert all(identity.values())
workers=[json.loads(x.read_text()) for d in ['execution-01','execution-02'] for x in (b/d).glob('*.worker-result.json')]
assert len(workers)==12 and all(x['status']=='COMPLETE' for x in workers)
cost={k:sum(x['seconds'] for x in workers if x['stage']==k) for k in ['parent','H2','OOF']};cost['cache_queue_seconds']=read('cache-execution-01/result.json')['seconds'];cost['sequential_cache_plus_job_wall_minutes']=sum(cost.values())/60
checks=read('paired-trajectory-check.json');assert checks['status']=='PASS_PAIRED_INITIALIZATION_AND_CPU_RNG'
finish=read('finish-02/result.json');assert finish['status']=='COMPLETE_ONE_PAIRED_STUDY' and finish['remaining_scope_pids']==[]
write('decision.json',dict(status='CLOSED_SCREEN_FAILED',created_utc=datetime.now(timezone.utc).isoformat(),verification_tier='B_exposed_discovery',scientific_claim_status='screen_failed',recipe_scope='Design A, raw211 shallow256 seed3, parent4ep EMAep3 + H2twoep ep1, fixed quarter anchors; no claim against the whole mechanism family',gate=gates,candidate_WP=primary['candidate']['WP'],fresh_control_WP=primary['control']['WP'],current_student4_WP=op['control']['WP'],delta_WP=primary['delta_WP'],fold_deltas={f:x['delta_WP'] for f,x in primary['folds'].items()},conditional95_chain_CI=primary['paired_chain_CI']['interval95'],target_rho_deltas=primary['delta_rho'],quiet64_delta=quiet['delta_WP'],fresh_control_prediction_byte_identity=identity,comparisons_independent=False,cost=cost,paired_seeds=[3],fits=8,OOF=4,evaluation_calls=1,quality_retries=0,resource_resumptions=1,pretraining_mechanical_fixes=1,promoted=False,new_full_data_training=False,native_candidate_qualification=False,github_pushes=0,public_submissions=0,report_sha256=sha(b/'evaluation-01/report.json'),receipt_sha256=sha(b/'evaluation-01/receipt.json'),preservation_sha256=sha(b/'completed-local-preservation.json'),next_discriminating_study='Diagnostic-only fixed ep3 parent OOF with unchanged quarter-weight partners: distinguish no parent gain from a parent/H2 stage interaction. New separately frozen study, no fit, endpoint search, blend search or retrospective promotion.',next_study_executed=False))
write('final-review.json',dict(status='PASS_INDEPENDENT_COMPLETION_AND_GATE_REVIEW',reviewer='/root/architecture_challenge',method='Read-only independent contract, raw report/receipt, hash lineage, paired-proof, all worker completions and released scope review; no model rerun or rescore.',gate='FAIL',blockers=[],findings=['Both fixed comparisons fail practical gain and both-fold positivity gates.','Fresh-control OOF arrays match current-student arrays bytewise in both folds; comparisons are not independent corroboration.','All 12 scientific jobs completed successfully, with 8 fits/4 OOF and no scientific retry.','Paired-proof and frozen evaluator provenance pass; worker scopes are released.','Close only this recipe; pooled CI crosses zero and one seed/exposed folds cannot support mechanism-wide impossibility.'],report_sha256=sha(b/'evaluation-01/report.json'),receipt_sha256=sha(b/'evaluation-01/receipt.json')))
text=f'''Полное обучение backbone под цель фиксированного ансамбля **не прошло заранее установленный порог продолжения**. Этот рецепт закрыт со статусом `screen_failed`; действующий `student_fixed4.zip` не заменён. Все 8 обучений и 4 OOF-прохода завершены, итоговая оценка выполнена один раз.

| Фиксированный ансамбль | Pooled GPU WP | Разница кандидата |
|---|---:|---:|
| Кандидат: ensemble-objective parent + H2 | {primary['candidate']['WP']:.10f} | — |
| Свежий парный контроль | {primary['control']['WP']:.10f} | {primary['delta_WP']:+.10f} |
| Прежний current_student4 | {op['control']['WP']:.10f} | {op['delta_WP']:+.10f} |

По фолдам ΔWP: **{primary['folds']['0']['delta_WP']:+.10f} / {primary['folds']['1']['delta_WP']:+.10f}**. По целям Δρ: **{primary['delta_rho'][0]:+.10f} / {primary['delta_rho'][1]:+.10f}**. Условный 95% интервал парного chain-bootstrap для pooled Δ: **[{primary['paired_chain_CI']['interval95'][0]:+.10f}; {primary['paired_chain_CI']['interval95'][1]:+.10f}]**. Он включает ноль; это не доказательство общего ухудшения или невозможности метода. Практический критерий был +0.0003 к обоим контролям и положительная разница на каждом фолде — он не выполнен.

В заранее заявленном диагностическом срезе `class3_quiet64` (276199 строк, 118 цепочек) ΔWP **{quiet['delta_WP']:+.10f}**, 95% CI [{quiet['paired_chain_CI']['interval95'][0]:+.10f}; {quiet['paired_chain_CI']['interval95'][1]:+.10f}]. Положительный частный фрагмент не использовался для выбора иной модели или смеси.

Свежий контроль **побайтно воспроизвёл старые OOF-предсказания обоих фолдов**. Поэтому primary и operational здесь численно совпадают и не являются независимыми подтверждениями. Это полезная проверка воспроизводимости контрольного пайплайна. SHA массивов сохранены в provenance [сырого отчёта](evaluation-01/report.json).

Проверялся один конкретный механизм: вся shallow256 сеть учится с главным прогнозом `e=(((p+a1)+a2)+a3)/4`, с множителем loss 4. Фиксированы три партнёра, seed3, 211 признаков, 4 эпохи parent с EMAep3, две эпохи H2 с ep1. Дистилляция, auxiliary/EMA losses, веса строк, данные, порядок операций и условия контроля сохранены. Это действительно полный backbone-тест, а не повтор head-only. [Контракт](contract.md), [команды](commands.json), [paired-proof](paired-trajectory-check.json), [решение](decision.json).

Оценка: 860 последовательностей / 250 целых цепочек / 1614484 scored rows, уже многократно использованные fold0/1. Bootstrap: 2000 общих парных выборок по цепочкам, стратификация fold×class, seed20261001. Уровень проверки B для локального измерения. Один парный seed; нет нового независимого holdout, проверки transfer или seed robustness. Интервалы условны на обученные модели, маргинальные, не покрывают выбор гипотезы, тренировочную вариативность и скрытый тест. In-sample train-предсказания партнёров — отдельный риск переобучения. Данный опыт не определяет, какая из этих причин объясняет результат.

[Независимый критик](final-review.json) проверил gate, связи хешей, парность запусков и завершение всех задач; блокеров нет. Грейдер заморожен до обучения (`c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9`); штатная формула также сверена с отдельным `fold_eval`. Повтор оценки фиксированных прогнозов не выдаётся за независимое обучение или подтверждение обобщения.

Затраты: шесть train-anchor проходов **{cost['cache_queue_seconds']/60:.2f} мин**, четыре parent **{cost['parent']/60:.2f} мин**, четыре H2 **{cost['H2']/60:.2f} мин**, четыре OOF **{cost['OOF']/60:.2f} мин**. Суммарно около **{cost['sequential_cache_plus_job_wall_minutes']:.2f} последовательных минут этих GPU-задач**, включая загрузку/служебное время внутри jobs, а не измерение GPU utilization. Финальный численный grader {r['seconds']:.2f} с. Пик памяти собственных GPU-процессов 6068 MiB.

Две технические ситуации сохранены явно: до первого обучения исправлен абсолютный путь `--inputs` отдельной версией контроллера; позднее чужая задача заняла GPU6 и следующий job не прошёл проверку свободной памяти. Пять завершённых шагов сохранены, семь ещё не начатых перенесены на однотипную GPU0. Ни один научный job не переобучался; чужие процессы не изменялись. [История переноса](resource-resumption.md), [составной результат](completed-study-01/result.json), [завершение и освобождение scope](finish-02/result.json). Вариативность CUDA между устройствами не объявляется невозможной.

Сохранены все веса, предсказания, команды, журналы и неудачные служебные попытки. [Манифест](completed-local-preservation.json): **233 удалённых файла, 11460932280 исходных байт**. Шесть train-кэшей локально представлены lossless gzip, каждый архив повторно распакован потоком и проверен по SHA исходника; сырые оригиналы на SSH сохранены. Экономия локального места **4401490739 байт**. [Все 12 job outputs](fit-local-preservation.json), [архивы кэшей](cache-local-archive.json), [инвентаризация SSH](remote-completed-inventory.json). Поздние документы закрытия перечисляются отдельно в `closure-sync.json`; сам манифест не включает себя рекурсивно.

Восстановление локального train-кэша: `gzip -dk research/iterations/2026-10-01-ensemble-backbone/cache/pred_RUN_train.npy.gz`. Для исторического воспроизведения использовать зафиксированные argv/env из двух allocation manifests и содержимое input bindings; существующие immutable выходы не перезаписывать. Грейдер может быть запущен с `--bindings /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-01-ensemble-backbone/evaluation-bindings-01.json --bindings-sha256 6890336d7c919b719249c37c8fe139f18a0910fe32a2a02536ee2cc9f80138d6 --output-dir НОВЫЙ_ПУТЬ`; это только воспроизведение той же фиксированной оценки.

Рабочая подача: [student_fixed4.zip](../2026-10-01-continuation/student-runtime/package/student_fixed4.zip), 6000321 байт, SHA `eec571629b0829aeb9c73db80a1170fcb7cb3ccfd132f72f3ddd51f779d2fdd4`. Её прежняя native OOF-оценка 0.702355321333 и проверенный runtime финального full-data ZIP 2672.212 с относятся к отдельной прошлой квалификации; текущий GPU-эксперимент её не обновлял. На GitHub и в соревнование ничего не отправлено.

Следующий наиболее различающий дешёвый шаг — отдельная диагностическая оценка **сохранённых parent ep3 до H2** в тех же фиксированных ансамблях. Четыре OOF-прохода, без нового обучения, выбора эпох или весов. Если parent уже проигрывает, искать причину внутри parent objective / in-sample партнёров; если parent выигрывает, а конечная связка проигрывает, проверять взаимодействие двух стадий. Даже такой результат лишь локализует стадию, а не доказывает причину. Он не разрешает задним числом выбрать parent как успешную подачу; для изменённого рецепта нужен новый заранее фиксированный тест. Этот следующий диагностический опыт здесь **не выполнялся**.

Отдельная [математическая проверка atom-декодера](alternative-audit/readout.md) нашла двойное амплитудное взвешивание в связке isolated CE→decode. Искусственный oracle подтверждает несогласованность формулы, но не объясняет автоматически проигрыш совместной atom-модели и не доказывает прирост исправления. Аудит не использован для запуска новой дорогой ветки.
'''
out=b/'readout.md';assert not out.exists();out.write_text(text)
print(json.dumps(dict(status='CLOSED_SCREEN_FAILED',cost=cost,report_sha256=sha(b/'evaluation-01/report.json'),readout=str(out)),ensure_ascii=False))
