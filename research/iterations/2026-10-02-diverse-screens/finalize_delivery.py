"""Package already qualified evidence; no training, inference, or metric selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[3]
A=ROOT/'research/iterations/2026-10-02-diverse-screens'
MD=ROOT/'research/iterations/2026-10-02-midpoint-delivery'
OUT=ROOT.parent/'delivery-2026-10-02-midpoint'
EXPECTED='ae5998ac36ee1153a3df2a6b261238c97cdbe5f9ba7aaba761f5711693d5b23c'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def main():
    preservation=read(MD/'preservation-runtime.json')
    assert preservation['status']=='PASS_LOCAL_SSH_LOSSLESS_PRESERVATION'
    r=read(MD/'runtime/execution/launch-result.json')
    w=read(MD/'runtime/execution/worker-result.json')
    n=read(MD/'native-quality/evaluation/report.json')
    plan=read(MD/'runtime/runtime-plan.json')
    assert r['plan_sha256']==w['plan_sha256']==sha(MD/'runtime/runtime-plan.json')
    assert plan['artifact']['sha256']==EXPECTED and plan['artifact']['bytes']==5995375
    assert plan['artifact']['path']==str((MD/'package/midpoint_student_fixed4.zip').relative_to(ROOT))
    assert preservation['phase']=='runtime'
    assert preservation['terminal_results']['runtime/execution/launch-result.json']==r
    assert r['status']=='COMPLETE' and r['exit_code']==0
    assert r['delivery_runtime_qualified'] and r['functional_complete'] and r['process_deadline_observed']
    assert r['actual_full_process_at_most_60min'] and not r['hard_timeout']
    assert r['worker_result_sha256']==sha(MD/'runtime/execution/worker-result.json')
    assert w['status']=='COMPLETE' and w['callback_count']==39400000 and w['emitted_count']==39204970
    assert w['sequence_resets_observed']==1970 and w['warmup_none_count']==195030
    assert w['final_member_count']==4 and w['member_count_transitions']==[]
    assert not w['targets_decoded'] and not w['scoring_masks_read'] and w['WP_queries']==0
    rows=[json.loads(x) for x in (MD/'runtime/execution/sequences.jsonl').read_text().splitlines()]
    assert len(rows)==1970 and all(x['fixed_member_count']==4 for x in rows)
    assert len({x['seq_ix'] for x in rows})==1970
    repeats=rows[1873:]
    assert len(repeats)==97
    for original, repeat in zip(rows[:97],repeats):
        assert original['raw_seq_ix']==repeat['raw_seq_ix'] and original['row_group']==repeat['row_group']
        assert original['emitted_f32_sha256']==repeat['emitted_f32_sha256']
        assert original['seq_ix']!=repeat['seq_ix']
    q=n['comparisons']['primary']['scopes']['all']
    d=n['comparisons']['native_minus_GPU_midpoint']['scopes']['all']
    assert n['status']=='COMPLETE_FIXED_NATIVE_MIDPOINT_COMPARISON'
    assert q['delta_WP']>=.0001 and all(v['delta_WP']>0 for v in q['folds'].values())
    assert d['delta_WP']>=-.0001 and all(v['delta_WP']>=-.00015 for v in d['folds'].values())
    zip_path=MD/'package/midpoint_student_fixed4.zip'
    assert sha(zip_path)==EXPECTED and zip_path.stat().st_size==5995375
    assert not OUT.exists()
    OUT.mkdir()
    files={
        'midpoint_student_fixed4.zip':zip_path,
        'midpoint_student_fixed4.manifest.json':MD/'package/midpoint_student_fixed4.manifest.json',
        'native-recipe-OOF.json':MD/'native-quality/evaluation/report.json',
        'runtime-result.json':MD/'runtime/execution/launch-result.json',
        'runtime-worker-result.json':MD/'runtime/execution/worker-result.json',
        'research-deltas.png':A/'fixed-screen-deltas.png',
    }
    for name,src in files.items():
        shutil.copyfile(src,OUT/name); assert sha(src)==sha(OUT/name)
    evidence={
        'status':'QUALIFIED_SELECTED_PACKAGE_LOCAL_SSH_NO_PUBLICATION',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'selected_package_sha256':EXPECTED,
        'selected_package_bytes':zip_path.stat().st_size,
        'native_OOF_recipe':{'candidate_WP':q['candidate']['WP'],'control_WP':q['control']['WP'],
            'delta_WP':q['delta_WP'],'conditional_chain_CI95':q['paired_chain_CI']['interval95'],
            'fold_deltas':{k:v['delta_WP'] for k,v in q['folds'].items()}},
        'actual_final_ZIP_runtime':{'seconds':r['process_wall_seconds'],'minutes':r['actual_full_process_minutes_39_4M'],
            'callback_count':w['callback_count'],'emitted_count':w['emitted_count'],
            'resets':w['sequence_resets_observed'],'repeat_sequences_byte_identical':97,
            'fixed_members':4,'max_observed_rss_KiB':w['max_observed_rss_KiB'],
            'quiet_runtime_qualified':r['quiet_runtime_qualified'],'native_level':w['natural_native_level'],
            'integer_backend':w['integer_backend']},
        'source_receipt_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [MD/'preservation-runtime.json',
            MD/'runtime/execution/launch-result.json',MD/'runtime/execution/worker-result.json',
            MD/'native-quality/evaluation/report.json',MD/'root-promotion-contract.json']},
        'delivery_files':{name:{'bytes':(OUT/name).stat().st_size,'sha256':sha(OUT/name)} for name in files},
        'fallback_package_sha256':'eec571629b0829aeb9c73db80a1170fcb7cb3ccfd132f72f3ddd51f779d2fdd4',
        'limits':['OOF is fold-specific recipe evidence, not held-out evaluation of final full-data ZIP.',
            'All folds exposed; midpoint designed after warm seed3/seed4. Conditional bootstrap omits adaptive selection and seed population uncertainty.',
            'Midpoint seed4 +0.0000289217 has confidence interval crossing zero.',
            'Legacy teacher-cache producer epochs unknown; same byte-bound caches as prior anchor.',
            'Measured natural v3/VNNI host only, warm filesystem cache; no AVX2/v2 runtime qualification.',
            'No public submission, leaderboard score, or GitHub push. Modest gain, no rank guarantee.']}
    for p in [OUT/'evidence.json',A/'delivery-decision.json']:
        with p.open('x') as f:json.dump(evidence,f,indent=2);f.write('\n')
    minutes,seconds=divmod(r['process_wall_seconds'],60)
    text=f"""# Пакет 2 октября: midpoint student fixed4

Готовый ZIP: [midpoint_student_fixed4.zip](midpoint_student_fixed4.zip).

Native OOF **рецепта**: {q['candidate']['WP']:.10f} против {q['control']['WP']:.10f}, разница **{q['delta_WP']:+.10f}**; оба folds положительны. Это небольшой адаптивно выбранный выигрыш на уже использованных folds. На втором seed midpoint даёт лишь +0,00002892 с интервалом через ноль. Убедительного подтверждения устойчивости по seeds нет.

Финальный ZIP обучен на полных данных. На beleriand, natural v3/VNNI, один CPU: **{int(minutes)} мин {seconds:.2f} с** на все 39 400 000 вызовов, включая startup, чтение и проверки. Все четыре модели активны постоянно; 1970 сбросов, 97 повторных последовательностей побитово совпадают. Пик RSS {w['max_observed_rss_KiB']/1024:.2f} MiB. Qualification изоляции SMT: {r['quiet_runtime_qualified']}. Другие ISA/CPU этим замером не квалифицированы.

Меняется только первый checkpoint: фиксированная середина исходных весов GRU256 и весов после дополнительного этапа обучения. Wrapper, binaries, остальные три участника и веса 1/4 сохранены. Время не сравнивается как эксперимент на ускорение: разные замеры могут зависеть от нагрузки хоста.

SHA256 ZIP: `{EXPECTED}`.

[Native OOF](native-recipe-OOF.json), [полный runtime](runtime-result.json), [сводка доказательств и ограничения](evidence.json), [отчёт по всем веткам](../WunderFunder-creative/research/iterations/2026-10-02-diverse-screens/readout.md).

Предыдущий квалифицированный пакет сохранён в `../delivery-2026-10-01-continuation/student_fixed4.zip`. GitHub и публичная отправка не выполнялись. Место на скрытом leaderboard и существенный скачок метрики не подтверждены.
"""
    (OUT/'README.md').write_text(text)
    print(json.dumps({'status':evidence['status'],'directory':str(OUT),'runtime_seconds':r['process_wall_seconds'],'delta_WP':q['delta_WP']}))
if __name__=='__main__':main()
