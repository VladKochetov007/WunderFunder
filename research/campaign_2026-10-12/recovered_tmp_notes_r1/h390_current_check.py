import importlib.util
p='research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/controller.py'
s=importlib.util.spec_from_file_location('h390_vt',p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
c='''date -u
for f in /root/run/h390_r3/s3.exit /root/run/h390_r3/s4.exit; do echo EXIT:$f; test -e $f && cat $f || echo active; done
echo H390
for f in /root/run/h390_r3/s3.supervisor.log /root/run/h390_r3/s4.log; do echo LOG:$f; rg '^(FIT |ep [0-9]+ batch|EXPORT |Saved|checkpoint|Traceback|ERROR|DONE)' $f | tail -n 12; done
echo OOD
rg '^(FIT |ep [0-9]+ batch|EXPORT |Saved|checkpoint|Traceback|ERROR|DONE)' /root/run/logs/pipeline_swn_r2.log | tail -n 12
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 /root/wnn | tail -1'''
r=m.ssh(c,timeout=60)
print(r.stdout)
