import json, time, glob, os, sys
d = "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-29/jobs"
seen = {}
for p in glob.glob(d + "/c2026100*.json"):
    try: seen[p] = json.load(open(p)).get("status")
    except Exception: pass
while True:
    for p in glob.glob(d + "/c2026100*.json"):
        try: j = json.load(open(p))
        except Exception: continue
        st = j.get("status")
        if seen.get(p) != st:
            seen[p] = st
            if st != "running" or p not in seen:
                print(f"{os.path.basename(p)[:-5]} {st} exit={j.get('exit_code')} {round(j.get('elapsed_seconds') or 0)}s vram={j.get('peak_observed_vram_bytes')} fail={j.get('failure')}", flush=True)
            else:
                print(f"{os.path.basename(p)[:-5]} started", flush=True)
    time.sleep(5)
