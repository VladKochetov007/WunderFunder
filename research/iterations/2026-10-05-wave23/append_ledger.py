from pathlib import Path
import json
R=Path.home()/'WunderFunder-creative'
events=json.loads((R/'research/iterations/2026-10-05-wave23/ledger_events.json').read_text())
for kind,rows in events.items():
 p=R/'research'/f'{kind}.jsonl'
 existing={json.loads(line)['id'] for line in p.read_text().splitlines() if line.strip()}
 with p.open('a') as f:
  for row in rows:
   if row['id'] not in existing:f.write(json.dumps(row)+'\n')
print('Ledger entries reconciled')
