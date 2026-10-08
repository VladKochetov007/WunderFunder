from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    changes = []
    for source in OUT.glob('f0_z3*_v2.json'):
        spec = json.loads(source.read_text())
        if 'hft' in spec['run_id']:
            spec['notes'] = 'Matched seed4 backbone/student; finishing seed3 matches actual Z3h. Fixed epoch0 GCorr finish, fixedhalf endpoint. Independent grader owns point/delta/bootstrap.'
        target = OUT / source.name.replace('_v2.json', '_R3.json')
        target.write_text(json.dumps(spec, indent=2) + '\n')
        changes.append({'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'corrected': target.name, 'corrected_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'run_id_unchanged': spec['run_id'], 'argv_unchanged': True})
    (OUT / 'Z3_EXACT_FOLD0_NOTES_CORRECTION_R3.json').write_text(json.dumps({'status': 'note_only_amendment_root_pending', 'changes': changes, 'scope': 'R2 remains immutable; R3 filename identifies corrected descriptive notes only. Commands/run names/priorities unchanged.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
