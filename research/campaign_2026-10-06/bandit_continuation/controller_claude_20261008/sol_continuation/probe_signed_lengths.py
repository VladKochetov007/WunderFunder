from __future__ import annotations

import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    receipt = json.loads((OUT / 'R5_SIGNED_INPUT_FEASIBILITY_R1.json').read_text())
    plan = json.loads(Path(receipt['private_plan_location']).read_text())
    results = []
    for item in plan['inputs']:
        request = urllib.request.Request(item['url'], method='HEAD')
        with urllib.request.urlopen(request, timeout=30) as response:
            observed = int(response.headers['Content-Length'])
            if observed != item['size_bytes']:
                raise ValueError('signed object content length differs from frozen identity')
            results.append({'remote_path': item['remote_path'], 'expected_bytes': item['size_bytes'], 'observed_content_length': observed, 'status': response.status, 'method': 'HEAD', 'body_bytes_read': 0})
    result = {'status': 'all_four_signed_content_lengths_match', 'inputs': results, 'scope': 'read-only HEAD; no body bytes read; actual remote full hashes still required'}
    (OUT / 'R5_SIGNED_CONTENT_LENGTH_R1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'count': len(results)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'exception_class': type(error).__name__}))
        raise SystemExit(1) from None
