"""Pure audit workload and extraction rules; does not import or run the artifact."""
import hashlib
from pathlib import Path
import zipfile

ROW_GROUPS = tuple(range(1873)) + tuple(range(97))
CALLBACKS = len(ROW_GROUPS) * 20000
EMITTED = len(ROW_GROUPS) * (20000 - 99)


def callback_identity(index, row_group, raw_id, raw_ids, seen, next_fresh):
    assert row_group == ROW_GROUPS[index]
    if index < 1873:
        assert raw_id not in seen
        result = raw_id
    else:
        assert raw_id == raw_ids[row_group]
        while next_fresh in seen:
            next_fresh += 1
        result = next_fresh
        next_fresh += 1
    assert result not in seen
    return result, next_fresh


def extract_exact(package, destination, manifest):
    destination = Path(destination)
    with zipfile.ZipFile(package) as archive:
        entries = archive.infolist()
        assert {e.filename for e in entries} == set(manifest)
        assert len(entries) == len(manifest)
        for entry in entries:
            p = Path(entry.filename)
            expected = manifest[entry.filename]
            assert not p.is_absolute() and '..' not in p.parts and str(p) not in ('', '.')
            assert entry.is_dir() == expected['is_dir']
            assert (entry.external_attr >> 16) & 0o170000 != 0o120000
            body = archive.read(entry)
            assert hashlib.sha256(body).hexdigest() == expected['sha256']
            assert len(body) == expected['bytes']
            dest = destination / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            if entry.is_dir():
                assert body == b''
                dest.mkdir(exist_ok=True)
            else:
                dest.write_bytes(body)
