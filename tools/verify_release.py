"""Verify the release coverage and source/third-party attribution boundary offline."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'SOURCE_COVERAGE.json').read_text())
    paths={r['destination_path'] for r in report['retained']}
    for record in report['retained']:
        path=ROOT/record['destination_path']
        assert path.is_file(), str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['destination_sha256'], str(path)
        if path.suffix=='.py':
            ast.parse(path.read_text(),filename=str(path))
        if record['provenance_class']=='FIRST_PARTY' and path.suffix in ('.py','.md','.txt'):
            text=path.read_text()
            assert not any(fragment in text for fragment in ('/home/','/root/','/srv/','/Users/')), str(path)
    for record in report['source_inventory']:
        assert record['reason']
        if record['disposition']!='EXCLUDED':
            assert record['destination'] in paths, record
    assert not (ROOT/'LICENSES/LICENSE').exists(), 'Original proprietary root terms must not override authorized first-party grant'
    assert (ROOT/'LICENSES/amazon_braket_NOTICE.txt').is_file()
    assert (ROOT/'THIRD_PARTY_NOTICES.md').is_file()
    # Upstream /opt container paths are public SDK conventions, not private estate paths.
    binaries=[r['destination_path'] for r in report['retained'] if Path(r['destination_path']).suffix=='.npz']
    assert all(path.startswith('algorithms_extended/qiskit_experiments_asymmetrica/') for path in binaries), binaries
    print(f'PASS: {len(paths)} retained artifacts hash-verified, source parses, upstream notices preserved')


if __name__=='__main__':
    main()
