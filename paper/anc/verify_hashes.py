"""Verify the shipped ancillary manifest without changing any file."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, name = line.split('  ', 1)
    path = root / name
    assert root in path.resolve().parents and path.is_file()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, name + ': checksum mismatch'
print('ALL ANCILLARY CHECKSUMS PASS')
