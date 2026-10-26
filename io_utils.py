from pathlib import Path
import hashlib,json

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def archive_bytes(archive, member):
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Missing tar member: {member}")
    return stream.read()
