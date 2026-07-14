import hashlib
from pathlib import Path


def sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Used for duplicate detection.
    """

    hash_obj = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            hash_obj.update(chunk)

    return hash_obj.hexdigest()