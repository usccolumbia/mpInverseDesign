#!/usr/bin/env python3
"""Restore the original train/val/test CIFs from the shipped JSON bundle.

Run from any directory: python3 /path/to/mpInverseDesign/scripts/extract_cifs.py
Only the Python standard library is required. Existing identical files are
skipped; different files are never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]


def prepare(bundle_path, output_dir):
    bundle = json.loads(bundle_path.read_bytes().decode("utf-8"))
    if bundle.get("format") != "mpInverseDesign.cifs.v1":
        raise ValueError("Unsupported CIF bundle format")
    if bundle.get("encoding") != "utf-8":
        raise ValueError("Unsupported CIF encoding")
    output_dir = output_dir.resolve()
    seen, counts, prepared = set(), Counter(), []
    for entry in bundle["files"]:
        name = entry["path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or len(path.parts) != 2
                or path.parts[0] not in {"train", "val", "test"}
                or path.suffix != ".cif" or "\\" in name
                or name != path.as_posix() or name in seen):
            raise ValueError(f"Invalid or duplicate CIF path: {name}")
        data = entry["cif"].encode("utf-8")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"CIF checksum mismatch: {name}")
        target = output_dir.joinpath(*path.parts)
        if not target.resolve().is_relative_to(output_dir):
            raise ValueError(f"CIF path leaves output directory: {name}")
        seen.add(name)
        counts[path.parts[0]] += 1
        prepared.append((target, data))
    if dict(counts) != bundle["counts"]:
        raise ValueError("CIF split counts do not match the bundle")
    return prepared, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data/mp20_5k/cifs.json",
                        help="CIF JSON bundle (default: repository data/mp20_5k/cifs.json)")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "data/mp20_5k",
                        help="Directory in which to restore train/val/test folders")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check bundle paths, counts and checksums without writing files")
    args = parser.parse_args()
    try:
        prepared, counts = prepare(args.input, args.output_dir)
        if args.verify_only:
            print(f"Verified {len(prepared):,} CIFs: {dict(counts)}")
            return
        # Check every destination before writing any CIFs.
        missing, skipped = [], 0
        for target, data in prepared:
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise ValueError(f"Refusing to overwrite a different file: {target}")
                skipped += 1
            else:
                missing.append((target, data))
        for target, data in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
        print(f"Restored {len(missing):,} CIFs; {skipped:,} identical files already present.")
        print(f"Split counts: {dict(counts)}")
        print(f"Output: {args.output_dir.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
