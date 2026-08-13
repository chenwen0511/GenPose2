#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-class", default="smt_tray")
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset not found: {root}")

    prefixes = sorted(root.rglob("*_color.png"))
    required_suffixes = ("color.png", "depth.exr", "mask.exr", "meta.json")
    split_counts = Counter()
    class_names = Counter()
    object_ids = Counter()
    dimensions = Counter()
    missing = []
    invalid_json = []
    hashes_by_split = defaultdict(set)
    cross_split_duplicates = []

    for color in prefixes:
        prefix = Path(str(color)[: -len("color.png")])
        rel = color.relative_to(root)
        split = next((part for part in rel.parts if part in {"train", "val", "test"}), "unknown")
        split_counts[split] += 1
        for suffix in required_suffixes:
            candidate = Path(str(prefix) + suffix)
            if not candidate.is_file() or candidate.stat().st_size == 0:
                missing.append(str(candidate))
        meta_path = Path(str(prefix) + "meta.json")
        try:
            meta = json.loads(meta_path.read_text())
            objects = meta.get("objects", [])
            if isinstance(objects, dict):
                objects = list(objects.values())
            for obj in objects:
                payload = obj.get("meta", obj)
                class_names[str(payload.get("class_name"))] += 1
                object_ids[str(payload.get("oid"))] += 1
                dims = payload.get("bbox_side_len")
                if dims is not None:
                    dimensions[tuple(round(float(x), 8) for x in dims)] += 1
        except Exception as exc:
            invalid_json.append(f"{meta_path}: {exc}")
        rgb_hash = digest(color)
        for other_split, seen in hashes_by_split.items():
            if other_split != split and rgb_hash in seen:
                cross_split_duplicates.append(str(rel))
        hashes_by_split[split].add(rgb_hash)

    report = {
        "dataset": str(root),
        "frames": len(prefixes),
        "split_counts": dict(split_counts),
        "class_names": dict(class_names),
        "unique_object_ids": len(object_ids),
        "dimensions": {str(key): value for key, value in dimensions.items()},
        "missing_or_empty": missing[:100],
        "invalid_json": invalid_json[:100],
        "cross_split_exact_rgb_duplicates": cross_split_duplicates[:100],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    required_splits = {"train", "val", "test"}
    failed = bool(
        missing
        or invalid_json
        or cross_split_duplicates
        or set(split_counts) != required_splits
        or set(class_names) != {args.expected_class}
        or len(dimensions) != 1
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
