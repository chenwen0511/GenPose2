#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_DIMS = [0.1778, 0.1778, 0.008]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def obj_bounds(path):
    vertices = []
    for line in path.read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.asarray(vertices)
    return (vertices.max(0) - vertices.min(0)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest_path = root / 'manifests/sample_manifest.jsonl'
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    splits = Counter(record['split'] for record in records)
    groups = defaultdict(set)
    hashes = defaultdict(dict)
    errors = []
    object_counts = Counter()
    b_rotated = 0

    for record in records:
        split = record['split']
        groups[record['group_id']].add(split)
        prefix = Path(record['new_path'])
        for suffix in ('color.png', 'depth.exr', 'mask.exr', 'meta.json'):
            path = Path(str(prefix) + '_' + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f'missing {path}')
        rgb_hash = sha256(Path(str(prefix) + '_color.png'))
        if rgb_hash in hashes and split not in hashes[rgb_hash]:
            errors.append(f'cross-split RGB duplicate {prefix}')
        hashes[rgb_hash][split] = str(prefix)
        meta = json.loads(Path(str(prefix) + '_meta.json').read_text())
        objects = meta.get('objects', {})
        if not isinstance(objects, dict):
            errors.append(f'objects not dict {prefix}')
            continue
        for obj in objects.values():
            object_counts[split] += 1
            if obj['meta']['bbox_side_len'] != EXPECTED_DIMS:
                errors.append(f'invalid dimensions {prefix}')
            q = np.asarray(obj['quaternion_wxyz'], dtype=float)
            if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6):
                errors.append(f'invalid quaternion {prefix}')
        if record['source_dataset'] == 'B':
            b_rotated += int(record.get('canonical_rotation_applied') is True)

    cross_groups = {group: sorted(values) for group, values in groups.items() if len(values) > 1}
    if cross_groups:
        errors.append(f'{len(cross_groups)} groups cross splits')
    if set(splits) != {'train', 'val', 'test'}:
        errors.append(f'invalid splits {dict(splits)}')
    b_count = sum(record['source_dataset'] == 'B' for record in records)
    if b_rotated != b_count:
        errors.append(f'B canonical conversion count {b_rotated}/{b_count}')

    report = {
        'status': 'PASS' if not errors else 'FAIL',
        'frames': len(records),
        'split_frames': dict(splits),
        'split_instances': dict(object_counts),
        'groups': len(groups),
        'cross_split_groups': len(cross_groups),
        'cross_split_rgb_duplicates': sum(len(v) > 1 for v in hashes.values()),
        'B_canonical_rotation_applied': b_rotated,
        'obj_bounds_m': obj_bounds(root / 'assets/tray_z_normal_v2.obj'),
        'expected_dimensions_m': EXPECTED_DIMS,
        'errors': errors[:100],
    }
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
