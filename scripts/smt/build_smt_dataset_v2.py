#!/usr/bin/env python3
"""Build the canonical, leak-free SMT GenPose2 training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEED = 20260813
CANONICAL_OID = 'smt_factory-smt_tray-000001'
CANONICAL_CLASS = 'smt_tray'
CANONICAL_DIMS = [0.1778, 0.1778, 0.008]
CANONICAL_SYMMETRY = {'any': False, 'x': 'none', 'y': 'none', 'z': 'any'}
FILE_TYPES = ('color.png', 'depth.exr', 'mask.exr', 'meta.json')


@dataclass
class Frame:
    source_dataset: str
    source_prefix: Path
    source_split: str
    scene_id: str
    frame_id: str
    group_id: str
    object_count: int
    split: str = ''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def quat_to_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = np.asarray([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def matrix_to_quat(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
             (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = [(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                 (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s]
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = [(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s,
                 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = [(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s,
                 (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    return q.tolist()


def b_canonical_rotation():
    # New canonical coordinates have z=old x, x=old y, y=old z.
    # p_old = C @ p_new, so T_camera_new = T_camera_old @ C.
    return np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def transform_obj(obj, source_dataset):
    old_q = obj['quaternion_wxyz']
    old_world_q = obj.get('world_quaternion_wxyz', old_q)
    if source_dataset == 'B':
        conversion = b_canonical_rotation()
        q = matrix_to_quat(quat_to_matrix(old_q) @ conversion)
        world_q = matrix_to_quat(quat_to_matrix(old_world_q) @ conversion)
    else:
        q = old_q
        world_q = old_world_q
    mask_id = int(obj.get('mask_id', obj.get('id')))
    return str(mask_id), {
        'meta': {
            'oid': CANONICAL_OID,
            'class_name': CANONICAL_CLASS,
            'class_label': 0,
            'instance_path': '',
            'scale': [1.0, 1.0, 1.0],
            'is_background': False,
            'bbox_side_len': CANONICAL_DIMS,
        },
        'quaternion_wxyz': q,
        'translation': obj['translation'],
        'is_valid': bool(obj.get('is_valid', True)),
        'id': mask_id,
        'material': obj.get('material', []),
        'world_quaternion_wxyz': world_q,
        'world_translation': obj.get('world_translation', obj['translation']),
    }


def transform_meta(meta, source_dataset):
    objects = meta.get('objects', {})
    iterable = objects if isinstance(objects, list) else objects.values()
    meta['objects'] = dict(transform_obj(obj, source_dataset) for obj in iterable)
    env = dict(meta.get('env_param', {}))
    env.update({
        'pose_frame_id': 'smt_tray_z_normal_v2',
        'canonical_dimensions_m': CANONICAL_DIMS,
        'canonical_symmetry': CANONICAL_SYMMETRY,
        'source_dataset': source_dataset,
    })
    meta['env_param'] = env
    return meta


def collect_a(root):
    frames = []
    for meta_path in sorted((root / 'SOPE').glob('*/train/Omni6DPose/*/*_meta.json')):
        meta = json.loads(meta_path.read_text())
        prefix = Path(str(meta_path)[:-len('_meta.json')])
        env = meta.get('env_param', {})
        # The camera/layout stratum is the leakage grouping unit for A.
        group = 'A_{n}_{gap}_{baf}_{d}_{elev}'.format(
            n=env.get('n_trays'), gap=env.get('gap_mm'), baf=env.get('baffle'),
            d=round(float(env.get('d_m', 0)), 1), elev=round(float(env.get('elev_deg', 0)), 0),
        )
        frames.append(Frame('A', prefix, 'train', meta_path.parts[-5], prefix.name,
                            group, len(meta.get('objects', {}))))
    return frames


def collect_b(root):
    frames = []
    for meta_path in sorted(root.glob('smt_ral7035_*/*/fixed_shelf/scene_*/*_meta.json')):
        meta = json.loads(meta_path.read_text())
        prefix = Path(str(meta_path)[:-len('_meta.json')])
        env = meta.get('env_param', {})
        source_split = str(env.get('split', meta_path.parts[-5]))
        camera_id = str(env['camera_id'])
        # Camera IDs are unique. Group by row/distance/hide setting and generation
        # shard to keep near-neighbour views on one side of the split.
        parts = camera_id.split('_')
        row_token = next((p for p in parts if p.startswith('row')), '')
        distance = str(env.get('distance_stratum', 'unknown'))
        hide = str(env.get('hide_ratio', 'unknown'))
        raw_frame = str(env.get('raw_frame', ''))
        shard = Path(raw_frame).parent.name if raw_frame else 'unknown'
        numeric = next((p for p in parts if p.isdigit()), camera_id)
        bucket = int(numeric) // 20 if numeric.isdigit() else 0
        group = f'B_{shard}_{row_token}_{distance}_{hide}_{bucket:04d}'
        frames.append(Frame('B', prefix, source_split, meta_path.parent.name,
                            prefix.name, group, len(meta.get('objects', []))))
    return frames


def assign_group_splits(frames, seed):
    group_frames = defaultdict(list)
    for frame in frames:
        group_frames[frame.group_id].append(frame)
    groups = sorted(group_frames)
    random.Random(seed).shuffle(groups)
    targets = {'train': round(len(frames) * 0.8), 'val': round(len(frames) * 0.1)}
    targets['test'] = len(frames) - targets['train'] - targets['val']
    counts = Counter()
    for group in groups:
        size = len(group_frames[group])
        deficits = {split: targets[split] - counts[split] for split in targets}
        split = max(deficits, key=lambda key: (deficits[key], key == 'train'))
        for frame in group_frames[group]:
            frame.split = split
        counts[split] += size
    return counts


def write_obj_meta(path):
    payload = {
        'class_list': [{'name': CANONICAL_CLASS, 'label': 0,
                        'instance_ids': [CANONICAL_OID], 'stat': {}}],
        'instance_dict': {CANONICAL_OID: {
            'object_id': CANONICAL_OID, 'source': 'smt_factory', 'name': CANONICAL_CLASS,
            'obj_path': 'assets/tray_z_normal_v2.obj',
            'tag': {'datatype': 'train', 'sceneChanger': False,
                    'symmetry': CANONICAL_SYMMETRY, 'materialOptions': ['uniform_pale_yellow'],
                    'upAxis': ['z']},
            'class_label': 0, 'class_name': CANONICAL_CLASS, 'dimensions': CANONICAL_DIMS,
        }},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def convert_obj(source, destination):
    conversion = b_canonical_rotation()
    output = []
    for line in source.read_text().splitlines():
        if line.startswith(('v ', 'vn ')):
            tag, *values = line.split()
            vector = np.asarray([float(v) for v in values[:3]])
            converted = conversion.T @ vector
            output.append(f'{tag} {converted[0]:.9f} {converted[1]:.9f} {converted[2]:.9f}')
        else:
            output.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(output) + '\n')


def build(args):
    source_a = args.source_root / 'train_set_blender_v2'
    source_b = args.source_root / 'source_ral7035_4000_20260812'
    destination = args.destination.resolve()
    temporary = destination.with_name(destination.name + '.building')
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        raise SystemExit(f'destination already exists: {destination}')
    temporary.mkdir(parents=True)

    frames_a = collect_a(source_a)
    frames_b = collect_b(source_b)
    frames = frames_a + frames_b
    counts = assign_group_splits(frames, args.seed)
    print('frames', len(frames), 'source', {'A': len(frames_a), 'B': len(frames_b)},
          'splits', dict(counts), flush=True)

    manifest_path = temporary / 'manifests/sample_manifest.jsonl'
    manifest_path.parent.mkdir(parents=True)
    split_objects = Counter()
    with manifest_path.open('w') as manifest:
        for index, frame in enumerate(frames, 1):
            dst_dir = temporary / 'SOPE' / frame.scene_id / frame.split / 'Omni6DPose' / frame.frame_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_prefix = dst_dir / frame.frame_id
            hashes = {}
            for suffix in FILE_TYPES:
                src = Path(str(frame.source_prefix) + '_' + suffix)
                dst = Path(str(dst_prefix) + '_' + suffix)
                if suffix == 'meta.json':
                    meta = transform_meta(json.loads(src.read_text()), frame.source_dataset)
                    dst.write_text(json.dumps(meta, indent=2))
                else:
                    os.link(src, dst)
                hashes[suffix] = sha256(dst)
            split_objects[frame.split] += frame.object_count
            manifest.write(json.dumps({
                'source_dataset': frame.source_dataset,
                'source_path': str(frame.source_prefix),
                'source_split': frame.source_split,
                'new_path': str(dst_prefix).replace(str(temporary), str(destination), 1),
                'split': frame.split,
                'group_id': frame.group_id,
                'scene_id': frame.scene_id,
                'frame_id': frame.frame_id,
                'n_objects': frame.object_count,
                'hashes': hashes,
                'canonical_rotation_applied': frame.source_dataset == 'B',
            }) + '\n')
            if index % 500 == 0:
                print('built', index, '/', len(frames), flush=True)

    write_obj_meta(temporary / 'Meta/obj_meta.json')
    convert_obj(source_b / 'assets/tray_scene_frame_centered.obj',
                temporary / 'assets/tray_z_normal_v2.obj')
    report = {
        'dataset': destination.name,
        'seed': args.seed,
        'frames': len(frames),
        'split_frames': dict(counts),
        'split_instances': dict(split_objects),
        'canonical_dimensions_m': CANONICAL_DIMS,
        'canonical_symmetry': CANONICAL_SYMMETRY,
        'canonical_frame': 'smt_tray_z_normal_v2',
        'source_frame_conversion': {'A': 'identity', 'B': 'R_old @ C_x_to_z'},
    }
    (temporary / 'reports').mkdir()
    (temporary / 'reports/dataset_report.json').write_text(json.dumps(report, indent=2))
    temporary.rename(destination)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path,
                        default=Path('/data/01-code/GenPose2/datasets'))
    parser.add_argument('--destination', type=Path,
                        default=Path('/data/quinn/smt/datasets/genpose2_smt_train_v2_20260813'))
    parser.add_argument('--seed', type=int, default=SEED)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
