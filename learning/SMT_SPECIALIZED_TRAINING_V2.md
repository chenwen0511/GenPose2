# SMT Tray GenPose2 V2

## Object contract

- Physical diameter: `0.1778 m`
- Physical thickness: `0.0080 m`
- Canonical local frame: tray normal is local `+Z`; the two in-plane axes are `X/Y`.
- Symmetry: continuous rotation around local `Z` (`z=any`). The normal direction is still observable; a 180-degree normal flip is not treated as equivalent.
- ScaleNet is disabled for the fixed-size object. `obj_meta.json` and the dataset metadata use the physical dimensions above.

## Dataset

Path: `/data/quinn/smt/datasets/genpose2_smt_train_v2_20260813`

- 5,200 frames in `SOPE/`
- Train/val/test frames: 4,160 / 520 / 520
- Train/val/test instances: 54,262 / 6,845 / 6,526
- 2,302 groups; no group crosses a split
- No exact RGB duplicate crosses a split
- B source conversion: `R_new = R_old @ C`, where `C` maps old local X thickness to new local Z thickness
- New mesh: `assets/tray_z_normal_v2.obj`

The previous v1 dataset is retained and is not modified.

## Loss treatment

GenPose2 ScoreNet uses score matching rather than direct pose regression. The training loader now samples a random continuous symmetry rotation for rows with exactly one `any` axis. This produces equivalent GT poses along the circular yaw orbit without changing the point cloud or translation. Non-symmetric rows are unchanged.

The evaluation path does not randomize the GT pose.

## Metrics

Primary SMT metrics:

- Normal-direction error: degrees between predicted and GT tray normal; yaw around local Z is ignored.
- Translation error: Euclidean distance in metres, also report the existing centimetre metric.
- ADD-S: mean closest-point distance in metres using the canonical OBJ.
- `ADD-S < 0.05D` and `ADD-S < 0.10D`, where `D=0.1778 m`.
- `normal < 10 deg AND translation < 2 cm`.

The existing calibrated SO(3) rotation metric remains for compatibility, but it is not the primary score for this circular tray. ADD/ordinary rotation error should not be used alone because arbitrary yaw is unobservable.

## Training entry point

```bash
cd /data/quinn/smt/GenPose2
bash scripts/smt/train_score.sh
```

The script runs v2 preflight, uses `--load_per_object`, enables `--symmetry_augment`, points evaluation to the canonical OBJ, and requires at least 14 GiB free GPU memory unless `ALLOW_BUSY_GPU=1` is explicitly set.

No long training job was launched during the repair. A CUDA single-batch ScoreNet forward and finite score loss were verified.
