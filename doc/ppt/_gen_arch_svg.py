# -*- coding: utf-8 -*-
"""Generate GenPose2 architecture SVGs for the magazine PPT (indigo porcelain)."""
from pathlib import Path

OUT = Path(r"d:\04-work\13-byd\01-code\GenPose2\doc\ppt\images")
INK = "#0a1f3d"
PAPER = "#f1f3f5"
TINT = "#e4e8ec"
ACCENT = "#2a5a9e"
MUTED = "#5a6f8a"
LINE = "#0a1f3d"


def box(x, y, w, h, fill, stroke=LINE, sw=1.4, rx=4):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def txt(x, y, s, size=11, fill=INK, weight="500", anchor="middle",
        family="IBM Plex Mono, Noto Sans SC, sans-serif"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}">{s}</text>'
    )


def arrow(x1, y1, x2, y2, color=INK):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="1.6" marker-end="url(#arr)"/>'
    )


def overview_svg():
    W, H = 1280, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        "<defs>",
        f'<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{INK}"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
        txt(640, 36, "GenPose2 · Architecture (based on GenPose / arXiv:2306.10531)", 15, MUTED, "500"),
        txt(160, 68, "INPUT", 12, MUTED, "600"),
        txt(480, 68, "BACKBONE", 12, MUTED, "600"),
        txt(820, 68, "DIFFUSION HEADS", 12, MUTED, "600"),
        txt(1120, 68, "OUTPUT", 12, MUTED, "600"),
        # Inputs
        box(40, 100, 200, 70, "#fff", sw=1.2),
        txt(140, 128, "ROI RGB", 13, INK, "600"),
        txt(140, 148, "H×W×3", 10, MUTED),
        box(40, 190, 200, 70, "#fff", sw=1.2),
        txt(140, 218, "Instance Points", 13, INK, "600"),
        txt(140, 238, "N×3  (N=1024)", 10, MUTED),
        box(40, 280, 200, 58, TINT, sw=1.0),
        txt(140, 305, "Camera / Mask", 12, MUTED, "500"),
        txt(140, 322, "→ back-project PC", 10, MUTED),
        # Backbone
        box(300, 95, 240, 88, "#fff", ACCENT, 2.0),
        txt(420, 125, "DINOv2 ViT-S/14", 13, INK, "700"),
        txt(420, 145, "frozen · dino_dim=384", 10, MUTED),
        txt(420, 163, "pointwise / global", 10, MUTED),
        box(300, 210, 240, 100, "#fff", ACCENT, 2.0),
        txt(420, 240, "PointNet++ Fus", 13, INK, "700"),
        txt(420, 260, "geom (+ DINO patches)", 10, MUTED),
        txt(420, 278, "→ pts_feat  [B, 1024]", 11, ACCENT, "600"),
        arrow(240, 135, 300, 135),
        arrow(240, 225, 300, 250),
        arrow(420, 183, 420, 210),
        # feat bus
        box(580, 230, 70, 60, ACCENT, ACCENT, 0),
        txt(615, 255, "feat", 11, PAPER, "700"),
        txt(615, 272, "bus", 10, PAPER),
        arrow(540, 260, 580, 260),
        # ScoreNet
        box(700, 90, 280, 155, "#fff", INK, 1.8),
        txt(840, 115, "PoseScoreNet  Φθ", 14, INK, "700"),
        txt(840, 136, "t_enc 128 · pose_enc 256", 10, MUTED),
        txt(840, 154, "heads: Rx / Ry / T  (9-D)", 10, MUTED),
        txt(840, 172, "score = f / σ(t)", 10, MUTED),
        txt(840, 195, "ODE / PC sampler × K=50", 11, ACCENT, "600"),
        txt(840, 218, "→ candidates [B,K,9]", 11, ACCENT, "600"),
        # EnergyNet
        box(700, 270, 280, 130, "#fff", INK, 1.8),
        txt(840, 295, "PoseEnergyNet  Ψφ", 14, INK, "700"),
        txt(840, 316, "same backbone / heads", 10, MUTED),
        txt(840, 334, "Ψ = ⟨p, Φ⟩  (IP mode)", 10, MUTED),
        txt(840, 352, "rank · retain_ratio=0.4", 10, MUTED),
        txt(840, 375, "→ quat mean (+ DBSCAN)", 11, ACCENT, "600"),
        # ScaleNet
        box(700, 430, 280, 110, "#fff", INK, 1.8),
        txt(840, 455, "ScaleNet", 14, INK, "700"),
        txt(840, 476, "axes_enc + pts_feat", 10, MUTED),
        txt(840, 494, "MLP → size_3d [l,w,h]", 10, MUTED),
        txt(840, 516, "~5 MB · optional / fixed", 11, ACCENT, "600"),
        arrow(650, 250, 700, 170),
        arrow(650, 260, 700, 320),
        arrow(650, 270, 700, 470),
        arrow(840, 245, 840, 270),
        txt(900, 262, "K candidates", 9, MUTED, "500", "start"),
        arrow(840, 400, 840, 430),
        txt(900, 418, "final R,T", 9, MUTED, "500", "start"),
        # Outputs
        box(1040, 130, 200, 70, TINT, ACCENT, 1.6),
        txt(1140, 158, "Pose candidates", 12, INK, "600"),
        txt(1140, 178, "[B, K, 9]", 11, MUTED),
        box(1040, 290, 200, 70, TINT, ACCENT, 1.6),
        txt(1140, 318, "Final 6D pose", 12, INK, "600"),
        txt(1140, 338, "R ∈ SO(3), T ∈ R³", 11, MUTED),
        box(1040, 450, 200, 70, TINT, ACCENT, 1.6),
        txt(1140, 478, "size_3d", 12, INK, "600"),
        txt(1140, 498, "[l, w, h]", 11, MUTED),
        arrow(980, 165, 1040, 165),
        arrow(980, 325, 1040, 325),
        arrow(980, 485, 1040, 485),
        # Bottom note
        box(40, 560, 1200, 120, "#fff", MUTED, 1.0),
        txt(640, 590, "Paper Fig.2 pipeline · GenPose2 extensions", 12, MUTED, "600"),
        txt(640, 615, "Train: DSM score-matching on Φθ  →  energy DSM on Ψφ (IP)  →  Scale regression", 11, INK),
        txt(640, 640, "Infer: noise → ODE reverse (Score) → energy rank & aggregate → Scale (or hard-code size)", 11, INK),
        txt(640, 662, "Pose rep: rot_matrix 9-D = [Rx|Ry|T] continuous 6-D rotation + translation", 10, MUTED),
        "</svg>",
    ]
    (OUT / "arch-overview.svg").write_text("\n".join(parts), encoding="utf-8")
    print("wrote arch-overview.svg")


def detail_svg():
    W, H = 1280, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        "<defs>",
        f'<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{INK}"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
        txt(640, 34, "PoseScoreNet / PoseEnergyNet · Internal Block Diagram", 15, MUTED, "500"),
        # Left
        box(30, 60, 360, 420, "#fff", ACCENT, 1.8),
        txt(210, 88, "Observation Encoder", 13, ACCENT, "700"),
        box(55, 110, 310, 55, TINT),
        txt(210, 132, "ROI RGB → DINOv2", 12, INK, "600"),
        txt(210, 150, "ViT-S/14 · 384-d · frozen", 10, MUTED),
        box(55, 185, 310, 55, TINT),
        txt(210, 207, "Points [B,1024,3]", 12, INK, "600"),
        txt(210, 225, "(+ pointwise DINO feats)", 10, MUTED),
        arrow(210, 240, 210, 265),
        box(55, 265, 310, 70, "#fff", INK, 1.5),
        txt(210, 292, "PointNet++ MSG Fus", 13, INK, "700"),
        txt(210, 312, "multi-scale set abstraction", 10, MUTED),
        arrow(210, 335, 210, 360),
        box(55, 360, 310, 90, ACCENT, ACCENT, 0),
        txt(210, 395, "pts_feat", 16, PAPER, "700"),
        txt(210, 420, "[B, 1024]", 12, PAPER),
        # Middle
        box(430, 60, 300, 420, "#fff", INK, 1.5),
        txt(580, 88, "Condition Encoders", 13, INK, "700"),
        box(455, 115, 250, 75, TINT),
        txt(580, 140, "t_encoder", 12, INK, "600"),
        txt(580, 158, "Gaussian Fourier → 128", 10, MUTED),
        txt(580, 175, "t ∈ (ε, 1]", 10, MUTED),
        box(455, 210, 250, 75, TINT),
        txt(580, 235, "pose_encoder", 12, INK, "600"),
        txt(580, 253, "Linear 9→256→256", 10, MUTED),
        txt(580, 270, "noisy pose p(t)", 10, MUTED),
        box(455, 305, 250, 55, TINT),
        txt(580, 328, "optional rgb_feat", 11, MUTED, "500"),
        txt(580, 346, "dino=global only", 10, MUTED),
        arrow(580, 360, 580, 390),
        box(455, 390, 250, 60, "#fff", ACCENT, 1.5),
        txt(580, 415, "concat", 12, INK, "700"),
        txt(580, 435, "128+256+1024[+D]", 10, MUTED),
        arrow(365, 405, 455, 415),
        # Right
        box(780, 60, 460, 420, "#fff", INK, 1.5),
        txt(1010, 88, "Regression Heads · Rx_Ry_and_T", 13, INK, "700"),
        box(810, 115, 400, 70, TINT),
        txt(1010, 140, "RotHead  Rx", 13, INK, "600"),
        txt(1010, 160, "MLP → 3  (first column of R)", 10, MUTED),
        box(810, 200, 400, 70, TINT),
        txt(1010, 225, "RotHead  Ry", 13, INK, "600"),
        txt(1010, 245, "MLP → 3  (second column of R)", 10, MUTED),
        box(810, 285, 400, 70, TINT),
        txt(1010, 310, "TransHead  T", 13, INK, "600"),
        txt(1010, 330, "MLP → 3  (translation)", 10, MUTED),
        arrow(1010, 355, 1010, 380),
        box(810, 380, 400, 75, ACCENT, ACCENT, 0),
        txt(1010, 410, "raw fθ ∈ R⁹   →   score = fθ / σ(t)", 13, PAPER, "700"),
        txt(1010, 435, "Energy: Ψ = ⟨p_rot, s_rot⟩ + ⟨p_t, s_t⟩", 11, PAPER),
        arrow(705, 420, 810, 250),
        # Bottom
        box(30, 510, 1220, 180, "#fff", MUTED, 1.0),
        txt(640, 540, "Training vs Inference", 13, MUTED, "700"),
        box(55, 560, 560, 105, TINT, MUTED, 0.8),
        txt(335, 585, "Train (DSM)", 12, INK, "700"),
        txt(335, 608, "perturb GT pose with σ(t) · match ∇ log p", 11, INK),
        txt(335, 630, "Score: Φθ ≈ (p0 − p_t) / σ²(t)", 11, INK),
        txt(335, 652, "Energy: same DSM on ∇Ψ ; IP param Ψ=⟨p,Φ⟩", 11, INK),
        box(650, 560, 575, 105, TINT, MUTED, 0.8),
        txt(937, 585, "Infer (Score path)", 12, INK, "700"),
        txt(937, 608, "p ~ N(0, σ²max I)  →  PF-ODE / PC reverse", 11, INK),
        txt(937, 630, "RK45 · call score each step · repeat K=50", 11, INK),
        txt(937, 652, "then Energy rank → mean pool → Scale", 11, INK),
        "</svg>",
    ]
    (OUT / "arch-detail.svg").write_text("\n".join(parts), encoding="utf-8")
    print("wrote arch-detail.svg")


def scale_energy_svg():
    W, H = 1280, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        "<defs>",
        f'<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{INK}"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
        txt(640, 32, "Energy Aggregation + ScaleNet", 15, MUTED, "500"),
        box(40, 55, 1200, 215, "#fff", INK, 1.4),
        txt(640, 85, "EnergyNet Aggregation · paper §3.2", 13, ACCENT, "700"),
    ]
    xs = [70, 310, 550, 790, 1030]
    labels = [
        ("① Candidates", "K=50 from Score"),
        ("② Energies Ψ", "IP ⟨p, sθ⟩"),
        ("③ Rank & Cut", "retain 40%"),
        ("④ Aggregate", "quat mean + T"),
        ("⑤ Final Pose", "R̂ , T̂"),
    ]
    for i, (x, (t1, t2)) in enumerate(zip(xs, labels)):
        w = 200 if i < 4 else 170
        box_fill = ACCENT if i == 4 else TINT
        tc = PAPER if i == 4 else INK
        parts.append(box(x, 115, w, 110, box_fill, ACCENT if i == 4 else MUTED, 1.2 if i < 4 else 0))
        parts.append(txt(x + w / 2, 155, t1, 13, tc, "700"))
        parts.append(txt(x + w / 2, 185, t2, 11, MUTED if i < 4 else PAPER))
        if i < 4:
            parts.append(arrow(x + w + 2, 170, xs[i + 1] - 2, 170))

    parts += [
        box(40, 300, 1200, 185, "#fff", INK, 1.4),
        txt(640, 330, "ScaleNet · size regression (GenPose2)", 13, ACCENT, "700"),
        box(80, 360, 220, 90, TINT),
        txt(190, 395, "pts_feat [B,1024]", 12, INK, "600"),
        txt(190, 418, "(reuse Score encoder)", 10, MUTED),
        box(360, 360, 220, 90, TINT),
        txt(470, 395, "axes R̂ [B,3,3]", 12, INK, "600"),
        txt(470, 418, "encode_axes → 256", 10, MUTED),
        box(640, 360, 220, 90, TINT),
        txt(750, 395, "Fusion MLP", 12, INK, "600"),
        txt(750, 418, "→ zero-init Linear", 10, MUTED),
        box(920, 360, 280, 90, ACCENT, ACCENT, 0),
        txt(1060, 395, "size_3d [l, w, h]", 14, PAPER, "700"),
        txt(1060, 420, "or hard-code if fixed", 11, PAPER),
        arrow(300, 405, 360, 405),
        arrow(580, 405, 640, 405),
        arrow(860, 405, 920, 405),
        "</svg>",
    ]
    (OUT / "arch-energy-scale.svg").write_text("\n".join(parts), encoding="utf-8")
    print("wrote arch-energy-scale.svg")


if __name__ == "__main__":
    overview_svg()
    detail_svg()
    scale_energy_svg()
