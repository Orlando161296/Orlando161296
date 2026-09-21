#!/usr/bin/env python3
"""
portrait.py — turn a photo into a 1-bit dithered dot-matrix portrait and
compose it inside a terminal-style "VISUAL.MAP / SYSTEM.INFO" profile banner.

Own implementation: Floyd–Steinberg serpentine dithering, square-crop with a
focus point, and SVG output aggregated into horizontal 1px runs for a crisp
pixel/terminal look. Dark and light variants are emitted so the README can
swap them with <picture> + prefers-color-scheme.

Usage
-----
    python scripts/portrait.py photo.jpg
    python scripts/portrait.py photo.jpg --focus 0.5,0.42 --cols 300
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

# --- frame geometry (SVG units) -------------------------------------------
W, H = 1180, 610
VIS_X, VIS_Y, VIS_W, VIS_H = 35, 88, 418, 472        # left visual frame
INFO_X, INFO_Y, INFO_W, INFO_H = 474, 88, 672, 472   # right info frame
GRID_COLS, GRID_ROWS = 360, 408                      # 1-bit dither lattice
BG_THRESH = 185                                     # brightness above which = background
FONT = "JetBrains Mono,ui-monospace,SFMono-Regular,Consolas,monospace"

# --- themes ----------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#0A0F1E",
        "panel": "#0D1424",
        "panel2": "#101A30",
        "line": "#24344E",
        "muted": "#7E93B0",
        "text": "#DCE8F8",
        "portrait": "#7DD3FC",
        "chrome": "#22D3EE",
        "accent": "#34D399",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#EDF3F7",
        "line": "#CBD7E1",
        "muted": "#64748B",
        "text": "#14203A",
        "portrait": "#0284C7",
        "chrome": "#0891B2",
        "accent": "#059669",
        "shadow": "#AAB7C4",
    },
}

# --- config ----------------------------------------------------------------
# Edit these rows to change what SYSTEM.INFO shows. `value` is shown right-aligned.
ROWS = [
    ("Subject", "Orlando"),
    ("Role", "Systems Engineer"),
    ("Education", "M.Sc. Computer Science · AI"),
    ("Origin", "Venezuela"),
    ("Status", "Open to AI/ML roles"),
    ("ToolChain", "VS Code · Git · Docker · Linux"),
    ("Core.Lang", "Python · TypeScript · Rust"),
    ("Core.Backend", "NestJS · FastAPI · Node.js"),
    ("Core.Web", "JavaScript"),
    ("Core.ML", "TensorFlow · PyTorch · scikit-learn"),
    ("Core.Data", "pandas · NumPy · Jupyter"),
    ("Grid.GitHub", "Orlando161296"),
    ("Grid.LinkedIn", "/in/nando-rojas"),
]

PROMPT = "orlando@venezuela:~$"
NODE_LABEL = "caracas-01 · UTC-4"
MAP_COORDS = "10.48N · 66.90W"


# --------------------------------------------------------------------------- #
# image -> 1-bit lattice
# --------------------------------------------------------------------------- #

def square_crop(img: Image.Image, fx: float, fy: float, zoom: float = 1.0) -> Image.Image:
    """Crop to 1:1 around a focus point given in 0..1 image coordinates.

    `zoom` scales the crop side relative to min(width,height); values < 1 zoom
    in on the subject (tighter head-and-shoulders framing).
    """
    w, h = img.size
    side = min(w, h) * zoom
    left = min(max(fx * w - side / 2, 0), w - side)
    top = min(max(fy * h - side / 2, 0), h - side)
    return img.crop((round(left), round(top), round(left) + side, round(top) + side))


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd–Steinberg dither; True == lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    h, w = work.shape
    for y in range(h):
        ltr = y % 2 == 0
        xs = range(w) if ltr else range(w - 1, -1, -1)
        d = 1 if ltr else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + d
            if 0 <= nx < w:
                work[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    work[y + 1, x - d] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    work[y + 1, nx] += err * 1 / 16
    return out


def lattice_from_image(path: Path, cols: int, rows: int, focus, zoom: float, dark: bool) -> tuple[np.ndarray, Image.Image]:
    """Return a bool lattice (dots) and the prepared square crop preview.

    Dots follow the SUBJECT, not the background: a bright background is masked
    out, the histogram is equalised against the subject only, then in dark mode
    the lit pixels (the face) become the dots, and in light mode the selection
    inverts to an ink-on-paper stipple (hair/shadows draw as dots).
    """
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    crop = square_crop(img, *focus, zoom).resize((cols, rows), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(crop)
    g = np.asarray(gray, dtype=np.float32)

    # Detect background polarity from the border, then mask the subject.
    # A dark background means the subject is bright (logo/emblem); a bright
    # background means the subject is dark (a face photo).
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    bg = float(np.median(border))
    if bg < 128:
        thr = max(bg + 25.0, 20.0)
        mask = (g > thr).astype(np.float32)
    else:
        thr = min(bg - 25.0, 235.0)
        mask = (g < thr).astype(np.float32)
    mask_img = Image.fromarray(np.uint8(mask * 255), "L").filter(ImageFilter.GaussianBlur(1.5))
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    binmask = Image.fromarray(np.uint8((mask > 0.5) * 255), "L")

    # Equalise against the subject's own histogram, then recover detail edges.
    gray = ImageOps.equalize(gray, mask=binmask)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=1))

    bits = floyd_steinberg(np.asarray(gray, dtype=np.float32))
    active = bits if dark else ~bits
    active &= mask > 0.5
    return active, crop


# --------------------------------------------------------------------------- #
# svg building
# --------------------------------------------------------------------------- #

def num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def run_path(active: np.ndarray, ox: float, oy: float, cell: float) -> str:
    """Aggregate horizontal 1px runs of lit cells into compact SVG path data."""
    rows, cols = active.shape
    parts: list[str] = []
    for y in range(rows):
        x = 0
        while x < cols:
            if active[y, x]:
                x0 = x
                while x + 1 < cols and active[y, x + 1]:
                    x += 1
                px = ox + x0 * cell
                py = oy + y * cell
                parts.append(f"M{num(px)} {num(py)}h{num((x - x0 + 1) * cell)}")
            x += 1
    return "".join(parts)


# --------------------------------------------------------------------------- #
# morphing animation — face fades out and the dots travel to form AI logos
# --------------------------------------------------------------------------- #

TRAVELLER_COUNT = 1200
LOOP_SECONDS = 16.0
MORPH_TIMES = [0.0, 2.8, 3.9, 5.7, 6.8, 8.6, 9.7, 11.5, 12.6, 14.4, 16.0]
TRAVELLER_OPACITY = "0;0;1;1;1;1;1;1;1;1;0"   # dots hidden while the face shows
FACE_OPACITY = "1;1;0;0;0;0;0;0;0;0;1"        # face fades out while logos show


def make_logos(cols: int, rows: int) -> dict[str, Image.Image]:
    """Draw AI logo silhouettes on a (cols, rows) RGBA canvas."""
    cx, cy = cols // 2, rows // 2
    logos: dict[str, Image.Image] = {}

    # 1. neural network — 3-4-2 nodes, fully connected
    nn = Image.new("RGBA", (cols, rows), (0, 0, 0, 0))
    d = ImageDraw.Draw(nn)
    layers = [[(cols * 0.20, cy - 60), (cols * 0.20, cy), (cols * 0.20, cy + 60)],
              [(cols * 0.50, cy - 90), (cols * 0.50, cy - 30), (cols * 0.50, cy + 30), (cols * 0.50, cy + 90)],
              [(cols * 0.80, cy - 40), (cols * 0.80, cy + 40)]]
    for a in layers[0]:
        for b in layers[1]:
            d.line([a, b], fill=(0, 0, 0, 255), width=5)
    for b in layers[1]:
        for c in layers[2]:
            d.line([b, c], fill=(0, 0, 0, 255), width=5)
    for layer in layers:
        for (x, y) in layer:
            d.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(0, 0, 0, 255))
    logos["nn"] = nn

    # 2. sparkle — the four-point "AI" star
    sparkle = Image.new("RGBA", (cols, rows), (0, 0, 0, 0))
    d = ImageDraw.Draw(sparkle)
    pts = []
    for i in range(16):
        a = -math.pi / 2 + i * math.pi / 8
        r = 120 if i % 4 == 0 else (40 if i % 2 == 0 else 22)
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    d.polygon(pts, fill=(0, 0, 0, 255))
    logos["sparkle"] = sparkle

    # 3. code — the </> chevrons
    code = Image.new("RGBA", (cols, rows), (0, 0, 0, 0))
    d = ImageDraw.Draw(code)
    d.line([(cx + 18, cy - 78), (cx - 60, cy), (cx + 18, cy + 78)], fill=(0, 0, 0, 255), width=26, joint="curve")
    d.line([(cx - 18, cy - 78), (cx + 60, cy), (cx - 18, cy + 78)], fill=(0, 0, 0, 255), width=26, joint="curve")
    logos["code"] = code

    # 4. robot — head with antenna and hollow eyes
    robot = Image.new("RGBA", (cols, rows), (0, 0, 0, 0))
    d = ImageDraw.Draw(robot)
    d.rounded_rectangle((cx - 55, cy - 70, cx + 55, cy + 40), radius=18, fill=(0, 0, 0, 255))
    d.ellipse((cx - 28, cy - 32, cx - 8, cy - 12), fill=(0, 0, 0, 0))   # left eye cutout
    d.ellipse((cx + 8, cy - 32, cx + 28, cy - 12), fill=(0, 0, 0, 0))  # right eye cutout
    d.line([(cx, cy - 70), (cx, cy - 96)], fill=(0, 0, 0, 255), width=8)
    d.ellipse((cx - 9, cy - 105, cx + 9, cy - 87), fill=(0, 0, 0, 255))
    d.line([(cx - 20, cy + 18), (cx + 20, cy + 18)], fill=(0, 0, 0, 255), width=6)
    logos["robot"] = robot
    return logos


def sample_points(image: Image.Image, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `count` (x, y) points from a silhouette's alpha channel."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    if len(xs) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    idx = rng.choice(len(xs), count, replace=len(xs) < count)
    return np.column_stack((xs[idx], ys[idx])).astype(np.float32)


def face_points(active: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `count` (x, y) points from the dithered face lattice."""
    ys, xs = np.where(active)
    idx = rng.choice(len(xs), count, replace=len(xs) < count)
    return np.column_stack((xs[idx], ys[idx])).astype(np.float32)


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorder target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def animate_values(frames: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in frames)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{num(x)} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, size: float) -> float:
    return len(text) * size * 0.605


def render_banner(theme: str, active: np.ndarray, crop: Image.Image, morph_frames: list | None = None) -> str:
    t = THEMES[theme]

    # fit the lattice inside the visual panel with a small margin
    cell = min((VIS_W - 30) / GRID_COLS, (VIS_H - 40) / GRID_ROWS)
    grid_w = cell * GRID_COLS
    grid_h = cell * GRID_ROWS
    ox = VIS_X + (VIS_W - grid_w) / 2
    oy = VIS_Y + 32 + (VIS_H - 40 - grid_h) / 2

    pts = run_path(active, ox, oy, cell)
    keytimes = ";".join(f"{v / LOOP_SECONDS:.4f}" for v in MORPH_TIMES)
    travellers: list[str] = []
    if morph_frames is not None:
        svg_frames = [np.column_stack((ox + p[:, 0] * cell, oy + p[:, 1] * cell)) for p in morph_frames]
        n = len(svg_frames[0])
        for i in range(n):
            positions = animate_values(svg_frames, i)
            travellers.append(
                f'<circle r="{cell * 0.7:.2f}" fill="{t["chrome"]}">'
                f'<animateTransform attributeName="transform" type="translate" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" keyTimes="{keytimes}" values="{positions}"/>'
                f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" repeatCount="indefinite" keyTimes="{keytimes}" values="{TRAVELLER_OPACITY}"/></circle>'
            )

    # map grid, corner brackets and crosshair (drawn behind the portrait)
    gx0, gy0 = ox, oy
    gx1, gy1 = ox + grid_w, oy + grid_h
    grid_d = "".join(
        [f"M{num(ox + grid_w * i / 8)} {num(oy)}V{num(oy + grid_h)}" for i in range(1, 8)]
        + [f"M{num(ox)} {num(oy + grid_h * i / 8)}H{num(ox + grid_w)}" for i in range(1, 8)]
    )
    bkt = 12
    corners = (
        f"M{num(gx0)} {num(gy0 + bkt)}V{num(gy0)}H{num(gx0 + bkt)}"
        f"M{num(gx1 - bkt)} {num(gy0)}H{num(gx1)}V{num(gy0 + bkt)}"
        f"M{num(gx1)} {num(gy1 - bkt)}V{num(gy1)}H{num(gx1 - bkt)}"
        f"M{num(gx0 + bkt)} {num(gy1)}H{num(gx0)}V{num(gy1 - bkt)}"
    )
    cxm, cym = ox + grid_w / 2, oy + grid_h / 2
    cross = (
        f"M{num(cxm - 14)} {num(cym)}H{num(cxm - 4)}M{num(cxm + 4)} {num(cym)}H{num(cxm + 14)}"
        f"M{num(cxm)} {num(cym - 14)}V{num(cym - 4)}M{num(cxm)} {num(cym + 4)}V{num(cym + 14)}"
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
        '<title id="t">Orlando — live system profile</title>',
        '<desc id="d">Terminal profile banner with a 1-bit dithered portrait and system info.</desc>',
        "<defs>",
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/></filter>',
        f'<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" flood-opacity=".35"/>'
        '<feComposite in2="b" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<pattern id="scanlines" width="3" height="3" patternUnits="userSpaceOnUse"><rect width="3" height="1" fill="#000000" opacity=".05"/></pattern>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="{W - 26}" height="{H - 26}" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H{W - 13}" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/><circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="{W / 2}" y="43" text-anchor="middle" fill="{t["muted"]}" '
        f'font-family="{FONT}" font-size="13" letter-spacing=".4">profile.sh --live</text>',
        # ---- left visual frame ----
        f'<rect x="{VIS_X}" y="{VIS_Y}" width="{VIS_W}" height="{VIS_H}" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M{VIS_X} {VIS_Y + 36}H{VIS_X + VIS_W}" stroke="{t["line"]}"/>',
        f'<text x="{VIS_X + 14}" y="{VIS_Y + 23}" fill="{t["chrome"]}" font-family="{FONT}" font-size="13" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="{VIS_X + VIS_W - 15}" y="{VIS_Y + 23}" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" font-size="11">{GRID_COLS}×{GRID_ROWS} / 1-BIT</text>',
        f'<path d="{grid_d}" stroke="{t["line"]}" stroke-width="1" opacity=".5" shape-rendering="crispEdges"/>',
        f'<path d="{corners}" fill="none" stroke="{t["chrome"]}" stroke-width="1.5" opacity=".55"/>',
        f'<path d="{cross}" fill="none" stroke="{t["chrome"]}" stroke-width="1" opacity=".4"/>',
        f'<g shape-rendering="crispEdges"><path d="{pts}" fill="none" stroke="{t["portrait"]}" stroke-width="{cell:.3f}">'
        f'<animate attributeName="opacity" dur="{LOOP_SECONDS}s" repeatCount="indefinite" keyTimes="{keytimes}" values="{FACE_OPACITY}"/></path></g>',
        "".join(travellers),
        f'<text x="{VIS_X + 23}" y="{VIS_Y + VIS_H - 9}" fill="{t["muted"]}" font-family="{FONT}" font-size="10">PTS {int(active.sum()):05d} · FS/SERPENTINE</text>',
        f'<text x="{VIS_X + VIS_W - 15}" y="{VIS_Y + VIS_H - 9}" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" font-size="10">{MAP_COORDS}</text>',
        # ---- right info frame ----
        f'<rect x="{INFO_X}" y="{INFO_Y}" width="{INFO_W}" height="{INFO_H}" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M{INFO_X} {INFO_Y + 36}H{INFO_X + INFO_W}" stroke="{t["line"]}"/>',
        f'<text x="{INFO_X + 16}" y="{INFO_Y + 23}" fill="{t["chrome"]}" font-family="{FONT}" font-size="13" font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
        '<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A"><animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/></circle></g>',
        f'<text x="927" y="111" fill="#FF4D5A" font-family="{FONT}" font-size="12" font-weight="700">LIVE</text>',
        f'<rect x="982" y="94" width="146" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" stroke="{t["chrome"]}"/>',
        f'<text x="1055" y="111" text-anchor="middle" fill="{t["chrome"]}" font-family="{FONT}" font-size="14" font-weight="700">@Orlando161296</text>',
    ]

    value_right = INFO_X + INFO_W - 19.0
    row_y = INFO_Y + 65.0
    for label, value in ROWS:
        vl = text_width(value, 14)
        ll = text_width(label, 14)
        leader_start = INFO_X + 17 + ll + 12
        leader_end = value_right - vl - 12
        parts.extend([
            f'<text x="{INFO_X + 17}" y="{num(row_y)}" fill="{t["muted"]}" font-family="{FONT}" font-size="14">{html.escape(label)}</text>',
            f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" fill="{t["text"]}" font-family="{FONT}" font-size="14" textLength="{num(vl)}" lengthAdjust="spacingAndGlyphs">{html.escape(value)}</text>',
        ])
        row_y += 23

    parts.extend([
        f'<path d="M{INFO_X + 16} {INFO_Y + INFO_H - 40}H{INFO_X + INFO_W - 15}" stroke="{t["line"]}"/>',
        f'<text x="{INFO_X + 17}" y="{INFO_Y + INFO_H - 22}" fill="{t["accent"]}" font-family="{FONT}" font-size="12">{PROMPT}</text>',
        f'<text x="{num(INFO_X + 17 + text_width(PROMPT, 12) + 4)}" y="{INFO_Y + INFO_H - 22}" fill="{t["accent"]}" font-family="{FONT}" font-size="12"><animate attributeName="opacity" values="1;0;1" dur="1.06s" repeatCount="indefinite"/>█</text>',
        f'<circle cx="{num(INFO_X + INFO_W - 16 - text_width(NODE_LABEL, 11) - 12)}" cy="{INFO_Y + INFO_H - 25}" r="3" fill="{t["accent"]}"><animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/></circle>',
        f'<text x="{INFO_X + INFO_W - 16}" y="{INFO_Y + INFO_H - 22}" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" font-size="11">{NODE_LABEL}</text>',
        f'<rect width="{W}" height="{H}" fill="url(#scanlines)" pointer-events="none"/>',
        "</svg>",
    ])
    return "".join(parts)


def preview_html() -> str:
    """Tiny local preview page showing the dark banner only."""
    return """<!doctype html><html><head><meta charset="utf-8"><title>portrait preview</title>
<style>body{font-family:ui-monospace,monospace;background:#111;color:#eee;padding:24px}
h1{font-size:16px}.card{margin:16px 0;padding:16px;border:1px solid #333;border-radius:8px}
img{max-width:100%;height:auto;border-radius:4px}</style></head><body>
<h1>Banner preview (dark)</h1>
<div class="card"><img src="assets/banner-dark.svg"></div>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--focus", default="0.5,0.5", help="x,y focus as fractions of the source")
    ap.add_argument("--zoom", type=float, default=1.0, help="crop side as fraction of min(w,h)")
    ap.add_argument("--cols", type=int, default=GRID_COLS)
    ap.add_argument("--rows", type=int, default=GRID_ROWS)
    args = ap.parse_args()
    fx, fy = (float(v) for v in args.focus.split(","))

    ASSETS.mkdir(parents=True, exist_ok=True)

    crop_previews: dict[str, Image.Image] = {}
    lattices: dict[str, np.ndarray] = {}
    for theme in ("dark", "light"):
        active, crop = lattice_from_image(args.image, args.cols, args.rows, (fx, fy), args.zoom, dark=(theme == "dark"))
        lattices[theme] = active
        crop_previews[theme] = crop

    crop_previews["dark"].save(ASSETS / "source-crop.jpg", quality=90)

    # morphing frames (lattice coords), shared across themes
    logos = make_logos(GRID_COLS, GRID_ROWS)
    rng = np.random.default_rng(42)
    fp = face_points(lattices["dark"], TRAVELLER_COUNT, rng)
    nn = transport(fp, sample_points(logos["nn"], TRAVELLER_COUNT, rng))
    sparkle = transport(nn, sample_points(logos["sparkle"], TRAVELLER_COUNT, rng))
    code = transport(sparkle, sample_points(logos["code"], TRAVELLER_COUNT, rng))
    robot = transport(code, sample_points(logos["robot"], TRAVELLER_COUNT, rng))
    morph_frames = [fp, fp, nn, nn, sparkle, sparkle, code, code, robot, robot, fp]

    for theme in ("dark", "light"):
        svg = render_banner(theme, lattices[theme], crop_previews[theme], morph_frames)
        out = ASSETS / f"banner-{theme}.svg"
        out.write_text(svg, encoding="utf-8")
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.1f} KiB, {int(lattices[theme].sum())} pts, {TRAVELLER_COUNT} travellers")

    (ROOT / "preview.html").write_text(preview_html(), encoding="utf-8")
    print("preview.html written")


if __name__ == "__main__":
    main()
