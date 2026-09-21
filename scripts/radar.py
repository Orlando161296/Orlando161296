#!/usr/bin/env python3
"""radar.py — render a radar/spider chart as a standalone SVG. Stdlib only.

Usage:
    python scripts/radar.py --data assets/skills.json -o assets/radar-skills.svg
    python scripts/radar.py --data assets/langmix.json -o assets/radar-langs.svg --values

Data shape:
    {"title": "Skill Radar", "axes": [{"label": "Python", "value": 82}, ...]}

The viewBox is sized around the labels so nothing gets clipped, and the data
polygon grows in with a SMIL scale animation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FONT = "JetBrains Mono,ui-monospace,SFMono-Regular,Consolas,monospace"

# Colours matched to the banner's dark theme.
C = {
    "grid": "#24344E",
    "spoke": "#24344E",
    "label": "#DCE8F8",
    "value": "#7E93B0",
    "title": "#22D3EE",
    "fill": "#22D3EE",
    "stroke": "#22D3EE",
    "vertex": "#7DD3FC",
}

LBL, VAL, TTL = 13, 11, 15


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s: str, size: float) -> float:
    return len(s) * size * 0.62


def ring(radius: float, n: int, start: float = -math.pi / 2):
    return [
        (radius * math.cos(start + i * 2 * math.pi / n),
         radius * math.sin(start + i * 2 * math.pi / n))
        for i in range(n)
    ]


def render(title, axes, size=440, rings=4, show_values=True, animate=True):
    n = len(axes)
    r = size / 2 - 8
    gap = 20
    vals = [max(0.0, min(100.0, float(v))) for _, v in axes]
    outer = ring(r, n)

    labels = []
    for i, (label, _) in enumerate(axes):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        cosv, sinv = math.cos(ang), math.sin(ang)
        lx, ly = (r + gap) * cosv, (r + gap) * sinv
        anchor = "middle" if abs(cosv) < 0.25 else ("start" if cosv > 0 else "end")
        dy = 4 if abs(sinv) < 0.25 else (14 if sinv > 0 else -5)
        labels.append((lx, ly + dy, anchor, label, vals[i]))

    minx = maxx = -r
    miny = maxy = r
    for lx, ly, anchor, label, v in labels:
        w = max(tw(label, LBL), tw(f"{v:g}", VAL) if show_values else 0.0)
        if anchor == "start":
            x0, x1 = lx, lx + w
        elif anchor == "end":
            x0, x1 = lx - w, lx
        else:
            x0, x1 = lx - w / 2, lx + w / 2
        y0 = ly - LBL
        y1 = ly + 4 + (VAL + 4 if show_values else 0)
        minx, maxx = min(minx, x0), max(maxx, x1)
        miny, maxy = min(miny, y0), max(maxy, y1)

    pad = 10
    title_h = TTL + 14 if title else 0
    W = round((maxx - minx) + 2 * pad)
    H = round((maxy - miny) + 2 * pad + title_h)
    ox, oy = -minx + pad, -miny + pad + title_h

    if title:
        need = round(tw(title, TTL) + 2 * pad)
        if need > W:
            ox += (need - W) / 2
            W = need

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" aria-label="{esc(title) or "radar chart"}" '
        f'font-family="{FONT}">'
    ]
    if title:
        parts.append(
            f'<text x="{W / 2:.1f}" y="{pad + TTL:.0f}" text-anchor="middle" '
            f'font-size="{TTL}" font-weight="700" fill="{C["title"]}">{esc(title)}</text>'
        )
    parts.append(f'<g transform="translate({ox:.1f},{oy:.1f})">')

    for k in range(rings, 0, -1):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in ring(r * k / rings, n))
        parts.append(
            f'<polygon points="{d}" fill="none" stroke="{C["grid"]}" '
            f'stroke-width="1" opacity="{0.35 + 0.5 * k / rings:.2f}"/>'
        )

    for x, y in outer:
        parts.append(
            f'<line x1="0" y1="0" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{C["spoke"]}" stroke-width="1"/>'
        )

    shape = [(px * v / 100, py * v / 100) for (px, py), v in zip(outer, vals)]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in shape)
    parts.append("<g>")
    if animate:
        parts.append(
            '<animateTransform attributeName="transform" type="scale" '
            'values="0.04;1" dur="1.1s" calcMode="spline" keyTimes="0;1" '
            'keySplines="0.22 1 0.36 1" fill="freeze"/>'
        )
    parts.append(
        f'<polygon points="{d}" fill="{C["fill"]}" fill-opacity="0.22" '
        f'stroke="{C["stroke"]}" stroke-width="2.5" stroke-linejoin="round"/>'
    )
    for x, y in shape:
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{C["vertex"]}" '
            f'stroke="{C["stroke"]}" stroke-width="1.2"/>'
        )
    parts.append("</g>")

    for lx, ly, anchor, label, v in labels:
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="{LBL}" font-weight="600" fill="{C["label"]}">{esc(label)}</text>'
        )
        if show_values:
            parts.append(
                f'<text x="{lx:.1f}" y="{ly + VAL + 4:.1f}" text-anchor="{anchor}" '
                f'font-size="{VAL}" fill="{C["value"]}">{v:g}</text>'
            )

    parts.append("</g></svg>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("assets/skills.json"))
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=440)
    ap.add_argument("--rings", type=int, default=4)
    ap.add_argument("--values", action="store_true", help="print the number per axis")
    ap.add_argument("--no-animate", dest="animate", action="store_false")
    args = ap.parse_args()

    d = json.loads(args.data.read_text(encoding="utf-8"))
    title = d.get("title", "")
    axes = [(a["label"], float(a["value"])) for a in d["axes"]]
    if len(axes) < 3:
        raise SystemExit("a radar chart needs at least 3 axes")

    svg = render(title, axes, args.size, args.rings, args.values, args.animate)
    args.out.write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}  ({len(axes)} axes)")


if __name__ == "__main__":
    main()
