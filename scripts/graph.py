#!/usr/bin/env python3
"""graph.py — generate a subtle animated network-graph texture band. Stdlib only.

Nodes drift slowly and pulse; a few edges carry a travelling dash ("data flow").
Everything stays faint so it never competes with the actual profile content.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

W, H = 1200, 240
COLOR = "#22D3EE"      # banner chrome cyan
CLUSTERS = 13


def gen(seed: int = 42) -> str:
    rng = random.Random(seed)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="animated network graph texture">'
    ]

    for _ in range(CLUSTERS):
        cx = rng.uniform(50, W - 50)
        cy = rng.uniform(30, H - 30)
        n = rng.randint(4, 7)
        nodes = []
        for _ in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(10, 42)
            nodes.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad))

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.38:
                    x1, y1 = nodes[i]
                    x2, y2 = nodes[j]
                    if rng.random() < 0.18:
                        parts.append(
                            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                            f'stroke="{COLOR}" stroke-width="0.7" stroke-dasharray="3 9" opacity="0.12">'
                            f'<animate attributeName="stroke-dashoffset" values="24;0" '
                            f'dur="{rng.uniform(2.0, 3.5):.1f}s" repeatCount="indefinite"/></line>'
                        )
                    else:
                        parts.append(
                            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                            f'stroke="{COLOR}" stroke-width="0.7" opacity="0.09"/>'
                        )

        for x, y in nodes:
            dx = rng.uniform(-6, 6)
            dy = rng.uniform(-6, 6)
            dur = rng.uniform(8, 15)
            pdur = rng.uniform(2.8, 4.6)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.1" fill="{COLOR}" opacity="0.16">'
                f'<animateTransform attributeName="transform" type="translate" '
                f'values="0 0;{dx:.1f} {dy:.1f};0 0" dur="{dur:.1f}s" repeatCount="indefinite" '
                f'calcMode="spline" keyTimes="0;0.5;1" keySplines="0.45 0 0.55 1;0.45 0 0.55 1"/>'
                f'<animate attributeName="opacity" values="0.08;0.45;0.08" '
                f'dur="{pdur:.1f}s" repeatCount="indefinite"/></circle>'
            )

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, default=Path("assets/graph.svg"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.write_text(gen(args.seed), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
