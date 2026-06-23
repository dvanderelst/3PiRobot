"""Architecture overview of the inverse model.

Instantiates SonarSlicesUQ_Wall3 (the deployed B architecture: one shared
3-output wall head, no z_sym), reads its real layer dimensions, and renders a
Graphviz data-flow diagram that shows the shared dual-ear trunk and the
symmetric / antisymmetric head wiring (the part `print(model)` hides).

This is NOT a matplotlib figure -- it shells out to the `dot` binary -- so it
does not use style.py. Tunables are the constants below.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_network.py

Outputs <NAME>.{dot,png,svg,pdf} into OUT.
"""

import subprocess
import sys

from paths import CONTROL, IMAGE_RESOURCES, IMAGES

sys.path.insert(0, str(CONTROL))
from Library.SonarModel import SonarSlicesUQ_Wall3  # noqa: E402

# ---- tunables -------------------------------------------------------------
CFG = dict(samples=200, conv_channels=[8, 16], conv_kernel=7,
           pool_out=8, fc_hidden=32, head_hidden=16, n_classes=3)
OUT = IMAGE_RESOURCES         # working files (.dot, .png, .svg) live here
PDF_OUT = IMAGES              # the paper-includable PDF lands in images/
NAME = "fig_network"
FONT = "Times"                # serif to match the paper; "Helvetica" for sans
RANKDIR = "LR"               # "LR" (left-right) or "TB" (top-bottom)
TRUNK_FILL, OUT_FILL = "#e8eefc", "#e6f5e6"
CLASS_NAMES = "wall / pole / none"
FORMATS = ("png", "svg", "pdf")
# ---------------------------------------------------------------------------

TEMPLATE = r'''digraph G {{
  rankdir={rankdir}; bgcolor="white";
  node [shape=box, style="rounded", fontsize=10, fontname="{font}"];
  edge [fontsize=8, fontname="{font}"]; graph [fontname="{font}"];

  subgraph cluster_in {{ label="Input (one emission)"; style=dashed; color=gray;
    L [label="Left ear\nenvelope ({samples})"]; R [label="Right ear\nenvelope ({samples})"]; }}

  trunk [label="{trunk}", style="rounded,filled", fillcolor="{trunk_fill}"];
  L -> trunk; R -> trunk;
  zL [shape=ellipse, label="z_L ({fc})"]; zR [shape=ellipse, label="z_R ({fc})"];
  trunk -> zL; trunk -> zR;

  comb [label="Combine\nz_LR=[z_L,z_R]   z_RL=[z_R,z_L]"];
  zL -> comb; zR -> comb;

  subgraph cluster_heads {{ label="Heads (Linear→{hh}→ReLU→Linear)"; style=dashed; color=gray;
    wall [label="Wall head\n(shared 3-out; in: z_LR and z_RL)\nleft/right swap, center average"];
    cls  [label="Class head\n(in: z_LR, z_RL → average)"];
    pole [label="Pole-az heads\nμ: ½(z_LR−z_RL)  antisym.\nlogσ²: ½(z_LR+z_RL)  sym."];
  }}
  comb -> wall; comb -> cls; comb -> pole;

  subgraph cluster_out {{ label="Outputs"; style=dashed; color=gray;
    ocls  [label="Class logits\n{classes}", style="rounded,filled", fillcolor="{out_fill}"];
    owall [label="Wall depth profile\nleft, center, right\n(mean + logσ² each)", style="rounded,filled", fillcolor="{out_fill}"];
    opole [label="Pole azimuth\n(mean + logσ²)", style="rounded,filled", fillcolor="{out_fill}"];
  }}
  wall -> owall; cls -> ocls; pole -> opole;
}}'''


def build_dot(m):
    convs = [l for l in m.encoder if l.__class__.__name__ == "Conv1d"]
    ch = " → ".join([str(convs[0].in_channels)] + [str(c.out_channels) for c in convs])
    pool = m.pool.output_size
    pool = pool[0] if isinstance(pool, (tuple, list)) else pool
    fc_in, fc_out = m.fc[0].in_features, m.fc[0].out_features
    trunk = (f"Shared trunk\\nConv1d {ch} (k={convs[0].kernel_size[0]}, ReLU)\\n"
             f"AdaptiveAvgPool → {pool}\\nflatten → Linear {fc_in}→{fc_out} (ReLU)")
    return TEMPLATE.format(
        rankdir=RANKDIR, font=FONT, samples=CFG["samples"], trunk=trunk,
        trunk_fill=TRUNK_FILL, fc=fc_out, hh=m.wall_mean_head[0].out_features,
        classes=CLASS_NAMES, out_fill=OUT_FILL)


def main():
    m = SonarSlicesUQ_Wall3(**CFG, symmetric=True)
    dot = build_dot(m)
    (OUT / f"{NAME}.dot").write_text(dot)
    for fmt in FORMATS:
        dest = PDF_OUT if fmt == "pdf" else OUT
        subprocess.run(["dot", f"-T{fmt}", "-o", str(dest / f"{NAME}.{fmt}")],
                       input=dot, text=True, check=True)
    n = sum(p.numel() for p in m.parameters())
    print(f"[network] {n:,} params -> {OUT}/{NAME}.{{dot,png,svg}} + {PDF_OUT}/{NAME}.pdf")


if __name__ == "__main__":
    main()
