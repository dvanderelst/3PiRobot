#!/usr/bin/env python3
"""
Policy assessment script: visualise the weights of a saved HistoryNNPolicy.
"""

# ── Settings ──────────────────────────────────────────────────────────────────
GENERATION     = "last"             # integer generation number, "last", or None for best_policy.json
POLICY_DIR     = "Policy"           # where training saved the policy JSON files
ASSESSMENT_DIR = "PolicyAssessment" # where plots are written (created if needed)

# Trajectory assessment
TRAIN_SESSIONS = ["sessionB02", "sessionB03", "sessionB04", "sessionB05"]
N_TRIALS       = 6    # episodes per session
MAX_STEPS      = 150  # steps per episode
SEED           = 42
# ─────────────────────────────────────────────────────────────────────────────

import glob
import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# ── Feature labels for the 6-element history row ──────────────────────────────
FEAT_NAMES  = ["iid", "dist", "rot1", "rot2", "drive", "blk"]
FEAT_COLORS = ["#e07b54", "#5b9bd5", "#70ad47", "#ffc000", "#7030a0", "#808080"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_policy_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def reconstruct_params(data: dict):
    """Slice the flat genome back into weight matrices.

    Returns:
        params : list of 10 np.ndarray (W1, b1, W2a, b2a, W3a, b3a, W2b, b2b, W3b, b3b)
        arch   : dict with h1, h2, history_len, in_dim, max_rotate1_deg, max_rotate2_deg
    """
    history_len = int(data["history_len"])
    h1, h2 = data["hidden_sizes"]
    in_dim = history_len * 6

    shapes = [
        (h1, in_dim), (h1,),       # shared encoder  W1, b1
        (h2, h1),     (h2,),       # head1 hidden    W2a, b2a
        (1,  h2),     (1,),        # head1 output    W3a, b3a
        (h2, h1 + 2), (h2,),       # head2 hidden    W2b, b2b  (+2 = iid_n, dist_n injected)
        (1,  h2),     (1,),        # head2 output    W3b, b3b
    ]
    genome = np.array(data["genome"], dtype=np.float32)
    params, off = [], 0
    for s in shapes:
        n = int(np.prod(s))
        params.append(genome[off: off + n].reshape(s))
        off += n

    arch = dict(
        history_len=history_len, h1=h1, h2=h2, in_dim=in_dim,
        max_rotate1_deg=data["max_rotate1_deg"],
        max_rotate2_deg=data["max_rotate2_deg"],
    )
    return params, arch


# ── Plot helpers ───────────────────────────────────────────────────────────────

def heatmap(ax, mat: np.ndarray, row_labels, col_labels, title: str, cmap="RdBu_r"):
    vmax = float(np.abs(mat).max()) or 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def bar_chart(ax, vec: np.ndarray, x_labels, title: str, ylabel: str = "weight"):
    colors = ["#d73027" if v >= 0 else "#4575b4" for v in vec]
    ax.bar(range(len(vec)), vec, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(vec)))
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")


def effective_sensitivity(W_out, W_hid, W_enc):
    """Linear-approximation path: W_out (1,h2) @ W_hid (h2,h1) @ W_enc (h1,in_dim) → (in_dim,)."""
    return (W_out @ W_hid @ W_enc).reshape(-1)


# ── Main visualisation ────────────────────────────────────────────────────────

def plot_weights(params, arch: dict, suptitle: str, output_path: str) -> None:
    W1, b1, W2a, b2a, W3a, b3a, W2b, b2b, W3b, b3b = params
    hl = arch["history_len"]
    h1, h2 = arch["h1"], arch["h2"]

    # Column labels for W1: "t-4·iid", "t-4·dist", … "t0·blk"
    col_w1 = []
    for s in range(hl):
        lag = s - (hl - 1)          # t-4 … t0
        tag = "t0" if lag == 0 else f"t{lag}"
        col_w1 += [f"{tag}·{fn}" for fn in FEAT_NAMES]

    row_h1 = [f"u{i}" for i in range(h1)]
    row_h2 = [f"u{i}" for i in range(h2)]
    step_labels = [f"t{s - (hl-1)}" if s < hl - 1 else "t0" for s in range(hl)]

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.65, wspace=0.45)

    # ── Row 0: Shared encoder W1 full heatmap + bias ──────────────────────────
    ax = fig.add_subplot(gs[0, :3])
    heatmap(ax, W1, row_h1, col_w1, f"Shared encoder  W1  ({h1}×{in_dim(arch)})")

    ax = fig.add_subplot(gs[0, 3])
    bar_chart(ax, b1, row_h1, f"Shared encoder  b1  ({h1},)", ylabel="bias")

    # ── Row 1: W1 feature importance & temporal importance ────────────────────
    W1_r = np.abs(W1).reshape(h1, hl, 6)   # (units, steps, features)

    ax = fig.add_subplot(gs[1, :2])
    feat_imp = W1_r.mean(axis=(0, 1))       # average over units & steps → (6,)
    bars = ax.bar(FEAT_NAMES, feat_imp, color=FEAT_COLORS,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, feat_imp):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("mean |W1|", fontsize=7)
    ax.set_title("W1 feature importance  (avg over units & steps)",
                 fontsize=9, fontweight="bold")

    ax = fig.add_subplot(gs[1, 2:])
    step_imp = W1_r.mean(axis=(0, 2))       # average over units & features → (steps,)
    bars2 = ax.bar(step_labels, step_imp, color="#5b9bd5",
                   edgecolor="black", linewidth=0.5)
    for b, v in zip(bars2, step_imp):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("mean |W1|", fontsize=7)
    ax.set_title("W1 temporal importance  (avg over units & features)",
                 fontsize=9, fontweight="bold")

    # ── Row 2: Head 1 — rotate1 (look direction) ──────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    heatmap(ax, W2a, row_h2, row_h1, f"Head1 hidden  W2a  ({h2}×{h1})")

    ax = fig.add_subplot(gs[2, 2])
    bar_chart(ax, W3a.reshape(-1), row_h2, f"Head1 output  W3a  (1×{h2})")

    ax = fig.add_subplot(gs[2, 3])
    eff1 = effective_sensitivity(W3a, W2a, W1).reshape(hl, 6)
    heatmap(ax, eff1, step_labels, FEAT_NAMES,
            "Head1 effective sensitivity\n(W3a·W2a·W1, linear approx)")

    # ── Row 3: Head 2 — rotate2 (body turn) ──────────────────────────────────
    col_w2b = row_h1 + ["iid★", "dist★"]   # last 2 columns = injected measurement
    ax = fig.add_subplot(gs[3, :2])
    heatmap(ax, W2b, row_h2, col_w2b,
            f"Head2 hidden  W2b  ({h2}×{h1+2})  ★=injected current measurement")

    ax = fig.add_subplot(gs[3, 2])
    bar_chart(ax, W3b.reshape(-1), row_h2, f"Head2 output  W3b  (1×{h2})")

    ax = fig.add_subplot(gs[3, 3])
    # Sensitivity of head2 to the TWO injected features (iid_n, dist_n)
    eff2_inj = (W3b @ W2b[:, h1:]).reshape(-1)     # (2,)
    colors = ["#e07b54", "#5b9bd5"]
    ax.bar(["iid★", "dist★"], eff2_inj, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    for i, (lbl, v) in enumerate(zip(["iid★", "dist★"], eff2_inj)):
        ax.text(i, v, f"{v:.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title("Head2 sensitivity to\ninjected measurement (linear approx)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("W3b·W2b[:,h1:]", fontsize=7)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def in_dim(arch: dict) -> int:
    return arch["history_len"] * 6


# ── Input–output correlation ──────────────────────────────────────────────────

def collect_history_windows(results: dict, history_len: int):
    """Reconstruct sliding history windows from stored trajectories.

    At step t the network received features from steps [t-history_len .. t-1]
    (zero-padded when t < history_len) and produced rotate2.

    Returns:
        X : np.ndarray, shape (N, history_len, 6)  — raw feature values
        y : np.ndarray, shape (N,)                 — -rotate2_deg
    """
    X_rows, y_rows = [], []
    for session_data in results.values():
        for ep in session_data["episodes"]:
            traj = ep.get("trajectory", [])
            for t in range(1, len(traj)):
                # Canonicalise based on the IID at step t (the step that produced rotate2).
                flip = traj[t]["iid_db"] < 0
                window = np.zeros((history_len, 6), dtype=float)
                for s in range(history_len):
                    src = t - history_len + s
                    if src >= 0:
                        st = traj[src]
                        # Flip iid, rot1, rot2 signs for canonical frame.
                        window[s] = [
                            -st["iid_db"]      if flip else st["iid_db"],
                            st["distance_mm"],
                            -st["rotate1_deg"] if flip else st["rotate1_deg"],
                            -st["rotate2_deg"] if flip else st["rotate2_deg"],
                            st["executed_drive_mm"],
                            float(st["blocked"]),
                        ]
                X_rows.append(window)
                y_rows.append(traj[t]["rotate2_deg"] if flip else -traj[t]["rotate2_deg"])
    return np.array(X_rows), np.array(y_rows)


def plot_input_correlations(results: dict, arch: dict, gen, output_path: str) -> None:
    """Heatmap of Pearson r between sensory inputs (iid, dist) and -rotate2."""
    hl = arch["history_len"]
    X, y = collect_history_windows(results, hl)     # X: (N, hl, 6)

    # Only sensory features: iid=col0, dist=col1
    sensor_indices = [0, 1]
    sensor_names   = ["iid", "dist"]

    corr = np.zeros((hl, len(sensor_indices)))
    y_c = y - y.mean()
    y_norm = np.sqrt((y_c ** 2).sum()) or 1.0
    for s in range(hl):
        for i, f in enumerate(sensor_indices):
            x_col = X[:, s, f]
            x_c = x_col - x_col.mean()
            x_norm = np.sqrt((x_c ** 2).sum())
            corr[s, i] = float((x_c @ y_c) / (x_norm * y_norm)) if x_norm > 1e-9 else 0.0

    step_labels = [("t0" if s == hl - 1 else f"t{s - (hl - 1)}") for s in range(hl)]
    alpha = min(1.0, max(0.05, 500.0 / max(len(y), 1)))

    fig, axes = plt.subplots(2, hl, figsize=(3.5 * hl, 7), sharey=True)
    fig.suptitle(
        f"Sensory input vs −rotate2  |  gen {gen}  |  N={len(y)} steps",
        fontsize=11, fontweight="bold",
    )

    row_info = [(0, "iid", "iid  (dB)", FEAT_COLORS[0]),
                (1, "dist", "dist  (mm)", FEAT_COLORS[1])]

    for row, feat_name, xlabel, color in row_info:
        fi = sensor_indices[row]
        for s in range(hl):
            ax = axes[row, s]
            x_col = X[:, s, fi]
            r = corr[s, row]
            ax.scatter(x_col, y, s=4, alpha=alpha, color=color, linewidths=0)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.axvline(0, color="#555555", linewidth=0.6, linestyle=":")
            ax.text(0.05, 0.95, f"r = {r:+.3f}",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="top", ha="left",
                    color="darkred" if abs(r) > 0.3 else "black")
            ax.set_title(f"{feat_name}  [{step_labels[s]}]", fontsize=9, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8)
            if s == 0:
                ax.set_ylabel("−rotate2  (deg)", fontsize=8)
            ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Look-vs-drive plot ────────────────────────────────────────────────────────

def collect_steps(results: dict, deadband_db: float):
    """Return arrays of (canonical_iid_db, canonical_-rotate2_deg).

    Steps inside the deadband are excluded.  Canonical frame: if physical
    iid < 0 (wall on left), flip iid and rotate2 signs so everything is
    presented as wall-on-right — consistent with how the policy sees the world.
    """
    iids, neg_rot2 = [], []
    for session_data in results.values():
        for ep in session_data["episodes"]:
            for step in ep.get("trajectory", []):
                iid = step["iid_db"]
                r2  = step["rotate2_deg"]
                if abs(iid) >= deadband_db:
                    flip = iid < 0
                    iids.append(-iid if flip else iid)
                    neg_rot2.append(r2 if flip else -r2)
    return np.array(iids, dtype=float), np.array(neg_rot2, dtype=float)


def running_mean(x, y, bins=20):
    """Bin x, return bin centres and mean y per bin (ignoring empty bins)."""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() > 2:
            centres.append(0.5 * (lo + hi))
            means.append(y[mask].mean())
    return np.array(centres), np.array(means)


def plot_look_vs_drive(results: dict, deadband_db: float, gen, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"Look-vs-drive angle  |  gen {gen}\n"
        f"y = −rotate2  (positive → sonar leads drive direction)\n"
        f"canonical frame (iid<0 steps reflected)  |  deadband (|IID| < {deadband_db} dB) excluded",
        fontsize=10, fontweight="bold",
    )

    iid, nr2 = collect_steps(results, deadband_db)
    if len(iid) == 0:
        ax.set_title("No data")
    else:
        hb = ax.hexbin(iid, nr2, gridsize=30, cmap="YlOrRd", mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="step count")
        cx, cy = running_mean(iid, nr2, bins=20)
        ax.plot(cx, cy, "o-", color="#1565c0", linewidth=2, markersize=4, label="bin mean", zorder=5)
        ax.axhline(0, color="black",   linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black",   linewidth=0.8, linestyle="--")
        ax.axvline( deadband_db, color="#757575", linewidth=0.8,
                    linestyle=":", label=f"±deadband ({deadband_db} dB)")
        ax.axvline(-deadband_db, color="#757575", linewidth=0.8, linestyle=":")
        ax.set_title(f"n={len(iid)} steps", fontsize=9, fontweight="bold")
        ax.set_xlabel("IID  (dB,  +ve = wall on right)", fontsize=9)
        ax.set_ylabel("−rotate2  (deg)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── -rotate2 histogram ────────────────────────────────────────────────────────

def plot_rot2_histogram(results: dict, deadband_db: float, gen, output_path: str) -> None:
    """
    Two-panel histogram of -rotate2.

    Left  — physical frame: raw sign of IID preserved.  Wall-on-right steps
            cluster at positive values, wall-on-left at negative → bimodal.
    Right — canonical frame: negative-IID steps sign-flipped so both modes
            collapse into one positive peak.  This is what the network 'sees'.
    """
    # ── Physical frame ────────────────────────────────────────────────────────
    nr2_phys = []
    for session_data in results.values():
        for ep in session_data["episodes"]:
            for step in ep.get("trajectory", []):
                iid = step["iid_db"]
                r2  = step["rotate2_deg"]
                if abs(iid) >= deadband_db:
                    nr2_phys.append(-r2)
    nr2_phys = np.array(nr2_phys, dtype=float)

    # ── Canonical frame (existing helper) ────────────────────────────────────
    _, nr2_canon = collect_steps(results, deadband_db)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.suptitle(
        f"−rotate2 distribution  |  gen {gen}  |  deadband (|IID| < {deadband_db} dB) excluded",
        fontsize=10, fontweight="bold",
    )

    bins = np.linspace(-35, 35, 50)

    for ax, data, title, colour in [
        (axes[0], nr2_phys,  "physical frame  (raw IID sign)",         "steelblue"),
        (axes[1], nr2_canon, "canonical frame  (|IID|, neg steps flipped)", "seagreen"),
    ]:
        if len(data) == 0:
            ax.set_title("No data")
            continue
        ax.hist(data, bins=bins, color=colour, edgecolor="white", linewidth=0.4)
        ax.axvline(0,           color="black",  linewidth=1.0, linestyle="--")
        ax.axvline(data.mean(), color="tomato", linewidth=1.2, linestyle="-",
                   label=f"mean = {data.mean():.1f}°")
        ax.set_xlabel("−rotate2  (deg)", fontsize=9)
        ax.set_title(
            f"{title}\nn={len(data)}  median={np.median(data):.1f}°  std={data.std():.1f}°",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("step count", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── File lookup ───────────────────────────────────────────────────────────────

def find_json(output_dir: str, generation) -> str:
    gen_dir = os.path.join(output_dir, "generation_best")

    if generation is None:
        path = os.path.join(output_dir, "best_policy.json")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"best_policy.json not found in {output_dir}")

    if str(generation).lower() == "last":
        files = sorted(glob.glob(os.path.join(gen_dir, "gen_*_best_policy.json")))
        if not files:
            raise FileNotFoundError(f"No generation JSON files found in {gen_dir}")
        return files[-1]

    gen_num = int(generation)
    path = os.path.join(gen_dir, f"gen_{gen_num:03d}_best_policy.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return path


# ── Trajectory assessment ─────────────────────────────────────────────────────

TRIAL_COLORS = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#e65100", "#00838f"]


def make_policy(data: dict):
    from SCRIPT_TrainPolicy import HistoryNNPolicy
    policy = HistoryNNPolicy(
        max_rotate1_deg=data["max_rotate1_deg"],
        max_rotate2_deg=data["max_rotate2_deg"],
        deadband_db=data["iid_deadband_db"],
        history_len=data["history_len"],
        hidden_sizes=tuple(data["hidden_sizes"]),
    )
    policy.set_genome(np.array(data["genome"], dtype=np.float32))
    return policy


def run_episodes(data: dict, sessions: list, n_trials: int,
                 max_steps: int, seed: int) -> dict:
    """Run n_trials episodes for each session."""
    import random as _random
    from SCRIPT_TrainPolicy import Evaluator, Config, build_simulator

    policy = make_policy(data)
    rng = _random.Random(seed)

    results = {}
    for session in sessions:
        print(f"  {session} ...", end=" ", flush=True)
        sim = build_simulator(session)

        cfg = Config()
        cfg.max_steps               = max_steps
        cfg.history_len             = data["history_len"]
        cfg.max_rotate1_deg         = data["max_rotate1_deg"]
        cfg.max_rotate2_deg         = data["max_rotate2_deg"]
        cfg.iid_deadband_db         = data["iid_deadband_db"]
        cfg.quiet_setup             = True
        cfg.use_empirical_starts    = True
        cfg.randomize_empirical_yaw = True

        ev  = Evaluator(sim, cfg)
        eps = [ev.episode(policy, ev.sample_start(rng)) for _ in range(n_trials)]
        results[session] = {"sim": sim, "episodes": eps}
        print("done")

    return results


def plot_trajectories(results: dict, gen, output_path: str) -> None:
    sessions   = list(results.keys())
    n_sessions = len(sessions)
    n_cols     = min(n_sessions, 2)
    n_rows     = (n_sessions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(f"Trajectories  |  gen {gen}  |  {N_TRIALS} trials per session  "
                 f"(arrows = look direction)",
                 fontsize=10, fontweight="bold")

    for idx, session in enumerate(sessions):
        row, col = divmod(idx, n_cols)
        ax       = axes[row, col]
        sim      = results[session]["sim"]
        episodes = results[session]["episodes"]
        walls    = getattr(sim.arena, "walls", None)

        all_x, all_y = [], []
        if walls is not None and len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=1, color="#bdbdbd", alpha=0.4, zorder=0)
            all_x.extend(walls[:, 0].tolist())
            all_y.extend(walls[:, 1].tolist())

        for i, ep in enumerate(episodes):
            color = TRIAL_COLORS[i % len(TRIAL_COLORS)]
            traj  = ep.get("trajectory", [])
            if not traj:
                continue
            xs = [s["x"] for s in traj]
            ys = [s["y"] for s in traj]
            all_x.extend(xs); all_y.extend(ys)
            ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.75, label=f"T{i+1}")
            ax.scatter(xs[0],  ys[0],  s=45, color=color, marker="o", zorder=5)
            ax.scatter(xs[-1], ys[-1], s=45, color=color, marker="x", zorder=5, linewidths=1.5)
            STRIDE = 5; ARROW_LEN = 160.0
            idxs = range(0, len(traj), STRIDE)
            ax.quiver(
                [traj[k]["x"] for k in idxs], [traj[k]["y"] for k in idxs],
                [ARROW_LEN * np.cos(np.deg2rad(traj[k]["look_yaw_deg"])) for k in idxs],
                [ARROW_LEN * np.sin(np.deg2rad(traj[k]["look_yaw_deg"])) for k in idxs],
                units="xy", angles="xy", scale_units="xy", scale=1,
                color=color, alpha=0.5, width=6.0, headwidth=4, headlength=5,
            )

        if all_x and all_y:
            pad_x = max(30.0, 0.05 * (max(all_x) - min(all_x)))
            pad_y = max(30.0, 0.05 * (max(all_y) - min(all_y)))
            ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
            ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

        ax.set_title(f"{session}", fontsize=9, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)", fontsize=7)
        ax.set_ylabel("Y (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=6, markerscale=0.8)

    # Hide any unused panels
    for idx in range(n_sessions, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    policy_dir     = abs_path(POLICY_DIR)
    assessment_dir = abs_path(ASSESSMENT_DIR)
    os.makedirs(assessment_dir, exist_ok=True)

    json_path = find_json(policy_dir, GENERATION)
    print(f"Loading: {json_path}")

    data = load_policy_json(json_path)
    params, arch = reconstruct_params(data)

    gen = data.get("generation", "best")
    h1, h2 = arch["h1"], arch["h2"]
    suptitle = (
        f"Policy weights  |  gen {gen}  |  "
        f"history_len={arch['history_len']}  hidden={h1}×{h2}\n"
        f"train_fitness={data.get('train_fitness', float('nan')):.4f}  "
        f"val_fitness={data.get('val_fitness', float('nan')):.4f}  "
        f"train_collision={data.get('train_collision_rate', float('nan')):.2f}  "
        f"val_collision={data.get('val_collision_rate', float('nan')):.2f}"
    )

    out_file = os.path.join(assessment_dir, f"plot_weights_gen{gen}.png")
    plot_weights(params, arch, suptitle, out_file)

    print(f"Running episodes ({N_TRIALS} trials × {len(TRAIN_SESSIONS)} sessions) ...")
    results = run_episodes(data, TRAIN_SESSIONS, N_TRIALS, MAX_STEPS, SEED)
    traj_file = os.path.join(assessment_dir, f"plot_trajectories_gen{gen}.png")
    plot_trajectories(results, gen, traj_file)

    look_file = os.path.join(assessment_dir, f"plot_look_vs_drive_gen{gen}.png")
    plot_look_vs_drive(results, data["iid_deadband_db"], gen, look_file)

    corr_file = os.path.join(assessment_dir, f"plot_input_correlations_gen{gen}.png")
    plot_input_correlations(results, arch, gen, corr_file)

    hist_file = os.path.join(assessment_dir, f"plot_rot2_histogram_gen{gen}.png")
    plot_rot2_histogram(results, data["iid_deadband_db"], gen, hist_file)


if __name__ == "__main__":
    main()
