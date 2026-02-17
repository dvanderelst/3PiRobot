import json
import os
import re
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from Library import DataProcessor


# ============================================
# CONFIGURATION
# ============================================
# If None, use newest robot-frame series file in Prediction/heatmap_series_robot/
series_npz_path = None
analysis_output_dir = 'Prediction/WallinessAnalysis'
training_output_dir = 'Training'

run_sonar_comparison = True

knn_k_values = [1, 3, 5, 10]
retrieval_radii_mm = [200.0, 500.0, 1000.0]
yaw_thresholds_deg = [10.0, 20.0, 45.0, 90.0]
yaw_to_mm = 15.0  # 1 degree yaw difference counts as this many mm in pose distance

permutation_count = 500
random_seed = 42
scatter_max_points = 15000


def find_default_series_file():
    candidates = glob('Prediction/heatmap_series_robot/*_robotframe_series_data.npz')
    if not candidates:
        raise FileNotFoundError(
            "No robot-frame walliness series file found. "
            "Run SCRIPT_PredictProfiles.py with robot-frame series enabled first."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def infer_session_from_series_path(path):
    base = os.path.basename(path)
    m = re.match(r'(session[^_]+)_robotframe_series_data\.npz$', base)
    if not m:
        raise ValueError(
            f"Could not infer session from filename '{base}'. "
            "Expected format like sessionB03_robotframe_series_data.npz"
        )
    return m.group(1)


def rankdata_simple(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def spearman_corr(x, y):
    rx = rankdata_simple(x)
    ry = rankdata_simple(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def angular_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def compute_retrieval_metrics(sim_mat, dist_mat, k_values, radii_mm, candidate_mask=None):
    n = sim_mat.shape[0]
    retrieval = {}
    nn_distances = []
    random_distances = []
    valid_queries = 0

    for i in range(n):
        sim_i = sim_mat[i].copy()
        valid = np.ones(n, dtype=bool) if candidate_mask is None else candidate_mask[i].copy()
        valid[i] = False
        if not np.any(valid):
            continue

        sim_i[~valid] = -np.inf
        neighbors = np.argsort(-sim_i)
        top1 = neighbors[0]
        nn_distances.append(float(dist_mat[i, top1]))

        valid_idx = np.where(valid)[0]
        j = valid_idx[np.random.randint(0, len(valid_idx))]
        random_distances.append(float(dist_mat[i, j]))
        valid_queries += 1

    nn_distances = np.asarray(nn_distances, dtype=float)
    random_distances = np.asarray(random_distances, dtype=float)

    for k in k_values:
        topk_dists = []
        recall_per_radius = {str(r): [] for r in radii_mm}
        valid_q_k = 0

        for i in range(n):
            sim_i = sim_mat[i].copy()
            valid = np.ones(n, dtype=bool) if candidate_mask is None else candidate_mask[i].copy()
            valid[i] = False
            if not np.any(valid):
                continue

            sim_i[~valid] = -np.inf
            valid_count = int(np.sum(valid))
            k_eff = min(k, valid_count)
            neighbors = np.argsort(-sim_i)[:k_eff]
            d = dist_mat[i, neighbors]
            topk_dists.append(float(np.mean(d)))
            for r in radii_mm:
                recall_per_radius[str(r)].append(float(np.any(d <= r)))
            valid_q_k += 1

        retrieval[f'k={k}'] = {
            'valid_queries': int(valid_q_k),
            'mean_topk_distance_mm': float(np.mean(topk_dists)) if len(topk_dists) > 0 else float('nan'),
            'std_topk_distance_mm': float(np.std(topk_dists)) if len(topk_dists) > 0 else float('nan'),
            'recall_within_radius': {
                rk: (float(np.mean(vals)) if len(vals) > 0 else float('nan'))
                for rk, vals in recall_per_radius.items()
            },
        }

    return {
        'valid_queries': int(valid_queries),
        'nearest_neighbor_mean_distance_mm': float(np.mean(nn_distances)) if len(nn_distances) > 0 else float('nan'),
        'nearest_neighbor_std_distance_mm': float(np.std(nn_distances)) if len(nn_distances) > 0 else float('nan'),
        'random_pair_mean_distance_mm': float(np.mean(random_distances)) if len(random_distances) > 0 else float('nan'),
        'random_pair_std_distance_mm': float(np.std(random_distances)) if len(random_distances) > 0 else float('nan'),
        'metrics_by_k': retrieval,
        'nn_distances': nn_distances,
        'random_distances': random_distances,
    }


def analyze_representation(rep_name, features, anchor_x, anchor_y, anchor_yaw_deg, output_dir):
    n = features.shape[0]

    X = features.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    sim_mat = Xn @ Xn.T

    dx = anchor_x[:, None] - anchor_x[None, :]
    dy = anchor_y[:, None] - anchor_y[None, :]
    dist_mat = np.sqrt(dx * dx + dy * dy)

    yaw_diff_mat = angular_diff_deg(anchor_yaw_deg[:, None], anchor_yaw_deg[None, :]) if anchor_yaw_deg is not None else None
    pose_dist_mat = np.sqrt(dist_mat * dist_mat + (yaw_to_mm * yaw_diff_mat) ** 2) if yaw_diff_mat is not None else None

    iu = np.triu_indices(n, k=1)
    sim_pairs = sim_mat[iu]
    dist_pairs = dist_mat[iu]

    rho = spearman_corr(sim_pairs, dist_pairs)

    rho_perm = np.zeros(permutation_count, dtype=float)
    for i in range(permutation_count):
        rho_perm[i] = spearman_corr(sim_pairs, np.random.permutation(dist_pairs))
    p_value = (np.sum(np.abs(rho_perm) >= abs(rho)) + 1) / (permutation_count + 1)

    # Pairwise plot
    plt.figure(figsize=(8, 6))
    plt.hexbin(dist_pairs, sim_pairs, gridsize=60, bins='log', mincnt=1, cmap='viridis')
    plt.colorbar(label='log10(count)')
    plt.xlabel('Anchor Spatial Distance (mm)')
    plt.ylabel('Cosine Similarity')
    plt.title(f'{rep_name}: Similarity vs Spatial Distance\nSpearman rho={rho:.3f}, p={p_value:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{rep_name.lower()}_similarity_vs_distance_hexbin.png'), dpi=220, bbox_inches='tight')
    plt.close()

    # Scatter variant of the same correlation plot (subsampled for readability).
    if len(dist_pairs) > scatter_max_points:
        sel = np.random.choice(len(dist_pairs), size=scatter_max_points, replace=False)
        x_sc = dist_pairs[sel]
        y_sc = sim_pairs[sel]
    else:
        x_sc = dist_pairs
        y_sc = sim_pairs
    plt.figure(figsize=(8, 6))
    plt.scatter(x_sc, y_sc, s=5, alpha=0.15)
    plt.xlabel('Anchor Spatial Distance (mm)')
    plt.ylabel('Cosine Similarity')
    plt.title(f'{rep_name}: Similarity vs Spatial Distance (Scatter)')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{rep_name.lower()}_similarity_vs_distance_scatter.png'), dpi=220, bbox_inches='tight')
    plt.close()

    yaw_pairwise = None
    if yaw_diff_mat is not None:
        yaw_pairwise = {}
        for thr in yaw_thresholds_deg:
            mask = yaw_diff_mat[iu] <= thr
            if np.sum(mask) < 10:
                yaw_pairwise[f'<= {thr} deg'] = {
                    'num_pairs': int(np.sum(mask)),
                    'spearman_similarity_vs_distance': float('nan'),
                }
                continue
            yaw_pairwise[f'<= {thr} deg'] = {
                'num_pairs': int(np.sum(mask)),
                'spearman_similarity_vs_distance': float(spearman_corr(sim_pairs[mask], dist_pairs[mask])),
            }

            # Yaw-filtered scatter plot
            d_thr = dist_pairs[mask]
            s_thr = sim_pairs[mask]
            if len(d_thr) > scatter_max_points:
                sel_thr = np.random.choice(len(d_thr), size=scatter_max_points, replace=False)
                d_thr = d_thr[sel_thr]
                s_thr = s_thr[sel_thr]
            plt.figure(figsize=(8, 6))
            plt.scatter(d_thr, s_thr, s=5, alpha=0.18)
            plt.xlabel('Anchor Spatial Distance (mm)')
            plt.ylabel('Cosine Similarity')
            plt.title(f'{rep_name}: Similarity vs Distance (|Δyaw| <= {thr:.0f}°)')
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'{rep_name.lower()}_similarity_vs_distance_scatter_yaw_le_{int(thr):02d}.png'),
                dpi=220,
                bbox_inches='tight'
            )
            plt.close()

    retrieval_spatial = compute_retrieval_metrics(
        sim_mat=sim_mat,
        dist_mat=dist_mat,
        k_values=knn_k_values,
        radii_mm=retrieval_radii_mm,
        candidate_mask=None,
    )

    retrieval_yaw_filtered = None
    retrieval_pose_distance = None
    if yaw_diff_mat is not None:
        retrieval_yaw_filtered = {}
        for thr in yaw_thresholds_deg:
            cand = yaw_diff_mat <= thr
            r = compute_retrieval_metrics(
                sim_mat=sim_mat,
                dist_mat=dist_mat,
                k_values=knn_k_values,
                radii_mm=retrieval_radii_mm,
                candidate_mask=cand,
            )
            r.pop('nn_distances', None)
            r.pop('random_distances', None)
            retrieval_yaw_filtered[f'<= {thr} deg'] = r

        pose_radii = [500.0, 1000.0, 1500.0, 2000.0]
        retrieval_pose_distance = compute_retrieval_metrics(
            sim_mat=sim_mat,
            dist_mat=pose_dist_mat,
            k_values=knn_k_values,
            radii_mm=pose_radii,
            candidate_mask=None,
        )
        retrieval_pose_distance.pop('nn_distances', None)
        retrieval_pose_distance.pop('random_distances', None)

    # Retrieval histogram
    nn_distances = retrieval_spatial['nn_distances']
    random_distances = retrieval_spatial['random_distances']
    plt.figure(figsize=(8, 6))
    plt.hist(random_distances, bins=40, alpha=0.5, label='Random pair distance')
    plt.hist(nn_distances, bins=40, alpha=0.7, label='Nearest-neighbor retrieval distance')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Count')
    plt.title(f'{rep_name}: Retrieval NN Distance vs Random Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{rep_name.lower()}_retrieval_nn_vs_random_hist.png'), dpi=220, bbox_inches='tight')
    plt.close()

    if retrieval_yaw_filtered is not None:
        x_thr, y_rec = [], []
        for thr in yaw_thresholds_deg:
            key = f'<= {thr} deg'
            val = retrieval_yaw_filtered[key]['metrics_by_k']['k=1']['recall_within_radius'].get('500.0', float('nan'))
            x_thr.append(thr)
            y_rec.append(val)
        plt.figure(figsize=(7, 5))
        plt.plot(x_thr, y_rec, marker='o')
        plt.xlabel('Allowed |Δyaw| (deg)')
        plt.ylabel('Recall@1 within 500 mm')
        plt.title(f'{rep_name}: Yaw-Filtered Retrieval Quality')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{rep_name.lower()}_retrieval_recall1_vs_yaw_threshold.png'), dpi=220, bbox_inches='tight')
        plt.close()

    # PCA embedding
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T
    var = (S ** 2) / max(n - 1, 1)
    var_ratio = var / np.maximum(np.sum(var), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=anchor_x, cmap='coolwarm', s=18)
    axes[0].set_title(f'{rep_name} PCA (color = anchor X)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    fig.colorbar(sc0, ax=axes[0], label='Anchor X (mm)')

    sc1 = axes[1].scatter(pcs[:, 0], pcs[:, 1], c=anchor_y, cmap='viridis', s=18)
    axes[1].set_title(f'{rep_name} PCA (color = anchor Y)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    fig.colorbar(sc1, ax=axes[1], label='Anchor Y (mm)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{rep_name.lower()}_pca_embedding_anchor_xy.png'), dpi=220, bbox_inches='tight')
    plt.close()

    return {
        'pairwise': {
            'spearman_similarity_vs_distance': float(rho),
            'permutation_p_value': float(p_value),
            'num_pairs': int(len(sim_pairs)),
            'permutation_count': int(permutation_count),
            'yaw_filtered_spearman': yaw_pairwise,
        },
        'retrieval': {
            'spatial_distance': {
                'valid_queries': retrieval_spatial['valid_queries'],
                'nearest_neighbor_mean_distance_mm': retrieval_spatial['nearest_neighbor_mean_distance_mm'],
                'nearest_neighbor_std_distance_mm': retrieval_spatial['nearest_neighbor_std_distance_mm'],
                'random_pair_mean_distance_mm': retrieval_spatial['random_pair_mean_distance_mm'],
                'random_pair_std_distance_mm': retrieval_spatial['random_pair_std_distance_mm'],
                'metrics_by_k': retrieval_spatial['metrics_by_k'],
            },
            'yaw_filtered_candidates': retrieval_yaw_filtered,
            'pose_distance_xy_plus_yaw': retrieval_pose_distance,
        },
        'embedding': {
            'pc1_variance_ratio': float(var_ratio[0]) if len(var_ratio) > 0 else float('nan'),
            'pc2_variance_ratio': float(var_ratio[1]) if len(var_ratio) > 1 else float('nan'),
        },
    }


def load_sonar_features_for_chunks(session_name, chunk_indices, training_params):
    dc = DataProcessor.DataCollection([session_name])
    sonar_data = dc.load_sonar(flatten=False)
    profiles_data, _ = dc.load_profiles(
        opening_angle=float(training_params['profile_opening_angle']),
        steps=int(training_params['profile_steps'])
    )

    finite_mask = np.isfinite(sonar_data).all(axis=(1, 2))
    finite_mask &= np.isfinite(profiles_data).all(axis=1)
    sonar_filtered = sonar_data[finite_mask]

    # chunk_indices are in filtered index space, so this should align exactly.
    max_idx = int(np.max(chunk_indices))
    if max_idx >= len(sonar_filtered):
        raise IndexError(
            f"chunk_indices max ({max_idx}) exceeds filtered sonar length ({len(sonar_filtered)})."
        )

    feats = []
    for row in chunk_indices:
        chunk = sonar_filtered[row]  # (chunk_size, 200, 2)
        feats.append(chunk.reshape(-1))
    return np.asarray(feats, dtype=np.float32)


def main():
    np.random.seed(random_seed)
    ensure_dir(analysis_output_dir)

    input_path = series_npz_path if series_npz_path else find_default_series_file()
    print(f"Loading walliness series: {input_path}")
    data = np.load(input_path)

    required = ['heatmaps', 'anchor_rob_x', 'anchor_rob_y', 'anchor_rob_yaw_deg', 'anchor_index', 'chunk_indices']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in npz: {missing}")

    heatmaps = data['heatmaps']
    anchor_x = data['anchor_rob_x']
    anchor_y = data['anchor_rob_y']
    anchor_yaw_deg = data['anchor_rob_yaw_deg']
    anchor_index = data['anchor_index']
    chunk_indices = data['chunk_indices']

    n = heatmaps.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 heatmaps for analysis, got {n}.")

    print(f"Loaded {n} heatmaps with shape {heatmaps.shape[1:]}.")

    # Analyze walliness features
    walliness_features = heatmaps.reshape(n, -1)
    walliness_results = analyze_representation(
        rep_name='Walliness',
        features=walliness_features,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        anchor_yaw_deg=anchor_yaw_deg,
        output_dir=analysis_output_dir,
    )

    sonar_results = None
    if run_sonar_comparison:
        session_name = infer_session_from_series_path(input_path)
        params_path = os.path.join(training_output_dir, 'training_params.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Missing training params for sonar alignment: {params_path}")
        with open(params_path, 'r') as f:
            training_params = json.load(f)

        print(f"Loading aligned sonar chunks for comparison (session={session_name})...")
        sonar_features = load_sonar_features_for_chunks(session_name, chunk_indices, training_params)

        sonar_results = analyze_representation(
            rep_name='Sonar',
            features=sonar_features,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            anchor_yaw_deg=anchor_yaw_deg,
            output_dir=analysis_output_dir,
        )

        # Comparison plot: Recall@1 within 500mm under yaw thresholds
        x_labels = ['No filter'] + [f"<= {int(t)}°" for t in yaw_thresholds_deg]
        wall_rec = [
            walliness_results['retrieval']['spatial_distance']['metrics_by_k']['k=1']['recall_within_radius']['500.0']
        ]
        sonar_rec = [
            sonar_results['retrieval']['spatial_distance']['metrics_by_k']['k=1']['recall_within_radius']['500.0']
        ]
        for t in yaw_thresholds_deg:
            key = f'<= {t} deg'
            wall_rec.append(walliness_results['retrieval']['yaw_filtered_candidates'][key]['metrics_by_k']['k=1']['recall_within_radius']['500.0'])
            sonar_rec.append(sonar_results['retrieval']['yaw_filtered_candidates'][key]['metrics_by_k']['k=1']['recall_within_radius']['500.0'])

        x = np.arange(len(x_labels))
        width = 0.38
        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, wall_rec, width=width, label='Walliness')
        plt.bar(x + width / 2, sonar_rec, width=width, label='Sonar')
        plt.xticks(x, x_labels)
        plt.ylim(0, 1.05)
        plt.ylabel('Recall@1 within 500 mm')
        plt.title('Position Retrieval: Walliness vs Sonar')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_output_dir, 'comparison_recall1_within500.png'), dpi=220, bbox_inches='tight')
        plt.close()

    summary = {
        'input_npz': input_path,
        'n_heatmaps': int(n),
        'heatmap_shape': [int(heatmaps.shape[1]), int(heatmaps.shape[2])],
        'walliness': walliness_results,
        'sonar': sonar_results,
        'data_refs': {
            'anchor_index_shape': list(anchor_index.shape),
            'chunk_indices_shape': list(chunk_indices.shape),
            'yaw_to_mm': float(yaw_to_mm),
        },
    }

    summary_path = os.path.join(analysis_output_dir, 'walliness_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved analysis outputs to: {analysis_output_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == '__main__':
    main()
