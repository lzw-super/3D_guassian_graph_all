import argparse
import json
import numpy as np
from plyfile import PlyData
import os
import sys
# ensure repo root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.opengs_utlis import load_code_book


def load_ply_positions(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    pts = np.vstack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])]).T
    return pts


def compute_stats(positions, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    stats = {}
    for cid in unique:
        mask = labels == cid
        pts = positions[mask]
        if pts.shape[0] == 0:
            continue
        mean = pts.mean(axis=0)
        median = np.median(pts, axis=0)
        dists_mean = np.linalg.norm(pts - mean[None, :], axis=1)
        dists_median = np.linalg.norm(pts - median[None, :], axis=1)
        stats[int(cid)] = {
            'count': int(pts.shape[0]),
            'mean': mean.tolist(),
            'median': median.tolist(),
            'mean_avg_dist': float(dists_mean.mean()),
            'median_avg_dist': float(dists_median.mean()),
            'mean_min_dist': float(dists_mean.min()),
            'median_min_dist': float(dists_median.min())
        }
    return stats


def compare_json_centers(stats, json_path):
    with open(json_path, 'r') as f:
        g = json.load(f)
    nodes = g.get('nodes', [])
    comp = []
    for n in nodes:
        cid = int(n['id'])
        json_center = np.array(n['center'], dtype=float)
        if cid not in stats:
            comp.append((cid, 'missing_cluster'))
            continue
        # compare distance from json_center to cluster points via min distance approx
        # We'll use mean and median centers distances
        s = stats[cid]
        mean = np.array(s['mean'])
        median = np.array(s['median'])
        d_json_mean = np.linalg.norm(json_center - mean)
        d_json_median = np.linalg.norm(json_center - median)
        comp.append((cid, d_json_mean, d_json_median))
    return comp


def summarize_stats(stats):
    import math
    means = []
    medians = []
    counts = []
    for k,v in stats.items():
        means.append(v['mean_avg_dist'])
        medians.append(v['median_avg_dist'])
        counts.append(v['count'])
    return {
        'clusters': len(stats),
        'points_total': int(sum(counts)),
        'mean_avg_of_avgdist': float(np.mean(means)) if len(means)>0 else float('nan'),
        'median_avg_of_avgdist': float(np.mean(medians)) if len(medians)>0 else float('nan')
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ply', required=True)
    p.add_argument('--codebook_dir', required=True)
    p.add_argument('--json', required=False)
    args = p.parse_args()

    print('Loading PLY:', args.ply)
    pts = load_ply_positions(args.ply)
    print('Num points:', pts.shape[0])

    print('Loading codebook (leaf indices) from:', args.codebook_dir)
    codebook, leaf_indices = load_code_book(args.codebook_dir)
    leaf_indices = np.asarray(leaf_indices)
    print('Loaded leaf indices length:', leaf_indices.shape)

    if leaf_indices.shape[0] != pts.shape[0]:
        print('WARNING: length mismatch between PLY points and leaf indices!')

    print('Computing per-cluster mean/median stats...')
    stats = compute_stats(pts, leaf_indices[:pts.shape[0]])
    summary = summarize_stats(stats)
    print('Summary:', json.dumps(summary, indent=2))

    # quick compare mean vs median
    better_mean = 0
    better_median = 0
    tie = 0
    for cid, s in stats.items():
        if s['mean_avg_dist'] < s['median_avg_dist']:
            better_mean += 1
        elif s['mean_avg_dist'] > s['median_avg_dist']:
            better_median += 1
        else:
            tie += 1
    print(f'Clusters where mean gives smaller avg intra-cluster distance: {better_mean}')
    print(f'Clusters where median gives smaller avg intra-cluster distance: {better_median}, ties: {tie}')

    if args.json:
        print('Comparing JSON centers to mean/median centers...')
        comp = compare_json_centers(stats, args.json)
        closer_to_mean = 0
        closer_to_median = 0
        missing = 0
        for item in comp:
            if item[1] == 'missing_cluster':
                missing += 1
                continue
            cid, djm, dmd = item
            if djm < dmd:
                closer_to_mean += 1
            elif dmd < djm:
                closer_to_median += 1
        print(f'JSON centers closer to mean: {closer_to_mean}, closer to median: {closer_to_median}, missing: {missing}')

    # Save a small CSV of per-cluster stats for inspection
    out_csv = os.path.join(os.path.dirname(args.ply), 'cluster_center_stats.csv')
    with open(out_csv, 'w') as fout:
        fout.write('cluster_id,count,mean_x,mean_y,mean_z,median_x,median_y,median_z,mean_avg_dist,median_avg_dist,mean_min_dist,median_min_dist\n')
        for cid, v in sorted(stats.items()):
            fout.write(f"{cid},{v['count']},{v['mean'][0]},{v['mean'][1]},{v['mean'][2]},{v['median'][0]},{v['median'][1]},{v['median'][2]},{v['mean_avg_dist']},{v['median_avg_dist']},{v['mean_min_dist']},{v['median_min_dist']}\n")
    print('Per-cluster CSV written to', out_csv)

if __name__ == '__main__':
    main()
