"""
Script to apply options 1+2+3 in a lightweight way without re-rendering:
- Load PLY and leaf_code_book to compute cluster_stats using mean centers
- Load existing scene_graph JSON
- Run modified post_process_graph (which now returns id_mapping)
- Recompute merged node centers using cluster_stats (option 3)
- Compare JSON centers (original) vs recomputed (new) and list top-20 differences
- Save a new JSON `scene_graph_{iter}_fixed.json`
"""
import os, sys, json
import numpy as np
from plyfile import PlyData
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.opengs_utlis import load_code_book
from build_scene_graph import post_process_graph


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
        min_pt = np.percentile(pts, 5, axis=0)
        max_pt = np.percentile(pts, 95, axis=0)
        stats[int(cid)] = {
            'center': [float(x) for x in mean.tolist()],
            'bbox': [[float(x) for x in min_pt.tolist()], [float(x) for x in max_pt.tolist()]],
            'point_count': int(pts.shape[0])
        }
    return stats


def main():
    base = 'output/room5_experiment'
    iter_dir = os.path.join(base, 'point_cloud/iteration_70000')
    ply = os.path.join(iter_dir, 'point_cloud.ply')
    codebook_dir = iter_dir + '/leaf_code_book'
    json_path = os.path.join(base, 'scene_graph_70000.json')

    print('Loading PLY...')
    pts = load_ply_positions(ply)
    print('Loading leaf indices...')
    codebook, inds = load_code_book(codebook_dir)
    inds = np.asarray(inds)
    stats = compute_stats(pts, inds[:pts.shape[0]])
    print(f'Computed stats for {len(stats)} clusters')

    print('Loading existing scene graph...')
    with open(json_path,'r') as f:
        g = json.load(f)

    print('Running post_process_graph...')
    new_graph, id_mapping = post_process_graph(g)

    # build reverse map
    rev_map = {}
    for orig_id in stats.keys():
        new_id = id_mapping.get(orig_id, orig_id)
        rev_map.setdefault(new_id, []).append(orig_id)

    # recompute centers
    recomputed = {}
    for node in new_graph['nodes']:
        nid = node['id']
        orig_ids = rev_map.get(nid, [nid])
        total = 0
        ssum = np.zeros(3)
        bmin = np.array([1e9,1e9,1e9])
        bmax = np.array([-1e9,-1e9,-1e9])
        for oid in orig_ids:
            st = stats.get(oid)
            if st is None: continue
            cnt = st['point_count']
            total += cnt
            ssum += np.array(st['center']) * cnt
            bmin = np.minimum(bmin, np.array(st['bbox'][0]))
            bmax = np.maximum(bmax, np.array(st['bbox'][1]))
        if total>0:
            c = (ssum/total).tolist()
        else:
            c = node['center']
        recomputed[nid] = {'center': c, 'bbox':[bmin.tolist(), bmax.tolist()], 'point_count': total}

    # Compare original vs recomputed centers
    diffs = []
    for node in new_graph['nodes']:
        nid = node['id']
        orig = np.array(node['center'], dtype=float)
        rec = np.array(recomputed[nid]['center'], dtype=float)
        d = np.linalg.norm(orig - rec)
        diffs.append((nid, d, node['point_count'], recomputed[nid]['point_count'], orig.tolist(), rec.tolist()))
    diffs.sort(key=lambda x: x[1], reverse=True)

    out_json = os.path.join(base, 'scene_graph_70000_fixed.json')
    # write updated graph (replace centers/bbox/point_count)
    for node in new_graph['nodes']:
        nid = node['id']
        r = recomputed[nid]
        node['center'] = [round(float(x),6) for x in r['center']]
        node['bbox'] = [[float(x) for x in r['bbox'][0]], [float(x) for x in r['bbox'][1]]]
        node['point_count'] = int(r['point_count'])

    with open(out_json,'w') as f:
        json.dump(new_graph, f, indent=2)

    print('Wrote fixed JSON to', out_json)

    print('\nTop-20 center differences (orig vs recomputed):')
    for it in diffs[:20]:
        print(f'ID: {it[0]:4d}  dist: {it[1]:.4f}  orig_count: {it[2]:6d}  new_count: {it[3]:6d}')
        print(f'  orig_center: {it[4]}')
        print(f'  new_center : {it[5]}')

if __name__=='__main__':
    main()
