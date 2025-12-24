import json
from collections import Counter
import numpy as np

file_path = 'output/playroom_experiment/scene_graph_70000.json'

with open(file_path, 'r') as f:
    data = json.load(f)

nodes = data['nodes']
print(f"Total nodes: {len(nodes)}")

# 1. Category Distribution
categories = [n['category'] for n in nodes]
cat_counts = Counter(categories)
print("\nTop 10 Categories:")
for cat, count in cat_counts.most_common(10):
    print(f"  {cat}: {count}")

# 2. Point Count Analysis
point_counts = [n['point_count'] for n in nodes]
print(f"\nPoint Counts:")
print(f"  Min: {min(point_counts)}")
print(f"  Max: {max(point_counts)}")
print(f"  Median: {np.median(point_counts)}")
print(f"  Mean: {np.mean(point_counts)}")

# 3. Small Objects (Potential Noise)
small_threshold = 100 # Arbitrary threshold
small_objects = [n for n in nodes if n['point_count'] < small_threshold]
print(f"\nObjects with < {small_threshold} points: {len(small_objects)}")
if small_objects:
    print("  Examples of small objects:")
    for n in small_objects[:5]:
        print(f"    ID {n['id']} ({n['category']}): {n['point_count']} points")

# 4. Check for "unknown"
unknowns = [n for n in nodes if n['category'] == 'unknown']
print(f"\nUnknown category objects: {len(unknowns)}")
