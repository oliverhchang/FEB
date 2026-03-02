import math
import json
import re

# File Paths
IGES_FILE = "chassis_wireframe.igs"
OUTPUT_FILE = "chassis_analysis.json"
LENGTH_TOLERANCE_MM = 2.0

# Your pasted cut list data
RAW_CUT_LIST = """
1 1 Al TUBE 1 SQR x 0.049 WALL 382.4
2 1 Al TUBE 1 SQR x 0.049 WALL 362.59
3 1 Al TUBE 1 SQR X 0.049 WALL 382.4
4 1 Al TUBE 1 SQR X 0.049 WALL 362.59
5 1 Al TUBE 1 SQR x 0.049 WALL 455.16
6 1 Al TUBE 1 SQR x 0.049 WALL 854.4
7 1 Al TUBE 1 SQR x 0.049 WALL 854.4
8 1 Al RND TUBE 1.25 OD x 0.049 Wall 604.45
9 1 AL RND TUBE 1.25 OD X 0.049 Wall 604.45
10 1 AL RND TUBE 1.25 OD X 0.049 Wall 625.44
11 1 AL RND TUBE 1.25 OD X 0.049 Wall 625.44
12 1 Al RND TUBE 1.25 OD x 0.049 Wall 782.33
13 1 AL RND TUBE 1.25 OD X 0.049 Wall 782.33
14 1 Al RND TUBE 0.75 OD x 0.028 Wall 109.03
15 1 Al RND TUBE 0.75 OD x 0.028 Wall 109.03
16 1 AL RND TUBE 1.25 OD X 0.049 Wall 398.57
17 1 AL RND TUBE 1.25 OD X 0.049 Wall 398.57
18 1 AL RND TUBE 1.25 OD X 0.049 Wall 260.3
19 1 AL RND TUBE 1.25 OD X 0.049 Wall 260.3
20 1 AL RND TUBE 1.25 OD X 0.049 Wall 331.91
21 1 AL RND TUBE 1.25 OD X 0.049 Wall 331.91
22 1 AL RND TUBE 1.25 OD X 0.049 Wall 469.69
23 1 AL RND TUBE 1.25 OD X 0.049 Wall 469.69
24 1 AL RND TUBE 1.25 OD X 0.049 Wall 627.19
25 1 AL RND TUBE 1.25 OD X 0.049 Wall 627.19
26 1 Al RND TUBE 1 OD x 0.049 Wall 709.39
27 1 Al RND TUBE 1 OD x 0.049 Wall 709.44
28 1 Al TUBE 1.25 SQR x 0.065 WALL 859.07
29 1 Al TUBE 1.25 SQR x 0.065 WALL 859.07
30 1 Al RND TUBE 1 OD x 0.049 Wall 649.64
31 1 Al RND TUBE 1 OD x 0.049 Wall 649.64
32 1 Al RND TUBE 1.25 OD x 0.049 Wall 523.8
33 1 Al RND TUBE 1 OD x 0.049 Wall 523.8
34 1 Al RND TUBE 0.75 OD x 0.035 Wall 430.19
35 1 Al RND TUBE 0.75 OD x 0.035 Wall 430.19
36 1 Al RND TUBE 1.25 OD x 0.049 Wall 871.27
37 1 Al RND TUBE 1.25 OD x 0.049 Wall 871.27
38 1 Al RND TUBE 1 OD x 0.049 Wall 306.4
39 1 Al RND TUBE 1 OD x 0.049 Wall 306.4
40 1 Al RND TUBE 1 OD x 0.049 Wall 216.19
41 1 Al RND TUBE 1 OD x 0.049 Wall 216.19
42 1 Al RND TUBE 1 OD x 0.049 Wall 276.01
43 1 Al RND TUBE 1 OD x 0.049 Wall 276.01
44 1 Al RND TUBE 1 OD x 0.049 Wall 288.36
45 1 Al RND TUBE 1 OD x 0.049 Wall 288.36
46 1 Al RND TUBE 1 OD x 0.049 Wall 280.92
47 1 Al RND TUBE 1 OD x 0.049 Wall 280.92
48 1 Al RND TUBE 1.125 OD x 0.049 Wall 303.31
49 1 Al RND TUBE 1.125 OD x 0.049 Wall 303.31
50 1 Al RND TUBE 1.125 OD x 0.049 Wall 623.42
51 1 Al RND TUBE 1.125 OD x 0.049 Wall 652.8
52 1 Al RND TUBE 1 OD x 0.049 Wall 647
53 1 Al RND TUBE 1 OD x 0.049 Wall 673.6
54 1 Al RND TUBE 1 OD x 0.049 Wall 134.36
55 1 Al RND TUBE 1 OD x 0.049 Wall 134.36
56 1 Al RND TUBE 1 OD x 0.095 Wall 645.01
57 1 Al RND TUBE 0.75 OD x 0.049 Wall 393.26
58 1 Al RND TUBE 1.25 OD x 0.065 Wall 631.28
59 1 AL RND TUBE 1.25 OD X 0.049 Wall 573.88
60 1 Al RND TUBE 0.75 OD x 0.028 Wall 137.57
61 1 Al RND TUBE 0.75 OD x 0.028 Wall 137.57
62 1 Al TUBE 1 SQR x 0.035 WALL 331.6
63 1 AL RND TUBE 0.75 OD x 0.035 Wall 340.2
64 1 Al RND TUBE 0.75 OD x 0.028 Wall 838.03
65 1 Al RND TUBE 1 OD x 0.095 Wall 1509.91
A 66 1 Al RND TUBE 1 OD x 0.095 Wall 2792.72
"""


def dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def parse_iges_lines(filepath):
    """Extracts Entity 110 (Lines) from an IGES file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Make sure it is in the same folder.")
        return []

    p_lines = [line[:64] for line in lines if len(line) >= 72 and line[72] == 'P']
    p_data = "".join(p_lines).replace('\n', '').replace(' ', '').split(';')

    segments = []
    for data in p_data:
        parts = data.split(',')
        if len(parts) >= 7 and parts[0] == '110':
            try:
                p1 = (float(parts[1]), float(parts[2]), float(parts[3]))
                p2 = (float(parts[4]), float(parts[5]), float(parts[6]))
                segments.append((p1, p2))
            except ValueError:
                continue
    return segments


print("Step 1: Parsing IGES Wireframe...")
raw_segments = parse_iges_lines(IGES_FILE)
if not raw_segments:
    exit()

print(f"Found {len(raw_segments)} pure line segments.")

# Node Clustering
nodes = []
nmap = {}
members_raw = []

for p1, p2 in raw_segments:
    node_ids = []
    for pt in (p1, p2):
        rounded_pt = (round(pt[0], 1), round(pt[1], 1), round(pt[2], 1))
        if rounded_pt not in nmap:
            nmap[rounded_pt] = len(nodes)
            nodes.append(rounded_pt)
        node_ids.append(nmap[rounded_pt])

    L = round(dist(p1, p2), 2)
    members_raw.append({"n1": node_ids[0], "n2": node_ids[1], "L_mm": L})

print(f"Clustered into {len(nodes)} exact nodes.")

print("Step 2: Parsing Text Cut List...")
cut_list_tubes = []
for line in RAW_CUT_LIST.strip().split('\n'):
    line = line.strip()
    if not line or 'DESCRIPTION' in line:
        continue

    # Extract everything up to the last number (which is the length)
    match = re.search(r'^(.*?)\s+([\d\.]+)$', line)
    if match:
        desc = match.group(1).strip()
        length = float(match.group(2))
        cut_list_tubes.append({"desc": desc, "length": length})

print("Step 3: Matching Geometry to Properties...")
matched_members = []
unmatched_count = 0

for mem in members_raw:
    target_L = mem["L_mm"]
    best_match = None
    best_diff = float('inf')
    best_idx = -1

    for idx, tube in enumerate(cut_list_tubes):
        diff = abs(target_L - tube["length"])
        if diff < best_diff and diff <= LENGTH_TOLERANCE_MM:
            best_diff = diff
            best_match = tube
            best_idx = idx

    if best_match:
        matched_tube = cut_list_tubes.pop(best_idx)
        desc = matched_tube["desc"]

        # Determine shape, OD, and Wall
        shape = "SQR" if "SQR" in desc.upper() else "RND"
        od_match = re.search(r'([\d.]+)\s*(?:OD|SQR)', desc, re.IGNORECASE)
        wall_match = re.search(r'([\d.]+)\s*(?:Wall|W)', desc, re.IGNORECASE)

        od_in = float(od_match.group(1)) if od_match else 1.0
        wall_in = float(wall_match.group(1)) if wall_match else 0.049

        mem["OD_mm"] = round(od_in * 25.4, 3)
        mem["wall_mm"] = round(wall_in * 25.4, 3)
        mem["shape"] = shape
        mem["desc"] = desc
        matched_members.append(mem)
    else:
        unmatched_count += 1
        mem["OD_mm"] = 25.4
        mem["wall_mm"] = 1.245
        mem["shape"] = "RND"
        mem["desc"] = "UNMATCHED DEFAULT"
        matched_members.append(mem)

print(f"Matched {len(matched_members) - unmatched_count} tubes successfully.")
if unmatched_count > 0:
    print(f"Warning: {unmatched_count} tubes could not be matched. Defaulted to 1 inch OD RND.")

output = {
    "nodes": {str(i): list(n) for i, n in enumerate(nodes)},
    "members": [{"id": i, **m} for i, m in enumerate(matched_members)]
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"Success! Saved clean node topology to {OUTPUT_FILE}.")