"""Chassis Geometry Extractor + Stiffness Solver - STEP file pipeline
Fixes: removes floating nodes, assigns OD per member, robust BCs"""
import json, re, math, itertools
from collections import defaultdict
import numpy as np

STEP_FILE = r"C:\Users\olich\Downloads\112675_SN5_TUBES.STEP"
E_MPA  = 200_000
SY_MPA = 460

SECTIONS = {
    19.05: {0.711: dict(A=40.96,  I=1724.7,  Z=181.1,  r=6.49),
             0.889: dict(A=50.72,  I=2096.1,  Z=220.1,  r=6.43),
             1.245: dict(A=69.64,  I=2773.2,  Z=291.1,  r=6.31)},
    25.40: {1.245: dict(A=94.48,  I=6908.8,  Z=544.0,  r=8.55),
             2.108: dict(A=154.25, I=10546.1, Z=830.4,  r=8.27),
             2.413: dict(A=174.26, I=11636.5, Z=916.3,  r=8.17)},
    28.575:{1.245: dict(A=106.90, I=10001.1, Z=700.0,  r=9.67)},
    31.75: {1.245: dict(A=119.31, I=13901.6, Z=875.7,  r=10.79),
             1.600: dict(A=151.55, I=17268.8, Z=1087.8, r=10.67),
             1.651: dict(A=156.12, I=17732.4, Z=1117.0, r=10.66)},
}
DEFAULT_WALL = {19.05: 0.711, 25.40: 1.245, 28.575: 1.245, 31.75: 1.245}

def snap_OD(d_mm):
    for nom in [19.05, 25.40, 28.575, 31.75]:
        if abs(d_mm - nom) < 0.5:
            return nom
    return None

def get_section(OD_mm):
    sec_dict = SECTIONS.get(OD_mm, {})
    if not sec_dict: return SECTIONS[25.40][1.245]
    w = DEFAULT_WALL.get(OD_mm)
    return sec_dict.get(w) or next(iter(sec_dict.values()))

def dist(a, b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Stage 1: Parsing STEP file")
print("=" * 60)

with open(STEP_FILE, "r", errors="replace") as f:
    raw = f.read()
raw = re.sub(r"\r\n|\r", "\n", raw)
raw = re.sub(r"\n\s+", " ", raw)

entities = {}
for m in re.finditer(r"#(\d+)\s*=\s*([^;]+);", raw):
    entities[int(m.group(1))] = m.group(2).strip()

cart_points = {}
for eid, body in entities.items():
    if body.startswith("CARTESIAN_POINT"):
        m = re.search(r"\(\s*'[^']*'\s*,\s*\(([^)]+)\)\s*\)", body)
        if m:
            try:
                c = [float(v) for v in m.group(1).split(",")]
                if len(c) == 3: cart_points[eid] = c
            except: pass

vertex_pts = {}
for eid, body in entities.items():
    if body.startswith("VERTEX_POINT"):
        m = re.search(r"VERTEX_POINT\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*\)", body)
        if m:
            pid = int(m.group(1))
            if pid in cart_points: vertex_pts[eid] = cart_points[pid]

line_ids = {eid for eid, b in entities.items() if b.startswith("LINE(") or b.startswith("LINE (")}

# Map EDGE_CURVE geom_id -> OD (from cylindrical surfaces sharing faces)
# Build: ADVANCED_FACE -> CYLINDRICAL_SURFACE radius
face_OD = {}
for eid, body in entities.items():
    if body.startswith("CYLINDRICAL_SURFACE"):
        m = re.search(r"CYLINDRICAL_SURFACE\s*\(\s*'[^']*'\s*,\s*#\d+\s*,\s*([\d.eE+\-]+)\s*\)", body)
        if m:
            r = float(m.group(1))
            nom = snap_OD(r * 2)
            if nom: face_OD[eid] = nom

# Build EDGE_LOOP -> EDGE_CURVE and ADVANCED_FACE -> EDGE_LOOP maps
# Then for each straight edge, find which face it belongs to -> get OD
edge_to_faces = defaultdict(set)  # edge_curve_id -> set of face ODs
for eid, body in entities.items():
    if body.startswith("ADVANCED_FACE"):
        # Find referenced edge loops
        face_cyl = None
        # Check if this face has a cylindrical geometry
        refs = re.findall(r"#(\d+)", body)
        for r in refs:
            if int(r) in face_OD:
                face_cyl = face_OD[int(r)]

        if face_cyl:
            # Find all EDGE_CURVEs in this face's loops
            for r in refs:
                bodies_to_check = entities.get(int(r), "")
                sub_refs = re.findall(r"#(\d+)", bodies_to_check)
                for sr in sub_refs:
                    sr_body = entities.get(int(sr), "")
                    if sr_body.startswith("EDGE_CURVE"):
                        edge_to_faces[int(sr)].add(face_cyl)

# Extract straight edges with OD hints
raw_edges = []
for eid, body in entities.items():
    if body.startswith("EDGE_CURVE"):
        m = re.search(r"EDGE_CURVE\s*\(\s*'([^']*)'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*\.", body)
        if m:
            v1, v2, gid = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if gid in line_ids:
                p1 = vertex_pts.get(v1)
                p2 = vertex_pts.get(v2)
                if p1 and p2:
                    L = dist(p1, p2)
                    if L > 8.0:
                        ods = list(edge_to_faces.get(eid, set()))
                        od_hint = ods[0] if len(ods) == 1 else (25.40 if not ods else max(ods))
                        raw_edges.append(dict(p1=p1, p2=p2, L=round(L,3), eid=eid, OD=od_hint))

print(f"  Raw straight edges (L>8mm): {len(raw_edges)}")

# ── Node clustering ────────────────────────────────────────────────────────────
TOL = 1.5
all_pts = [e["p1"] for e in raw_edges] + [e["p2"] for e in raw_edges]
NE = len(raw_edges)
nmap = [-1] * len(all_pts)
ncoords = []
for i, pt in enumerate(all_pts):
    if nmap[i] != -1: continue
    nid = len(ncoords)
    nmap[i] = nid
    cluster = [pt]
    for j in range(i+1, len(all_pts)):
        if nmap[j] == -1 and dist(pt, all_pts[j]) < TOL:
            nmap[j] = nid
            cluster.append(all_pts[j])
    ncoords.append([round(sum(p[k] for p in cluster)/len(cluster),3) for k in range(3)])
print(f"  Clustered nodes: {len(ncoords)}")

# ── Build members, deduplicate, assign OD ────────────────────────────────────
seen_pairs = defaultdict(list)
for idx, e in enumerate(raw_edges):
    n1 = nmap[idx]
    n2 = nmap[NE + idx]
    if n1 == n2: continue
    key = (min(n1,n2), max(n1,n2))
    seen_pairs[key].append((e["L"], e["OD"]))

members_raw = []
for (n1,n2), data in seen_pairs.items():
    L   = round(sum(d[0] for d in data)/len(data), 3)
    # OD: if any face gave a hint use most common, else 25.4
    od_votes = [d[1] for d in data if d[1]]
    OD = max(set(od_votes), key=od_votes.count) if od_votes else 25.40
    members_raw.append(dict(n1=n1, n2=n2, L=L, OD=OD))
print(f"  Members after dedup: {len(members_raw)}")

# ── Remove floating nodes (not connected to >=2 members) ──────────────────────
from collections import Counter
degree = Counter()
for m in members_raw:
    degree[m["n1"]] += 1
    degree[m["n2"]] += 1


# Iteratively remove nodes with degree < 2 (dangling ends)
def prune(members_raw):
    while True:
        degree = Counter()
        for m in members_raw:
            degree[m["n1"]] += 1
            degree[m["n2"]] += 1

        # Only flag nodes that actually exist in the active member list with < 2 connections
        bad = {n for n, count in degree.items() if count < 2}

        if not bad:
            break

        members_raw = [m for m in members_raw if m["n1"] not in bad and m["n2"] not in bad]
    return members_raw


members_raw = prune(members_raw)
print(f"  Members after pruning dangling ends: {len(members_raw)}")

# Reindex nodes to only those still in use
used_nodes = sorted({m["n1"] for m in members_raw} | {m["n2"] for m in members_raw})
old_to_new = {old: new for new, old in enumerate(used_nodes)}
nodes = {new: ncoords[old] for new, old in enumerate(used_nodes)}
for m in members_raw:
    m["n1"] = old_to_new[m["n1"]]
    m["n2"] = old_to_new[m["n2"]]
N = len(nodes)
print(f"  Final nodes: {N}, members: {len(members_raw)}")

# Build final member list with section props
members = []
for i, m in enumerate(members_raw):
    OD  = m["OD"]
    sec = get_section(OD)
    members.append({
        "id": i, "n1": m["n1"], "n2": m["n2"],
        "L_mm": m["L"], "OD_mm": OD,
        "wall_mm": DEFAULT_WALL.get(OD),
        "A_mm2":  sec["A"], "I_mm4": sec["I"],
        "r_mm":   sec["r"],
    })

od_dist = Counter(m["OD_mm"] for m in members)
print(f"  OD distribution (members): {dict(sorted(od_dist.items()))}")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Stage 2: Direct Stiffness Method")
print("=" * 60)

DOF = 3 * N
K   = np.zeros((DOF, DOF))
def xyz(nid): return np.array(nodes[nid], dtype=float)

for mem in members:
    i, j   = mem["n1"], mem["n2"]
    pi, pj = xyz(i), xyz(j)
    L      = np.linalg.norm(pj - pi)
    if L < 1e-6: continue
    A  = mem["A_mm2"]
    k  = E_MPA * A / L
    d  = (pj - pi) / L
    T  = np.outer(d, d)

    di = slice(3*i, 3*i+3); dj = slice(3*j, 3*j+3)
    K[di, di] +=  k * T
    K[di, dj] += -k * T
    K[dj, di] += -k * T
    K[dj, dj] +=  k * T
print("  K assembled")

# ── Boundary conditions ────────────────────────────────────────────────────────
xs = [nodes[i][0] for i in range(N)]
x_min, x_max = min(xs), max(xs)

# Fix ALL nodes at rear face (within 50mm of min X)
rear_nodes  = [i for i in range(N) if nodes[i][0] < x_min + 50]
front_nodes = [i for i in range(N) if nodes[i][0] > x_max - 50]

print(f"  Rear fixed nodes: {len(rear_nodes)}")
print(f"  Front load nodes: {len(front_nodes)}")

fixed_dofs = sorted({d for fn in rear_nodes for d in [3*fn, 3*fn+1, 3*fn+2]})
free_dofs  = [d for d in range(DOF) if d not in set(fixed_dofs)]

# ── Load case ─────────────────────────────────────────────────────────────────
MASS = 300; G = 9.81
F = np.zeros(DOF)

if front_nodes:
    n = len(front_nodes)
    for fn in front_nodes:
        F[3*fn]   -= MASS * G * 1.0 / n   # braking -X
        F[3*fn+1] += MASS * G * 1.5 / n   # cornering +Y

# ── Solve ─────────────────────────────────────────────────────────────────────
K_ff = K[np.ix_(free_dofs, free_dofs)]
F_f  = F[free_dofs]

# Check conditioning
cond = np.linalg.cond(K_ff)
print(f"  K_ff condition number: {cond:.2e}")

try:
    U_f = np.linalg.solve(K_ff, F_f)
    U   = np.zeros(DOF)
    for idx, d in enumerate(free_dofs):
        U[d] = U_f[idx]
    print("  ✓ Solved")
    solved = True
except np.linalg.LinAlgError as e:
    print(f"  ✗ Still singular: {e}")
    # Try pseudo-inverse as fallback
    U_f, *_ = np.linalg.lstsq(K_ff, F_f, rcond=None)
    U = np.zeros(DOF)
    for idx, d in enumerate(free_dofs):
        U[d] = U_f[idx]
    print("  ✓ Used least-squares (pinned DOFs may exist)")
    solved = True

# ── Member forces ─────────────────────────────────────────────────────────────
results = []
for mem in members:
    i, j   = mem["n1"], mem["n2"]
    pi, pj = xyz(i), xyz(j)
    L      = np.linalg.norm(pj - pi)
    if L < 1e-6: continue

    d      = (pj - pi) / L
    delta  = float(np.dot(d, U[3*j:3*j+3] - U[3*i:3*i+3]))

    A      = mem["A_mm2"]; I = mem["I_mm4"]; r = mem["r_mm"]
    F_ax   = E_MPA * A / L * delta
    sig    = F_ax / A
    Pcr    = math.pi**2 * E_MPA * I / L**2
    slend  = L / r

    if sig < 0:
        FOS = min(SY_MPA / max(abs(sig),0.001), Pcr / max(abs(F_ax),0.001))
        mode = "COMP"
    else:
        FOS = SY_MPA / max(abs(sig), 0.001)
        mode = "TENS"

    results.append({
        "id": mem["id"], "n1": i, "n2": j,
        "L_mm": round(L,2), "OD_mm": mem["OD_mm"],
        "F_N": round(float(F_ax),2),
        "sigma_MPa": round(float(sig),3),
        "slenderness": round(slend,1),
        "FOS": round(min(float(FOS),999),2),
        "mode": mode,
        "Pcr_N": round(Pcr,1),
    })

fos_vals = [r["FOS"] for r in results if r["FOS"] < 500]
if fos_vals:
    print(f"\n  Min FOS: {min(fos_vals):.2f}  |  Max σ: {max(abs(r['sigma_MPa']) for r in results):.1f} MPa")
    crit = sorted([r for r in results if r["FOS"] < 3.0], key=lambda x: x["FOS"])
    print(f"  Members FOS < 3.0: {len(crit)}")
    for r in crit[:10]:
        print(f"    M{r['id']:3d}  L={r['L_mm']:6.1f}mm  OD={r['OD_mm']}  "
              f"σ={r['sigma_MPa']:+8.2f}MPa  FOS={r['FOS']:.2f}  {r['mode']}")

output = {
    "meta": {"source": STEP_FILE, "E_MPa": E_MPA, "Sy_MPa": SY_MPA,
             "load": "1g brake + 1.5g corner, rear fixed", "units": "mm N MPa"},
    "nodes":   {str(k): v for k,v in nodes.items()},
    "members": members,
    "results": results,
}

with open("chassis_analysis.json","w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved chassis_analysis.json  ({N} nodes, {len(members)} members)")