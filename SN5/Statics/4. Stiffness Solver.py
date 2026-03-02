import json
import numpy as np
import math

INPUT_FILE = "chassis_analysis.json"
OUTPUT_FILE = "chassis_results.json"

# Material Properties: 4130 Chromoly Steel
E_MPA = 200000  # Young's Modulus (N/mm^2)
SY_MPA = 460  # Yield Strength (MPa)

print(f"Loading {INPUT_FILE}...")
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

nodes = data["nodes"]
members = data["members"]
N_nodes = len(nodes)
DOF = 3 * N_nodes
node_keys = list(nodes.keys())


def get_node_idx(node_str):
    return node_keys.index(node_str)


# 1. Calculate Cross-Sectional Properties
for mem in members:
    od = mem["OD_mm"]
    wall = mem["wall_mm"]
    inner = od - (2 * wall)

    if mem.get("shape") == "SQR":
        mem["A_mm2"] = (od ** 2) - (inner ** 2)
        mem["I_mm4"] = ((od ** 4) - (inner ** 4)) / 12
    else:
        mem["A_mm2"] = math.pi * ((od ** 2) - (inner ** 2)) / 4
        mem["I_mm4"] = math.pi * ((od ** 4) - (inner ** 4)) / 64

# 2. Assemble Global Stiffness Matrix (K)
print("Assembling Global Stiffness Matrix...")
K = np.zeros((DOF, DOF))

for mem in members:
    idx_i = get_node_idx(str(mem["n1"]))
    idx_j = get_node_idx(str(mem["n2"]))

    pi = np.array(nodes[str(mem["n1"])])
    pj = np.array(nodes[str(mem["n2"])])

    L = np.linalg.norm(pj - pi)
    if L < 1e-6: continue
    mem["L_true_mm"] = L

    A = mem["A_mm2"]
    k_val = (E_MPA * A) / L
    d = (pj - pi) / L
    T = np.outer(d, d)

    di = slice(3 * idx_i, 3 * idx_i + 3)
    dj = slice(3 * idx_j, 3 * idx_j + 3)

    K[di, di] += k_val * T
    K[di, dj] += -k_val * T
    K[dj, di] += -k_val * T
    K[dj, dj] += k_val * T

# 3. Apply Boundary Conditions (Fix Rear Bulkhead)
print("Applying Boundary Conditions (Rear Fixed)...")
xs = [coord[0] for coord in nodes.values()]
x_min = min(xs)
rear_node_indices = [i for i, k in enumerate(node_keys) if nodes[k][0] < x_min + 150]
fixed_dofs = sorted({d for fn in rear_node_indices for d in [3 * fn, 3 * fn + 1, 3 * fn + 2]})
free_dofs = [d for d in range(DOF) if d not in set(fixed_dofs)]

# 4. Inject Exact Suspension Forces
print("Calculating Suspension Load Vectors...")
F = np.zeros(DOF)

# Data from Hand Calcs PDF (Coordinates in Inches, Force in Newtons)
suspension_links = [
    # ─── FRONT LEFT ─────────────────────────────────────────
    {
        "name": "Front Left Lower A-arm fore",
        "chassis_in": [6.11, 7.39, -2.17],
        "upright_in": [0.15, 22.16, -3.73],
        "force_N": 3334.37 # Max Compression
    },
    {
        "name": "Front Left Lower A-arm aft",
        "chassis_in": [-5.52, 7.39, -2.17],
        "upright_in": [0.15, 22.16, -3.73],
        "force_N": 4292.69 # Max Compression
    },
    {
        "name": "Front Left Upper A-arm fore",
        "chassis_in": [3.88, 9.21, 4.20],
        "upright_in": [-0.15, 21.68, 3.73],
        "force_N": 2879.67 # Max Compression
    },
    {
        "name": "Front Left Upper A-arm aft",
        "chassis_in": [-3.87, 9.85, 6.45],
        "upright_in": [-0.15, 21.68, 3.73],
        "force_N": 3117.05 # Max Compression
    },

    # ─── FRONT RIGHT (Y-coordinates flipped) ────────────────
    {
        "name": "Front Right Lower A-arm fore",
        "chassis_in": [6.11, -7.39, -2.17],
        "upright_in": [0.15, -22.16, -3.73],
        "force_N": 3334.37
    },
    {
        "name": "Front Right Lower A-arm aft",
        "chassis_in": [-5.52, -7.39, -2.17],
        "upright_in": [0.15, -22.16, -3.73],
        "force_N": 4292.69
    },
    {
        "name": "Front Right Upper A-arm fore",
        "chassis_in": [3.88, -9.21, 4.20],
        "upright_in": [-0.15, -21.68, 3.73],
        "force_N": 2879.67
    },
    {
        "name": "Front Right Upper A-arm aft",
        "chassis_in": [-3.87, -9.85, 6.45],
        "upright_in": [-0.15, -21.68, 3.73],
        "force_N": 3117.05
    }
]


def find_closest_node(target_xyz_mm):
    best_dist = float('inf')
    best_idx = -1
    for i, key in enumerate(node_keys):
        n_xyz = np.array(nodes[key])
        dist = np.linalg.norm(n_xyz - target_xyz_mm)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_dist


for link in suspension_links:
    # Convert inches to mm
    c_pt_mm = np.array(link["chassis_in"]) * 25.4
    u_pt_mm = np.array(link["upright_in"]) * 25.4

    # Find the nearest actual node on your SolidWorks mesh
    node_idx, error_mm = find_closest_node(c_pt_mm)
    print(f"  Mapped '{link['name']}' to Node {node_idx} (Error: {error_mm:.1f}mm)")

    # Vector math: Compression pushes from Upright INTO Chassis
    vector = c_pt_mm - u_pt_mm
    unit_vector = vector / np.linalg.norm(vector)

    # Force components (Fx, Fy, Fz)
    force_components = unit_vector * link["force_N"]

    # Add to global force array
    F[3 * node_idx] += force_components[0]
    F[3 * node_idx + 1] += force_components[1]
    F[3 * node_idx + 2] += force_components[2]

# 5. Solve for Displacements
print("Solving system...")
K_ff = K[np.ix_(free_dofs, free_dofs)]
F_f = F[free_dofs]

try:
    U_f = np.linalg.solve(K_ff, F_f)
except np.linalg.LinAlgError:
    print("Matrix is singular. Using least-squares fallback.")
    U_f, _, _, _ = np.linalg.lstsq(K_ff, F_f, rcond=None)

U = np.zeros(DOF)
for idx, dof in enumerate(free_dofs):
    U[dof] = U_f[idx]

# 6. Calculate Stresses and FOS
print("Calculating Stresses and FOS...")
for mem in members:
    idx_i = get_node_idx(str(mem["n1"]))
    idx_j = get_node_idx(str(mem["n2"]))

    pi = np.array(nodes[str(mem["n1"])])
    pj = np.array(nodes[str(mem["n2"])])
    L = mem["L_true_mm"]

    d = (pj - pi) / L
    delta = float(np.dot(d, U[3 * idx_j: 3 * idx_j + 3] - U[3 * idx_i: 3 * idx_i + 3]))

    A = mem["A_mm2"]
    I = mem["I_mm4"]

    F_ax = (E_MPA * A / L) * delta
    sig = F_ax / A
    Pcr = (math.pi ** 2 * E_MPA * I) / (L ** 2)

    if sig < 0:
        fos = min(SY_MPA / max(abs(sig), 0.001), Pcr / max(abs(F_ax), 0.001))
        mode = "COMP"
    else:
        fos = SY_MPA / max(abs(sig), 0.001)
        mode = "TENS"

    mem["F_N"] = round(float(F_ax), 2)
    mem["sigma_MPa"] = round(float(sig), 3)
    mem["FOS"] = round(float(fos), 2)
    mem["mode"] = mode

output = {
    "nodes": nodes,
    "results": members
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"Success! Analysis saved to {OUTPUT_FILE}.")