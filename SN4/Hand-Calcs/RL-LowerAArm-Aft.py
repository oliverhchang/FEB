import numpy as np
import itertools


def calculate_internal_forces(node_coords, tube_endpoints, a_arm_endpoint, applied_force_magnitude):
    """
    Solves for the internal axial forces in three chassis tubes connected at a single node.
    """
    # --- 1. Convert coordinate dictionaries to NumPy arrays ---
    node_pt = np.array([node_coords['x'], node_coords['y'], node_coords['z']])
    a_arm_pt = np.array([a_arm_endpoint['x'], a_arm_endpoint['y'], a_arm_endpoint['z']])
    tube_pts = [np.array([p[1]['x'], p[1]['y'], p[1]['z']]) for p in tube_endpoints]

    # --- 2. Calculate the unit vector for the applied A-arm force ---
    a_arm_vec = node_pt - a_arm_pt
    a_arm_unit_vec = a_arm_vec / np.linalg.norm(a_arm_vec)

    # --- 3. Calculate the full applied force vector ---
    applied_force_vec = applied_force_magnitude * a_arm_unit_vec

    # --- 4. Calculate the unit vectors for the reacting chassis tubes ---
    tube_unit_vectors = []
    for pt in tube_pts:
        vec = pt - node_pt
        unit_vec = vec / np.linalg.norm(vec)
        tube_unit_vectors.append(unit_vec)

    # --- 5. Set up and solve the system of linear equations (Ax = B) ---
    A = np.array(tube_unit_vectors).T
    B = -applied_force_vec

    try:
        tube_forces = np.linalg.solve(A, B)
        return tube_forces
    except np.linalg.LinAlgError:
        return None  # Return None if the combination is geometrically unstable


def run_analysis_for_load_case(title, node_coords, tubes, a_arm_coords, applied_force):
    """
    Runs the full brute-force analysis for a given load case and prints results.
    """
    print(f"\n--- {title} ---")
    print(f"Analyzing Node: {node_coords} for Applied Force: {applied_force:.2f} N\n")

    tube_combinations = list(itertools.combinations(tubes, 3))
    results = []

    for combo in tube_combinations:
        tube_endpoints = list(combo)
        forces = calculate_internal_forces(node_coords, tube_endpoints, a_arm_coords, applied_force)

        if forces is not None:
            load_metric = np.sum(np.abs(forces))
            results.append({
                "combination": [tube[0] for tube in combo],
                "forces": forces,
                "metric": load_metric
            })
        else:
            results.append({
                "combination": [tube[0] for tube in combo],
                "forces": [0, 0, 0],
                "metric": float('inf')
            })

    sorted_results = sorted(results, key=lambda x: x['metric'])

    print("--- Analysis of Tube Combination(s) ---")
    for i, result in enumerate(sorted_results):
        rank = i + 1
        rank_label = f"--- Combination #{rank} |"
        print(f"{rank_label} Total Load Metric: {result['metric']:.2f} N ---")

        combo_names = result['combination']
        forces = result['forces']

        if result['metric'] == float('inf'):
            print(f"  Combination: {combo_names}")
            print("  Result: GEOMETRICALLY UNSTABLE (Cannot solve)")
        else:
            print(f"  Combination:")
            for j in range(3):
                status = "Tension" if forces[j] > 0 else "Compression"
                print(f"    - {combo_names[j]:<35}: {forces[j]:9.2f} N ({status})")
        print("-" * 65)

    if not results:
        print("No tube combinations to analyze.")
    elif sorted_results[0]['metric'] != float('inf'):
        best_combo = sorted_results[0]
        print(f"\nConclusion for {title}:")
        print("The primary load-bearing combination is:")
        for name in best_combo['combination']:
            print(f"  - {name}")
        print(f"This combination has a total internal force of {best_combo['metric']:.2f} N.")
    else:
        print(f"\nConclusion for {title}: The tube combination is geometrically unstable.")
    print("=" * 65)


# --- Input Data for Rear Left Bottom A-Arm (AFT) ---
connection_node = {'x': -5.52, 'y': 7.39, 'z': -2.17}
a_arm_connection = {'x': -62.59, 'y': 22.16, 'z': -3.73}  # Lower Upright
all_tubes = [
    ("Horizontally to fore lower a-arm", {'x': -52.18, 'y': 11.20, 'z': -4.63}),
    ("Vertically to aft upper a-arm", {'x': -62.11, 'y': 10.11, 'z': 3.67}),
    ("Laterally to other side", {'x': -61.76, 'y': -10.30, 'z': -4.35})
]

# Applied Forces from table
peak_compression_force = -4429.986722
peak_tension_force = 5130.701087

# --- Run Full Analysis ---
print("--- Analysis for Rear Left Bottom A-Arm (AFT) ---")
# Run for Peak Compression
run_analysis_for_load_case(
    "Peak Compression Load Case",
    connection_node,
    all_tubes,
    a_arm_connection,
    peak_compression_force
)
# Run for Peak Tension
run_analysis_for_load_case(
    "Peak Tension Load Case",
    connection_node,
    all_tubes,
    a_arm_connection,
    peak_tension_force
)


