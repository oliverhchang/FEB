import numpy as np
from itertools import combinations


def calculate_internal_forces(node_coords, tube_endpoints, a_arm_endpoint, applied_force_magnitude):
    """
    Solves for the internal axial forces in three chassis tubes connected at a single node.

    This function uses the Method of Joints for a 3D truss. It assumes that the
    system is statically determinate (3 unknown tube forces, 3 equilibrium equations).

    Args:
        node_coords (dict): A dictionary with 'x', 'y', 'z' keys for the connection node.
        tube_endpoints (list of dicts): A list containing three dictionaries, each with 'x', 'y', 'z'
                                       keys for the endpoint of a chassis tube.
        a_arm_endpoint (dict): A dictionary with 'x', 'y', 'z' for the A-arm's connection point.
        applied_force_magnitude (float): The magnitude of the force applied by the A-arm.

    Returns:
        numpy.ndarray or None: An array containing the calculated axial forces for the three tubes,
                              or None if the system is singular.
    """
    # --- 1. Convert coordinate dictionaries to NumPy arrays ---
    node_pt = np.array([node_coords['x'], node_coords['y'], node_coords['z']])
    a_arm_pt = np.array([a_arm_endpoint['x'], a_arm_endpoint['y'], a_arm_endpoint['z']])
    tube_pts = [np.array([p['x'], p['y'], p['z']]) for p in tube_endpoints]

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

    # --- 5. Set up the system of linear equations (Ax = B) ---
    A = np.array(tube_unit_vectors).T  # Transpose to get unit vectors in columns
    B = -applied_force_vec

    # --- 6. Solve for the unknown tube forces ---
    try:
        tube_forces = np.linalg.solve(A, B)
        return tube_forces
    except np.linalg.LinAlgError:
        return None


def analyze_tube_combination(combo_indices, tube_names, all_tubes, connection_node, a_arm_connection, force_magnitude):
    """
    Analyze a specific combination of 3 tubes and return results.
    """
    # Get the 3 tubes for this combination
    selected_tubes = [all_tubes[i] for i in combo_indices]
    selected_names = [tube_names[i] for i in combo_indices]

    # Calculate forces
    forces = calculate_internal_forces(connection_node, selected_tubes, a_arm_connection, force_magnitude)

    if forces is None:
        return None  # Singular system

    # Calculate total load magnitude (sum of absolute values)
    total_load = np.sum(np.abs(forces))

    return {
        'combination_indices': combo_indices,
        'tube_names': selected_names,
        'forces': forces,
        'total_load': total_load
    }


def brute_force_analysis(connection_node, all_tubes, tube_names, a_arm_connection, force_magnitude, force_case_name):
    """
    Analyze all possible combinations of 3 tubes and rank by total load.
    """
    print(f"\n=== {force_case_name} Analysis ===")
    print(f"Applied Force: {force_magnitude:.2f} N")
    print(f"Analyzing all combinations of 3 tubes from {len(all_tubes)} available tubes...\n")

    valid_combinations = []

    # Generate all combinations of 3 tubes from the available tubes
    for combo_indices in combinations(range(len(all_tubes)), 3):
        result = analyze_tube_combination(combo_indices, tube_names, all_tubes,
                                          connection_node, a_arm_connection, force_magnitude)

        if result is not None:  # Valid (non-singular) combination
            valid_combinations.append(result)

    if not valid_combinations:
        print("No valid combinations found! All combinations result in singular systems.")
        return None

    # Sort by total load (descending - highest load first)
    valid_combinations.sort(key=lambda x: x['total_load'], reverse=True)

    print(f"Found {len(valid_combinations)} valid combinations:")
    print("Ranked by Total Load (highest to lowest):\n")

    for rank, combo in enumerate(valid_combinations, 1):
        print(f"Rank {rank}: Total Load = {combo['total_load']:.2f} N")
        print(f"  Tubes: {', '.join(combo['tube_names'])}")
        print(f"  Individual Forces:")
        for i, (name, force) in enumerate(zip(combo['tube_names'], combo['forces'])):
            status = "Tension" if force > 0 else "Compression"
            print(f"    {name}: {force:9.2f} N ({status})")
        print()

    return valid_combinations


# --- Input Data ---
# Node of Interest on the Chassis
connection_node = {'x': 3.88, 'y': 9.21, 'z': 4.20}

# All 5 chassis tubes with updated coordinates
all_tube_endpoints = [
    {'x': 25.8, 'y': 7.39, 'z': 10.77},  # Tube #1: Front bulkhead top
    {'x': -1.77, 'y': 8.64, 'z': 15.57},  # Tube #2: FRH top
    {'x': -3.87, 'y': 9.85, 'z': 6.45},  # Tube #3: Other upper a-arm mount
    {'x': 6.11, 'y': 7.39, 'z': -2.17},  # Tube #4: To bottom
    {'x': 25.80, 'y': 7.39, 'z': -2.17}  # Tube #5: Front bulkhead bottom
]

# Tube names for easy identification
tube_names = [
    "Front Bulkhead Top",
    "FRH Top",
    "Other Upper A-Arm Mount",
    "To Bottom",
    "Front Bulkhead Bottom"
]

# Connection point on the suspension upright/A-arm
a_arm_connection = {'x': -0.15, 'y': 21.68, 'z': 3.73}

# Applied Forces
peak_compression_force = -2879.67
peak_tension_force = 569.42

# --- Main Analysis ---
print("=== BRUTE FORCE CHASSIS TUBE LOAD ANALYSIS ===")
print(f"Connection Node: ({connection_node['x']}, {connection_node['y']}, {connection_node['z']})")
print(f"A-Arm Connection: ({a_arm_connection['x']}, {a_arm_connection['y']}, {a_arm_connection['z']})")
print(f"Total tubes available: {len(all_tube_endpoints)}")
print(f"Total combinations to analyze: {len(list(combinations(range(len(all_tube_endpoints)), 3)))}")

# Analyze compression case
compression_results = brute_force_analysis(
    connection_node, all_tube_endpoints, tube_names,
    a_arm_connection, peak_compression_force, "Peak Compression"
)

# Analyze tension case
tension_results = brute_force_analysis(
    connection_node, all_tube_endpoints, tube_names,
    a_arm_connection, peak_tension_force, "Peak Tension"
)

# Summary of most critical combinations
print("\n=== SUMMARY ===")
if compression_results:
    top_compression = compression_results[0]
    print(f"Most Load-Bearing Combination (Compression): {', '.join(top_compression['tube_names'])}")
    print(f"  Total Load: {top_compression['total_load']:.2f} N")

if tension_results:
    top_tension = tension_results[0]
    print(f"Most Load-Bearing Combination (Tension): {', '.join(top_tension['tube_names'])}")
    print(f"  Total Load: {top_tension['total_load']:.2f} N")