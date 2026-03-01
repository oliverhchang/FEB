import math

# ----------------------------------------------------------------------
# --- 1. INPUTS: MODIFY THESE VALUES TO TEST DIFFERENT DESIGNS ---
# ----------------------------------------------------------------------

# --- Material Properties (Using 4130 Steel, Normalized) ---
# (205 GPa = 205e9 Pa, 435 MPa = 435e6 Pa)
MODULUS_OF_ELASTICITY_PA = 205e9  # (E) Modulus of Elasticity in Pascals (N/m^2)
YIELD_STRENGTH_PA = 435e6  # (sigma_y) Yield Strength in Pascals (N/m^2)

# --- Tube Geometry (millimeters) ---
# (1.25 in = 31.75 mm, 0.049 in = 1.2446 mm)
OUTER_DIAMETER_MM = 31.75
WALL_THICKNESS_MM = 1.2446

# --- Beam & Load Setup ---
TOTAL_LENGTH_MM = 450.0  # (L) Total end-to-end length of the tube
LOAD_DISTANCE_MM = 60.0  # (a) Distance from welded end to the load tab
INVERTER_TOTAL_MASS_KG = 7.5  # Total mass of the component
NUM_TABS_TOTAL = 4  # Total number of tabs supporting the mass

# --- Dynamic Analysis (FSAE) ---
G_FACTOR = 10.0  # The 'g' multiplier for dynamic loads (e.g., 10g bump)

# ----------------------------------------------------------------------
# --- CONSTANTS (DO NOT MODIFY) ---
# ----------------------------------------------------------------------
M_PER_MM = 0.001
NEWTONS_PER_KG = 9.80665  # Standard gravity


# ----------------------------------------------------------------------
# --- HELPER FUNCTIONS ---
# ----------------------------------------------------------------------

def calculate_moment_of_inertia(D_m: float, t_m: float) -> float:
    """
    Calculates the Area Moment of Inertia (I) for a hollow circular tube.
    All inputs must be in METERS.
    D_m = Outer Diameter (meters)
    t_m = Wall Thickness (meters)
    Returns Moment of Inertia in m^4.
    """
    d_m = D_m - (2.0 * t_m)  # Calculate inner diameter
    I_m4 = (math.pi / 64.0) * (D_m ** 4 - d_m ** 4)
    return I_m4


def calculate_bending_stress(M_Nm: float, I_m4: float, D_m: float) -> float:
    """
    Calculates the maximum bending stress (sigma) for a circular tube.
    All inputs must be in METERS, NEWTONS.
    M_Nm = Bending Moment (N-m)
    I_m4 = Moment of Inertia (m^4)
    D_m  = Outer Diameter (meters)
    Returns Bending Stress in Pascals (Pa or N/m^2).
    """
    c_m = D_m / 2.0  # Distance from neutral axis to outer fiber (meters)
    sigma_pa = (M_Nm * c_m) / I_m4
    return sigma_pa


# ----------------------------------------------------------------------
# --- MAIN CALCULATION SCRIPT ---
# ----------------------------------------------------------------------

def main():
    """
    Runs the full beam analysis based on the inputs defined at the top.
    """

    # --- 1. Unit Conversions (Convert all inputs to meters) ---
    L_m = TOTAL_LENGTH_MM * M_PER_MM
    a_m = LOAD_DISTANCE_MM * M_PER_MM
    D_m = OUTER_DIAMETER_MM * M_PER_MM
    t_m = WALL_THICKNESS_MM * M_PER_MM

    # --- 2. Load Calculations (in Newtons) ---
    mass_per_tab_kg = INVERTER_TOTAL_MASS_KG / NUM_TABS_TOTAL
    P_static_N = mass_per_tab_kg * NEWTONS_PER_KG
    P_dynamic_N = P_static_N * G_FACTOR

    # --- 3. Geometry Calculation ---
    try:
        I_m4 = calculate_moment_of_inertia(D_m, t_m)
        if I_m4 <= 0:
            print("Error: Invalid tube geometry. Wall thickness may be too large.")
            return
    except Exception as e:
        print(f"Error calculating geometry: {e}")
        return

    # --- 4. Static Analysis (Car at rest) ---
    # M_max = P*a*(L-a) / L
    M_max_static_Nm = (P_static_N * a_m * (L_m - a_m)) / L_m
    sigma_static_pa = calculate_bending_stress(M_max_static_Nm, I_m4, D_m)
    fos_static = YIELD_STRENGTH_PA / sigma_static_pa

    # v_max = (P*a^2)/(24*E*I) * (3*L - 4*a)
    v_max_static_m = (P_static_N * a_m ** 2) / (24 * MODULUS_OF_ELASTICITY_PA * I_m4) * (3 * L_m - 4 * a_m)

    # --- 5. Dynamic Analysis (FSAE g-load) ---
    M_max_dynamic_Nm = (P_dynamic_N * a_m * (L_m - a_m)) / L_m
    sigma_dynamic_pa = calculate_bending_stress(M_max_dynamic_Nm, I_m4, D_m)
    fos_dynamic = YIELD_STRENGTH_PA / sigma_dynamic_pa

    v_max_dynamic_m = (P_dynamic_N * a_m ** 2) / (24 * MODULUS_OF_ELASTICITY_PA * I_m4) * (3 * L_m - 4 * a_m)

    # --- 6. Print Results ---
    print("--- FSAE BEAM ANALYSIS (METRIC) ---")
    print("\n--- Inputs ---")
    print(f"  Material: 4130 Steel")
    print(f"  Yield Strength: {YIELD_STRENGTH_PA / 1e6:.1f} MPa")
    print(f"  Modulus (E): {MODULUS_OF_ELASTICITY_PA / 1e9:.1f} GPa")
    print(f"  Tube OD: {OUTER_DIAMETER_MM:.3f} mm")
    print(f"  Wall: {WALL_THICKNESS_MM:.3f} mm")
    print(f"  Beam Length (L): {L_m:.3f} m ({TOTAL_LENGTH_MM} mm)")
    print(f"  Load Distance (a): {a_m:.3f} m ({LOAD_DISTANCE_MM} mm)")

    print("\n--- Calculated Properties ---")
    print(f"  Moment of Inertia (I): {I_m4 * 1e12:.2f} mm^4  (or {I_m4:.2e} m^4)")
    print(f"  Static Load (P_stat): {P_static_N:.2f} N (per tab)")
    print(f"  Dynamic Load (P_dyn): {P_dynamic_N:.2f} N (@ {G_FACTOR}g)")

    print("\n--- STATIC ANALYSIS (Car at rest) ---")
    print(f"  Max Bending Moment: {M_max_static_Nm:.2f} N-m")
    print(f"  Max Bending Stress: {sigma_static_pa / 1e6:.3f} MPa")  # Convert Pa to MPa
    print(f"  Max Deflection: {v_max_static_m * 1000:.6f} mm")  # Convert m to mm
    print(f"  Factor of Safety: {fos_static:,.0f}")

    print("\n--- DYNAMIC ANALYSIS ({G_FACTOR}g load) ---")
    print(f"  Max Bending Moment: {M_max_dynamic_Nm:.2f} N-m")
    print(f"  Max Bending Stress: {sigma_dynamic_pa / 1e6:.2f} MPa")  # Convert Pa to MPa
    print(f"  Max Deflection: {v_max_dynamic_m * 1000:.6f} mm")  # Convert m to mm
    print(f"  Factor of Safety: {fos_dynamic:.1f}")
    print("---------------------------------")


if __name__ == "__main__":
    main()