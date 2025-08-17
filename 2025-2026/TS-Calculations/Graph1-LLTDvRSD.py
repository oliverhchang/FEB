import numpy as np
import matplotlib.pyplot as plt


def calculate_LLTD(lambda_vals, mu, params):
    """
    Calculates the lateral load transfer proportion using Equation 3.6.
    """
    l = np.copy(lambda_vals)

    # Calculate common denominator for roll-dependent terms
    roll_effect_denominator = l ** 2 - l - mu
    roll_effect_denominator[roll_effect_denominator == 0] = 1e-9

    # --- Term 1: Front sprung mass roll moment contribution ---
    front_roll_moment_modifier = (l ** 2 - (mu + 1) * l) / roll_effect_denominator
    front_sprung_mass_physical_ratio = (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    front_roll_moment_contribution = front_roll_moment_modifier * front_sprung_mass_physical_ratio

    # --- Term 2: Rear sprung mass roll moment contribution (transmitted through chassis) ---
    rear_roll_moment_modifier = (mu * l) / roll_effect_denominator
    rear_sprung_mass_physical_ratio = (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])
    rear_roll_moment_contribution = rear_roll_moment_modifier * rear_sprung_mass_physical_ratio

    # --- Term 3: Direct geometric load transfer from front roll center ---
    front_roll_center_geometric_contribution = (params['z_F'] * params['m_sF']) / (params['h_G'] * params['m'])

    # --- Term 4: Direct geometric load transfer from front unsprung mass ---
    front_unsprung_mass_geometric_contribution = (params['z_uF'] * params['m_uF']) / (params['h_G'] * params['m'])

    # --- Sum all components ---
    LLTD = (front_roll_moment_contribution -
                 rear_roll_moment_contribution +
                 front_roll_center_geometric_contribution +
                 front_unsprung_mass_geometric_contribution)

    return LLTD


def main():

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55,    # Distance between front and rear axle centerlines
        'a_s': 0.7525,        # Longitudinal distance from the overall CG to the front axle
        'h_G': 0.279,         # Height of the vehicle's overall center of gravity from the ground
        'z_F': 0.115,         # Height of the front roll center from the ground
        'z_R': 0.165,         # Height of the rear roll center from the ground
        'd_sF': 0.150,        # Vertical distance between sprung mass CG and front roll center (front roll moment arm)
        'd_sR': 0.130,        # Vertical distance between sprung mass CG and rear roll center (rear roll moment arm)
        'z_uF': 0.12,         # Height of the front unsprung mass CG from the ground

        # Mass Parameters [kg]]
        'm': 310.0,           # Total mass of the vehicle
        'm_uF': 12.0,         # Unsprung mass for a single front corner (wheel, tire, upright, etc.)

        # Stiffness Parameters [Nm/deg
        'k_f': 240,           # Front suspension roll stiffness (from springs, anti-roll bars, etc.)
        'k_r': 120,           # Rear suspension roll stiffness (from springs, anti-roll bars, etc.)
    }

    # --- Step 2: Perform Initial Calculations Based on Parameters ---

    total_unsprung_mass = vehicle_params['m_uF'] * 2
    total_sprung_mass = vehicle_params['m'] - total_unsprung_mass

    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung_mass * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung_mass * vehicle_params['a_s'] / vehicle_params['wheelbase']

    total_roll_stiffness = vehicle_params['k_f'] + vehicle_params['k_r']

    # --- Step 3: Run Simulation and Plot ---

    lambda_range = np.linspace(0.01, 0.99, 500)
    kc_values_to_plot = [50, 100, 200, 400, 800, 1400, 2000, 6000]

    # Set figure size to be square
    plt.figure(figsize=(8, 8))

    for kc in kc_values_to_plot:
        mu_val = kc / total_roll_stiffness
        chi_values = calculate_LLTD(lambda_range, mu_val, vehicle_params)
        plt.plot(lambda_range, chi_values,
                 label=f'k_c = {kc} N·m/deg (μ ≈ {mu_val:.2f})',
                 linewidth=2)

    # Styling
    plt.title("LLTD vs RSD", fontsize=16)
    plt.xlabel("λ (Roll Stiffness Distribution)", fontsize=12)
    plt.ylabel("χ (Front Load Transfer Distribution)", fontsize=12)
    plt.legend(title="Chassis Stiffness")
    plt.grid(True, linestyle=':')
    plt.xlim(0, 1)
    plt.ylim(0.2, 0.75)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
