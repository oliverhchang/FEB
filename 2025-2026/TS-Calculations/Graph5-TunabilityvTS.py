import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_LLTD(lambda_vals, mu, params):
    """
    Calculates the lateral load transfer proportion using Equation 3.6.
    """
    l = np.copy(lambda_vals)

    # Calculate common denominator for roll-dependent terms
    roll_effect_denominator = l ** 2 - l - mu

    # FIX: Handle both single float and numpy array inputs to prevent TypeError
    if isinstance(roll_effect_denominator, np.ndarray):
        roll_effect_denominator[roll_effect_denominator == 0] = 1e-9
    elif roll_effect_denominator == 0:
        roll_effect_denominator = 1e-9

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
    """Main function to generate the LLTD Tuning Window vs. Chassis Stiffness plot."""

    # --- Tuning Window Parameters ---
    # Define your car's physical adjustment range
    physical_rsd_range = [0.38, 0.66]  # The car's actual min/max lambda

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0,
        # Stiffness Parameters [N·m/rad]
        'k_f': 240,
        'k_r': 176,
    }

    # --- Step 2: Perform Initial Calculations Based on Parameters ---
    total_unsprung_mass = vehicle_params['m_uF'] * 2
    total_sprung_mass = vehicle_params['m'] - total_unsprung_mass
    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung_mass * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung_mass * vehicle_params['a_s'] / vehicle_params['wheelbase']
    total_roll_stiffness = vehicle_params['k_f'] + vehicle_params['k_r']

    # --- Step 3: Calculate LLTD Boundaries for a Sweep of Chassis Stiffnesses ---
    kc_sweep = np.logspace(np.log10(50), np.log10(8000), 200)  # Sweep from 50 to 8000 Nm/deg
    lltd_min_boundary = []
    lltd_max_boundary = []

    for kc in kc_sweep:
        mu_val = kc / total_roll_stiffness

        # Calculate LLTD at the softest and hardest ARB settings
        lltd_at_min_rsd = calculate_LLTD(physical_rsd_range[0], mu_val, vehicle_params)
        lltd_at_max_rsd = calculate_LLTD(physical_rsd_range[1], mu_val, vehicle_params)

        lltd_min_boundary.append(lltd_at_min_rsd)
        lltd_max_boundary.append(lltd_at_max_rsd)

    # --- Step 4: Generate the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the boundaries of the tuning window
    ax.plot(kc_sweep, lltd_min_boundary, color='cornflowerblue', lw=2.5, label='Softest ARB Setting (Min λ)')
    ax.plot(kc_sweep, lltd_max_boundary, color='salmon', lw=2.5, label='Hardest ARB Setting (Max λ)')

    # Fill the area between the lines to represent the tuning window
    ax.fill_between(kc_sweep, lltd_min_boundary, lltd_max_boundary, color='grey', alpha=0.2, label='Tuning Window')

    # --- Step 5: Styling ---
    ax.set_xscale('log')
    ax.set_title("LLTD Tuning Window vs. Chassis Torsional Stiffness", fontsize=16)
    ax.set_xlabel("Chassis Torsional Stiffness (k_c) [N·m/deg]", fontsize=12)
    ax.set_ylabel("LLTD (χ)", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", linestyle=':')

    # Format x-axis ticks to be regular numbers
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
