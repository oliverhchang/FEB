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
    """Main function to generate the LLTD and Error plots."""

    # --- Tuning Window Parameters based on SN4 Data ---
    # Min RSD corresponds to soft front ARB / hard rear ARB
    # Max RSD corresponds to hard front ARB / soft rear ARB
    physical_rsd_range = [0.36, 0.61]

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0,
        # Stiffness Parameters [N·m/deg] - From SN4 Constants
        'k_f': 275,  # Nominal front roll stiffness (springs only)
        'k_r': 292,  # Nominal rear roll stiffness (springs only)
    }

    # --- Step 2: Perform Initial Calculations Based on Parameters ---
    total_unsprung_mass = vehicle_params['m_uF'] * 2
    total_sprung_mass = vehicle_params['m'] - total_unsprung_mass
    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung_mass * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung_mass * vehicle_params['a_s'] / vehicle_params['wheelbase']
    total_roll_stiffness = vehicle_params['k_f'] + vehicle_params['k_r']

    # --- Step 3: Calculate LLTD Boundaries for a Sweep of Chassis Stiffnesses ---
    kc_sweep = np.logspace(np.log10(50), np.log10(8000), 200)
    lltd_at_min_rsd = []
    lltd_at_max_rsd = []

    for kc in kc_sweep:
        mu_val = kc / total_roll_stiffness
        lltd_min = calculate_LLTD(physical_rsd_range[0], mu_val, vehicle_params)
        lltd_max = calculate_LLTD(physical_rsd_range[1], mu_val, vehicle_params)
        lltd_at_min_rsd.append(lltd_min)
        lltd_at_max_rsd.append(lltd_max)

    # Convert lists to numpy arrays for calculations
    lltd_at_min_rsd = np.array(lltd_at_min_rsd)
    lltd_at_max_rsd = np.array(lltd_at_max_rsd)

    # --- Step 4: Generate the First Plot (Original) ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    # Plot the boundaries of the tuning window
    ax1.plot(kc_sweep, lltd_at_min_rsd, color='cornflowerblue', lw=2.5, label='Min RSD (λ = 0.36)')
    ax1.plot(kc_sweep, lltd_at_max_rsd, color='salmon', lw=2.5, label='Max RSD (λ = 0.61)')

    # Robustly fill the area between the two lines
    lower_bound = np.minimum(lltd_at_min_rsd, lltd_at_max_rsd)
    upper_bound = np.maximum(lltd_at_min_rsd, lltd_at_max_rsd)
    ax1.fill_between(kc_sweep, lower_bound, upper_bound, color='grey', alpha=0.2, label='Tuning Window')

    # Styling for the first plot
    # ax1.set_xscale('log') # Removed to make x-axis linear
    ax1.set_title("LLTD Tuning Window vs. Chassis Torsional Stiffness", fontsize=16, pad=15)
    ax1.set_xlabel("Chassis Torsional Stiffness (k_c) [N·m/deg]", fontsize=12)
    ax1.set_ylabel("LLTD (χ)", fontsize=12)
    ax1.legend()
    ax1.grid(True, which="both", linestyle=':')
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    plt.tight_layout()
    plt.show()

    # --- Step 5: Calculate Error Compared to Infinitely Rigid Chassis ---

    # Calculate the reference LLTD for an "infinitely" stiff chassis
    mu_infinite = 1e9
    lltd_inf_min = calculate_LLTD(physical_rsd_range[0], mu_infinite, vehicle_params)
    lltd_inf_max = calculate_LLTD(physical_rsd_range[1], mu_infinite, vehicle_params)

    # Calculate percentage error for the swept values
    error_min_rsd = ((lltd_at_min_rsd - lltd_inf_min) / lltd_inf_min) * 100
    error_max_rsd = ((lltd_at_max_rsd - lltd_inf_max) / lltd_inf_max) * 100

    # Calculate the total error range (width of the error window)
    lower_error_bound = np.minimum(error_min_rsd, error_max_rsd)
    upper_error_bound = np.maximum(error_min_rsd, error_max_rsd)
    total_error_range = upper_error_bound - lower_error_bound

    # --- Step 6: Generate the Second Plot (Error Plot) ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    # Plot the single cumulative error range line
    ax2.plot(kc_sweep, total_error_range, color='darkviolet', lw=2.5, label='Total LLTD Error Range')

    # Styling for the second plot
    # ax2.set_xscale('log') # Removed to make x-axis linear
    ax2.set_title("Total LLTD Error Range vs. Chassis Torsional Stiffness", fontsize=16, pad=15)
    ax2.set_xlabel("Chassis Torsional Stiffness (k_c) [N·m/deg]", fontsize=12)
    ax2.set_ylabel("Total Error Range (Upper - Lower Bound) [%]", fontsize=12)
    ax2.legend()
    ax2.grid(True, which="both", linestyle=':')

    # Format axes ticks
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

