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
    """Main function to generate the LLTD vs RSD plot with a highlighted operating window."""

    # --- Tuning Window Parameters ---
    # Define your car's adjustable RSD range and the LLTD range of interest
    rsd_range = [0.38, 0.66]  # Min and Max Roll Stiffness Distribution (lambda)
    lltd_range = [0.4, 0.6]  # Min and Max LLTD of interest (chi)

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

    # --- Step 3: Run Simulation and Plot ---
    lambda_range = np.linspace(0.01, 0.99, 500)
    kc_values_to_plot = [50, 100, 200, 400, 800, 1400, 2000, 6000]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    for kc in kc_values_to_plot:
        mu_val = kc / total_roll_stiffness
        chi_values = calculate_LLTD(lambda_range, mu_val, vehicle_params)
        ax.plot(lambda_range, chi_values,
                label=f'k_c = {kc} N·m/deg (μ ≈ {mu_val:.2f})',
                linewidth=2)

    # --- Step 4: Add Annotations for the Operating Window ---
    # Draw vertical lines for the RSD range
    ax.axvline(x=rsd_range[0], color='black', linestyle='--', lw=1.5)
    ax.axvline(x=rsd_range[1], color='black', linestyle='--', lw=1.5)

    # Draw horizontal lines for the LLTD range
    ax.axhline(y=lltd_range[0], color='black', linestyle='--', lw=1.5)
    ax.axhline(y=lltd_range[1], color='black', linestyle='--', lw=1.5)

    # Add a shaded rectangle to highlight the operating box
    rect_width = rsd_range[1] - rsd_range[0]
    rect_height = lltd_range[1] - lltd_range[0]
    operating_box = plt.Rectangle((rsd_range[0], lltd_range[0]), rect_width, rect_height,
                                  color='grey', alpha=0.15, zorder=0)
    ax.add_patch(operating_box)

    # Add text labels for the lines
    text_props = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='white')
    ax.text(rsd_range[0] + 0.01, 0.70, f'Min RSD: {rsd_range[0]}', color='white', ha='left', va='center',
            bbox=text_props)
    ax.text(rsd_range[1] - 0.01, 0.73, f'Max RSD: {rsd_range[1]}', color='white', ha='right', va='center',
            bbox=text_props)
    ax.text(0.98, lltd_range[0] + 0.01, f'Min LLTD: {lltd_range[0]}', color='white', ha='right', va='bottom',
            bbox=text_props)
    ax.text(0.98, lltd_range[1] - 0.01, f'Max LLTD: {lltd_range[1]}', color='white', ha='right', va='top',
            bbox=text_props)

    # --- Step 5: Styling ---
    ax.set_title("LLTD vs RSD with Highlighted Operating Window", fontsize=16)
    ax.set_xlabel("λ (Roll Stiffness Distribution)", fontsize=12)
    ax.set_ylabel("χ (Front Load Transfer Distribution)", fontsize=12)
    ax.legend(title="Chassis Stiffness")
    ax.grid(True, linestyle=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.75)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
