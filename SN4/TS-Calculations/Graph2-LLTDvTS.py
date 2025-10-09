import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_LLTD(lambda_vals, mu, params):
    """
    Calculates the lateral load transfer proportion (chi) using Equation 3.6.
    """
    l = np.copy(lambda_vals)

    # Calculate common denominator for roll-dependent terms
    roll_effect_denominator = l ** 2 - l - mu
    # Avoid division by zero where the denominator might become zero
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
    """Main function to generate LLTD vs Chassis Stiffness Ratio plots."""

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

        # Mass Parameters [kg]
        'm': 310.0,           # Total mass of the vehicle
        'm_uF': 12.0,         # Unsprung mass for a single front corner (wheel, tire, upright, etc.)

        # Stiffness Parameters [N/rad] - Note: These are not directly used in this plot's loops
        'k_f': 240,
        'k_r': 120,
    }

    # --- Step 2: Perform Initial Calculations Based on Parameters ---
    # These calculations are still needed for the terms inside the LLTD function.
    total_unsprung_mass = vehicle_params['m_uF'] * 2
    total_sprung_mass = vehicle_params['m'] - total_unsprung_mass

    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung_mass * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung_mass * vehicle_params['a_s'] / vehicle_params['wheelbase']

    # --- Step 3: Run Simulation and Plot ---

    # Define the range for mu (the x-axis) on a logarithmic scale from 0.1 to 10
    mu_range = np.logspace(-1, 1, 500)  # 10^-1 to 10^1

    # Define the constant lambda values for each line to be plotted
    lambda_values_to_plot = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Set figure size to be square
    plt.figure(figsize=(8, 8))

    # Loop through each constant lambda value
    for lambda_val in lambda_values_to_plot:
        # Calculate chi for the entire mu_range at this constant lambda
        chi_values = calculate_LLTD(lambda_val, mu_range, vehicle_params)
        # Plot mu on the x-axis and chi on the y-axis
        plt.plot(mu_range, chi_values,
                 label=f'λ = {lambda_val:.1f}',
                 linewidth=2)

    # Styling to match the reference graph
    plt.xscale('log')  # Set the x-axis to a logarithmic scale
    plt.title("LLTD vs Chassis Stiffness Normalized", fontsize=16)
    plt.xlabel("μ (Chassis Stiffness Ratio)", fontsize=12)
    plt.ylabel("χ (Front Load Transfer Distribution)", fontsize=12)
    plt.legend(title="Roll Stiffness Distribution (λ)")
    plt.grid(True, which="both", linestyle=':') # Show grid for major and minor ticks
    plt.xlim(0.1, 10)
    plt.ylim(0.2, 0.9)

    # --- Format x-axis to avoid scientific notation and hide minor labels ---
    ax = plt.gca()
    # Manually set major ticks and their labels
    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # Hide the labels for the minor ticks
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
