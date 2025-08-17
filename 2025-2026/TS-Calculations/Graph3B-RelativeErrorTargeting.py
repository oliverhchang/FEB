import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_LLTD(l, mu, params):
    """
    Calculates the lateral load transfer proportion (chi) for a flexible chassis.
    """
    # Create a mask for the denominator to avoid division by zero
    denominator = l ** 2 - l - mu
    # Use a very small number where the denominator is zero
    denominator[denominator == 0] = 1e-9

    term1 = ((l ** 2 - (mu + 1) * l) / denominator) * (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    term2 = (mu * l / denominator) * (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])
    term3 = (params['z_F'] * params['m_sF']) / (params['h_G'] * params['m'])
    term4 = (params['z_uF'] * params['m_uF']) / (params['h_G'] * params['m'])

    return term1 - term2 + term3 + term4


def calculate_LLTD_rigid(l, params):
    """
    Calculates the ideal lateral load transfer proportion (chi_0) for a perfectly rigid chassis.
    This is the limit of calculate_LLTD as mu approaches infinity.
    """
    term1_rigid = l * (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    term2_rigid = -l * (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])
    term3 = (params['z_F'] * params['m_sF']) / (params['h_G'] * params['m'])
    term4 = (params['z_uF'] * params['m_uF']) / (params['h_G'] * params['m'])

    return term1_rigid - term2_rigid + term3 + term4


def main():
    """Main function to generate the LLTD error contour plot."""

    # --- Design Goal Parameters ---
    # Define the maximum acceptable error
    error_threshold = 0.03

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0,
        # Stiffness Parameters [N·m/deg]
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

    # --- Step 3: Set up the Grid for the Contour Plot ---
    lambda_vals = np.linspace(0.01, 1, 200)
    mu_vals = np.logspace(-1, 1, 200)  # 0.1 to 10
    L, M = np.meshgrid(lambda_vals, mu_vals)

    # --- Step 4: Calculate the Error Metric ---
    chi = calculate_LLTD(L, M, vehicle_params)
    chi_zero = calculate_LLTD_rigid(L, vehicle_params)

    # Avoid division by zero in the error calculation
    chi_zero[chi_zero == 0] = 1e-9
    error = np.abs(chi - chi_zero) / np.abs(chi_zero)

    # Dynamically set the color limit based on the maximum error in the data
    max_error_val = np.max(error)

    # --- Step 5: Generate the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    contour_filled = ax.contourf(M, L, error, levels=np.linspace(0, max_error_val, 50), cmap='jet')
    ax.contour(M, L, error, levels=np.linspace(0, max_error_val, 25), colors='black', linewidths=0.5)

    # --- Step 6: Find and Plot the "Sweet Spot" ---
    # Draw the contour for the threshold and get its path data
    threshold_contour = ax.contour(M, L, error, levels=[error_threshold], colors='white', linewidths=3)

    mu_sweet_spot = None
    lambda_at_sweet_spot = None

    # Extract the path data from the contour object
    if threshold_contour.allsegs[0]:
        # Find the rightmost point of the contour line (max mu)
        all_points = np.vstack(threshold_contour.allsegs[0])
        rightmost_point_index = np.argmax(all_points[:, 0])
        mu_sweet_spot, lambda_at_sweet_spot = all_points[rightmost_point_index]

    if mu_sweet_spot is not None:
        # Calculate the corresponding torsional stiffness in N·m/rad
        kc_sweet_spot_deg = mu_sweet_spot * total_roll_stiffness
        # Convert to N·m/deg for the second annotation

        # Plot the dashed lines using the perfectly calculated intersection point
        ax.axhline(y=lambda_at_sweet_spot, color='white', linestyle='--', lw=2)
        ax.plot([mu_sweet_spot, mu_sweet_spot], [0, lambda_at_sweet_spot], color='white', linestyle='--', lw=2)
        ax.plot([0.1, mu_sweet_spot], [lambda_at_sweet_spot, lambda_at_sweet_spot], color='white', linestyle='--', lw=2)
        ax.annotate(
            f'Optimal λ ≈ {lambda_at_sweet_spot:.2f}\nMin. μ ≈ {mu_sweet_spot:.2f}\n(k_c ≈ {kc_sweet_spot_deg:.0f} N·m/deg)',
            xy=(mu_sweet_spot, 0), xytext=(mu_sweet_spot, 0.05),
            color='white', ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='white'))

    # --- Step 7: Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('λ', fontsize=14, rotation=0, labelpad=15)
    ax.set_title('Relative Error in LLTD |χ - χ₀| / |χ₀|', fontsize=16)
    ax.set_box_aspect(1)

    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(contour_filled, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Error Magnitude', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
