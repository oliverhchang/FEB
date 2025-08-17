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


def calculate_rigid_sensitivity(params):
    """
    Calculates the ideal sensitivity for a perfectly rigid chassis.
    This is the limit of the derivative of the flexible equation as mu -> infinity.
    """
    # This represents the constant 'A' from the equation's roll moment terms
    term1_deriv_component = (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    # This represents the constant 'B'
    term2_deriv_component = (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])

    # The correct limit of the derivative is A + B
    return term1_deriv_component + term2_deriv_component


def main():
    """Main function to generate the LLTD Sensitivity contour plot."""

    # --- Design Goal Parameters ---
    # Define the maximum acceptable non-linearity error for tunability
    sensitivity_error_threshold = 0.10  # Example: 10% maximum acceptable sensitivity error
    # Define the realistic operating range for your suspension setup
    target_lambda_range = [0.01, 1.0]  # e.g., from 0% to 100% front stiffness

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    # These parameters are now set to be symmetrical to match the reference paper's graph.
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55,
        'a_s': 1.55 / 2,  # Set CG to be perfectly in the middle for 50/50 mass distribution
        'h_G': 0.279, 'z_F': 0.115, 'z_R': 0.165,
        'd_sF': 0.150,  # Set front and rear roll moment arms to be equal
        'd_sR': 0.150,
        'z_uF': 0.12,
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

    # --- Step 3: Set up the Grid for the Contour Plot ---
    lambda_vals = np.linspace(0.01, 1, 200)
    mu_vals = np.logspace(-1, 1, 200)  # 0.1 to 10
    L, M = np.meshgrid(lambda_vals, mu_vals)

    # --- Step 4: Calculate the Sensitivity Error (Non-Linearity Error) ---
    chi_flexible = calculate_LLTD(L, M, vehicle_params)
    sensitivity_flexible = np.gradient(chi_flexible, lambda_vals, axis=1)
    sensitivity_rigid = calculate_rigid_sensitivity(vehicle_params)

    # The rigid sensitivity is now a non-zero constant, so the check is simpler
    if sensitivity_rigid == 0:
        sensitivity_rigid = 1e-9

    sensitivity_error = np.abs(sensitivity_flexible - sensitivity_rigid) / np.abs(sensitivity_rigid)
    max_error_val = np.max(sensitivity_error)

    # --- Step 5: Generate the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    contour_filled = ax.contourf(M, L, sensitivity_error, levels=np.linspace(0, max_error_val, 50), cmap='jet')
    ax.contour(M, L, sensitivity_error, levels=np.linspace(0, max_error_val, 25), colors='black', linewidths=0.5)

    # --- Step 6: Find and Plot the "Sweet Spot" with a more robust method ---
    # Draw the contour for the threshold to visualize the boundary
    threshold_contour = ax.contour(M, L, sensitivity_error, levels=[sensitivity_error_threshold], colors='white',
                                   linewidths=3)

    worst_mu = 0.0
    worst_lambda = None

    # Iterate through each lambda column to find the worst-case mu required
    for i in range(len(lambda_vals)):
        lambda_val = lambda_vals[i]
        if target_lambda_range[0] <= lambda_val <= target_lambda_range[1]:
            error_column = sensitivity_error[:, i]
            # Find the last point where the error is still above the threshold
            above_indices = np.where(error_column >= sensitivity_error_threshold)[0]
            if above_indices.size > 0:
                last_above_idx = above_indices[-1]
                if last_above_idx + 1 < len(mu_vals):
                    # Interpolate to find the precise mu where the error crosses the threshold
                    mu1, mu2 = mu_vals[last_above_idx], mu_vals[last_above_idx + 1]
                    err1, err2 = error_column[last_above_idx], error_column[last_above_idx + 1]
                    log_mu_interp = np.interp(sensitivity_error_threshold, [err2, err1], [np.log(mu2), np.log(mu1)])
                    mu_interp = np.exp(log_mu_interp)

                    if mu_interp > worst_mu:
                        worst_mu = mu_interp
                        worst_lambda = lambda_val

    # Only visualize the target range if it's not the entire plot
    if not (target_lambda_range[0] <= 0.01 and target_lambda_range[1] >= 1.0):
        ax.axhspan(target_lambda_range[0], target_lambda_range[1], color='white', alpha=0.15, zorder=0)

    if worst_mu > 0:
        kc_sweet_spot_rad = worst_mu * total_roll_stiffness
        kc_sweet_spot_deg = worst_mu * total_roll_stiffness * (np.pi / 180)

        ax.axhline(y=worst_lambda, color='white', linestyle='--', lw=2)
        ax.plot([worst_mu, worst_mu], [0, worst_lambda], color='white', linestyle='--', lw=2)
        ax.annotate(
            f'Worst Case λ: {worst_lambda:.2f}\nRequired μ: {worst_mu:.2f}\n(k_c ≈ {kc_sweet_spot_deg:.0f} N·m/deg)',
            xy=(worst_mu, 0), xytext=(worst_mu, -0.1),
            color='white', ha='center', va='top',
            arrowprops=dict(arrowstyle='->', color='white'))

    # --- Step 7: Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('λ', fontsize=14, rotation=0, labelpad=15)
    ax.set_title('LLTD Sensitivity Error (Non-Linearity)', fontsize=16)
    ax.set_box_aspect(1)

    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(contour_filled, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Sensitivity Error', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
