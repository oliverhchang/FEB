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
    Calculates the ideal sensitivity (d(chi_0)/d(lambda)) for a perfectly rigid chassis.
    This is a constant value, representing predictable tuning response.
    """
    term1_deriv = (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    term2_deriv = (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])

    # The correct limit of the derivative is A + B
    return term1_deriv + term2_deriv


def main():
    """Main function to generate the LLTD Sensitivity contour plot."""

    # --- Design Goal Parameters ---
    # Define the maximum acceptable non-linearity error for tunability
    sensitivity_error_threshold = 0.1  # Example: 10% maximum acceptable sensitivity error
    # Define the realistic operating range for your suspension setup
    target_lambda_range = [0.4, 0.6]  # e.g., from 40% to 60% front stiffness

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0,
        # Stiffness Parameters [N·m/deg] - Fixed units
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
    if sensitivity_rigid == 0:
        sensitivity_rigid = 1e-9
    sensitivity_error = np.abs(sensitivity_flexible - sensitivity_rigid) / np.abs(sensitivity_rigid)
    max_error_val = np.max(sensitivity_error)

    # --- Step 5: Generate the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    contour_filled = ax.contourf(M, L, sensitivity_error, levels=np.linspace(0, max_error_val, 50), cmap='jet')
    ax.contour(M, L, sensitivity_error, levels=np.linspace(0, max_error_val, 25), colors='black', linewidths=0.5)

    # --- Step 6: Find and Plot the "Sweet Spot" with a more robust method ---
    threshold_contour = ax.contour(M, L, sensitivity_error, levels=[sensitivity_error_threshold], colors='white',
                                   linewidths=3)

    worst_mu = 0.0
    worst_lambda = None

    for i in range(len(lambda_vals)):
        lambda_val = lambda_vals[i]
        if target_lambda_range[0] <= lambda_val <= target_lambda_range[1]:
            error_column = sensitivity_error[:, i]
            above_indices = np.where(error_column >= sensitivity_error_threshold)[0]
            if above_indices.size > 0:
                last_above_idx = above_indices[-1]
                if last_above_idx + 1 < len(mu_vals):
                    mu1, mu2 = mu_vals[last_above_idx], mu_vals[last_above_idx + 1]
                    err1, err2 = error_column[last_above_idx], error_column[last_above_idx + 1]
                    log_mu_interp = np.interp(sensitivity_error_threshold, [err2, err1], [np.log(mu2), np.log(mu1)])
                    mu_interp = np.exp(log_mu_interp)

                    if mu_interp > worst_mu:
                        worst_mu = mu_interp
                        worst_lambda = lambda_val

    # Highlight the target operating range
    ax.axhspan(target_lambda_range[0], target_lambda_range[1], color='white', alpha=0.15, zorder=0)

    if worst_mu > 0:
        kc_sweet_spot_deg = worst_mu * total_roll_stiffness

        ax.axhline(y=worst_lambda, color='white', linestyle='--', lw=2)
        ax.plot([worst_mu, worst_mu], [0, worst_lambda], color='white', linestyle='--', lw=2)
        ax.annotate(
            f'Worst Case λ: {worst_lambda:.2f}\nRequired μ: {worst_mu:.2f}\n(k_c ≈ {kc_sweet_spot_deg:.0f} N·m/deg)',
            xy=(worst_mu, 0), xytext=(worst_mu, 0.1),
            color='white', ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='white'))

    # --- Step 7: Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('λ', fontsize=14, rotation=0, labelpad=15)
    ax.set_title('LLTD Sensitivity Error (Non-Linearity)', fontsize=16)
    ax.set_box_aspect(1)
    ax.set_xlim(0.1, 10)  # Ensure proper x-axis limits
    ax.set_ylim(0, 1)     # Standard y-axis limits

    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(contour_filled, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Sensitivity Error', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()