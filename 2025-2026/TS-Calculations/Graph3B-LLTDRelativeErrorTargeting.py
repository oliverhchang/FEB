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
    """Main function to generate the LLTD error contour plot with targeting."""

    # --- Design Goal Parameters ---
    # Define the maximum acceptable non-linearity error for tunability
    error_threshold = 0.04  # Example: 5% maximum acceptable relative error
    # The search for the worst-case scenario will now cover all lambda values.
    # target_lambda_range = [0.4, 0.6] # This is no longer used.

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0,
        # Stiffness Parameters [N·m/deg] - for annotation
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
    max_error_val = np.max(error)

    # --- Step 5: Generate the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    contour_filled = ax.contourf(M, L, error, levels=np.linspace(0, max_error_val, 50), cmap='jet')
    ax.contour(M, L, error, levels=np.linspace(0, max_error_val, 25), colors='black', linewidths=0.5)

    # --- Step 6: Find and Plot the "Sweet Spot" across the entire lambda range ---
    ax.contour(M, L, error, levels=[error_threshold], colors='white', linewidths=3)

    worst_mu = 0.0
    worst_lambda = None

    # Iterate through each lambda value to find the global worst-case
    for i in range(len(lambda_vals)):
        lambda_val = lambda_vals[i]
        # Get the column of error values for the current lambda
        error_column = error[:, i]
        # Find where the error is above the threshold
        above_indices = np.where(error_column >= error_threshold)[0]

        if above_indices.size > 0:
            # Find the last point *above* the threshold. The crossover point to a "good"
            # region (lower mu) will be just after this.
            last_above_idx = above_indices[-1]
            if last_above_idx + 1 < len(mu_vals):
                # Interpolate to find the precise mu value at the threshold crossing
                mu1, mu2 = mu_vals[last_above_idx], mu_vals[last_above_idx + 1]
                err1, err2 = error_column[last_above_idx], error_column[last_above_idx + 1]

                # Perform interpolation in log space for mu for better accuracy
                log_mu_interp = np.interp(error_threshold, [err2, err1], [np.log(mu2), np.log(mu1)])
                mu_interp = np.exp(log_mu_interp)

                # If this interpolated mu is the highest we've seen, it's our new "worst-case"
                if mu_interp > worst_mu:
                    worst_mu = mu_interp
                    worst_lambda = lambda_val

    # If a worst-case scenario was found, annotate it
    if worst_mu > 0:
        kc_sweet_spot_deg = worst_mu * total_roll_stiffness

        # Draw lines to the worst-case point
        ax.axhline(y=worst_lambda, color='white', linestyle='--', lw=2)
        ax.plot([worst_mu, worst_mu], [0, worst_lambda], color='white', linestyle='--', lw=2)

        # Add the annotation text
        ax.annotate(
            f'Worst Case λ: {worst_lambda:.2f}\nRequired μ: {worst_mu:.2f}\n(k_c ≈ {kc_sweet_spot_deg:.0f} N·m/deg)',
            xy=(worst_mu, 0), xytext=(worst_mu, 0.1),
            color='white', ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='white'))

    # --- Step 7: Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('λ', fontsize=14, rotation=0, labelpad=15)
    ax.set_title('Relative Error in LLTD |χ - χ₀| / |χ₀|', fontsize=16)
    ax.set_box_aspect(1)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 1)

    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(contour_filled, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Error Magnitude', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()