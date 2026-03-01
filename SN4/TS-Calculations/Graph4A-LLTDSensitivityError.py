import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_LLTD(l, mu, params):
    """Calculates the lateral load transfer proportion (chi) for a flexible chassis."""
    denominator = l ** 2 - l - mu
    denominator[denominator == 0] = 1e-9
    term1 = ((l ** 2 - (mu + 1) * l) / denominator) * (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    term2 = (mu * l / denominator) * (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])
    term3 = (params['z_F'] * params['m_sF']) / (params['h_G'] * params['m'])
    term4 = (params['z_uF'] * params['m_uF']) / (params['h_G'] * params['m'])
    return term1 - term2 + term3 + term4


def calculate_rigid_sensitivity(params):
    """Calculates the ideal sensitivity for a perfectly rigid chassis."""
    term1_deriv = (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    term2_deriv = (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])
    return term1_deriv + term2_deriv


def get_curvature(x_vals, y_vals, is_log_x=True):
    """Calculates geometric curvature after normalizing axes to [0,1]."""
    # Use log values for x if the plot is log-scaled to find the visual 'knee'
    x = np.log10(x_vals) if is_log_x else x_vals
    y = y_vals

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    dx, dy = np.gradient(x_norm), np.gradient(y_norm)
    ddx, ddy = np.gradient(dx), np.gradient(dy)

    return np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5


def main():
    vehicle_params = {
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        'm': 310.0, 'm_uF': 12.0,
        'k_f': 240, 'k_r': 176,
    }

    # Step 2: Initial Calculations
    total_unsprung = vehicle_params['m_uF'] * 2
    total_sprung = vehicle_params['m'] - total_unsprung
    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung * vehicle_params['a_s'] / vehicle_params['wheelbase']
    k_total = vehicle_params['k_f'] + vehicle_params['k_r']

    # Step 3: Grid Setup
    lambda_vals = np.linspace(0.01, 1, 200)
    mu_vals = np.logspace(-1, 1, 200)
    L, M = np.meshgrid(lambda_vals, mu_vals)

    # Step 4: Sensitivity Error Calculation
    chi_flexible = calculate_LLTD(L, M, vehicle_params)
    sensitivity_flexible = np.gradient(chi_flexible, lambda_vals, axis=1)
    sensitivity_rigid = calculate_rigid_sensitivity(vehicle_params)
    sensitivity_error = np.abs(sensitivity_flexible - sensitivity_rigid) / np.abs(sensitivity_rigid)

    # Step 5: Knee Point Detection across Operating Range [0.4, 0.6]
    target_range = [0.4, 0.6]
    op_lambdas, knee_mus, error_at_knee = [], [], []

    for i, l_val in enumerate(lambda_vals):
        if target_range[0] <= l_val <= target_range[1]:
            err_slice = sensitivity_error[:, i]
            curv = get_curvature(mu_vals, err_slice, is_log_x=True)
            idx = np.argmax(curv)

            op_lambdas.append(l_val)
            knee_mus.append(mu_vals[idx])
            error_at_knee.append(err_slice[idx] * 100)  # Convert to %

    # --- Step 6: Generate the Three Panel Plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Original Heatmap + Knee Locus
    max_err = np.max(sensitivity_error)
    cp = ax1.contourf(mu_vals, lambda_vals, sensitivity_error.T, levels=np.linspace(0, max_err, 50), cmap='jet')
    ax1.contour(mu_vals, lambda_vals, sensitivity_error.T, levels=np.linspace(0, max_err, 25), colors='black',
                linewidths=0.5)

    # Overlay the knee locus
    ax1.scatter(knee_mus, op_lambdas, color='white', s=5, label='Knee Locus')

    ax1.set_xscale('log')
    ax1.set_xlabel('μ', fontsize=12)
    ax1.set_ylabel('λ', fontsize=12)
    ax1.set_title('Panel 1: Sensitivity Error Heatmap', fontsize=14)
    ax1.set_xticks([0.1, 1, 10])
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.colorbar(cp, ax=ax1, orientation='horizontal', pad=0.15)

    # Panel 2: Knee Stiffness (Nm/deg) vs Lambda
    knee_kcs = np.array(knee_mus) * k_total
    ax2.plot(op_lambdas, knee_kcs, color='blue', lw=2, marker='o', markersize=4)
    ax2.set_title('Panel 2: Required Stiffness at Knee', fontsize=14)
    ax2.set_xlabel('λ (Suspension Setup)')
    ax2.set_ylabel('k_c (Nm/deg)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Error at Knee (%) vs Lambda
    ax3.plot(op_lambdas, error_at_knee, color='red', lw=2, marker='o', markersize=4)
    ax3.set_title('Panel 3: Sensitivity Error at Knee', fontsize=14)
    ax3.set_xlabel('λ (Suspension Setup)')
    ax3.set_ylabel('Error (%)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()