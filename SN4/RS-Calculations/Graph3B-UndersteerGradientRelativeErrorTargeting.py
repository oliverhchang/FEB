import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def calculate_understeer_gradient(L, M, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis.
    Uses the final physics model and unit handling.
    """
    g = 9.81  # m/s^2

    # Part 1: Standard Understeer from Tire/Mass properties
    tire_mass_component = (params['m'] / params['l']) * (
            (params['b_cg_rear'] / params['C_F_deg']) - (params['a_cg_front'] / params['C_R_deg'])
    )

    # Part 2: Roll Steer Component for a Flexible Chassis
    denominator = M + L - L ** 2  # M is mu, L is lambda
    denominator[denominator == 0] = 1e-9

    phi_y = params['phi_y']  # deg per m/s^2

    phi_f_component = (M + (1 - L) * params['zeta']) / denominator
    phi_r_component = (M + L * (1 - params['zeta'])) / denominator

    roll_steer_component = (phi_f_component * params['eps_F'] - phi_r_component * params['eps_R']) * phi_y

    # Multiply the intermediate result by g to get the final units of deg/g
    ku_deg_per_g = (tire_mass_component - roll_steer_component) * g
    return ku_deg_per_g


def calculate_rigid_understeer(params):
    """
    Calculates the rigid understeer gradient (ku0) based on the final physics model.
    """
    g = 9.81  # m/s^2

    # Part 1: Tire/Mass component
    tire_mass_component = (params['m'] / params['l']) * (
            (params['b_cg_rear'] / params['C_F_deg']) - (params['a_cg_front'] / params['C_R_deg'])
    )

    # Part 2: Rigid Roll steer component
    phi_y = params['phi_y']
    roll_steer_component_rigid = (params['eps_F'] - params['eps_R']) * phi_y

    k_u0_deg_per_g = (tire_mass_component - roll_steer_component_rigid) * g
    return k_u0_deg_per_g


def main():
    """Main function to generate the relative error contour plot with target analysis."""

    # --- Design Goal Parameters ---
    # NOTE: The threshold was lowered because the max error for these vehicle
    # parameters is less than 15%. This ensures the annotation appears.
    relative_error_threshold = 0.01  # Example: 5% maximum acceptable error
    target_lambda_range = [0.3, 0.7]  # e.g., from 30% to 70% front stiffness

    # --- Vehicle Parameters ---
    vehicle_params = {
        'm': 310.0, 'm_uF': 12.0,
        'm_uR': 12.0,
        'l': 1.592,
        'front_weight_dist': 0.535,
        'd_sF': 0.273 - 0.115,
        'd_sR': 0.273 - 0.165,
        'k_F_springs': 240.0,
        'k_R_springs': 176.0,  # N·m/deg
        'eps_F': 0.07,
        'eps_R': 0.12,
        'C_F': 250.0,
        'C_R': 250.0,  # Tire cornering stiffness in N/deg
    }

    # --- Pre-calculations ---
    vehicle_params['a_cg_front'] = vehicle_params['l'] * (1 - vehicle_params['front_weight_dist'])
    vehicle_params['b_cg_rear'] = vehicle_params['l'] * vehicle_params['front_weight_dist']

    # Sprung masses
    m_sF = (vehicle_params['m'] * vehicle_params['b_cg_rear'] / vehicle_params['l']) - vehicle_params['m_uF']
    m_sR = (vehicle_params['m'] * vehicle_params['a_cg_front'] / vehicle_params['l']) - vehicle_params['m_uR']

    vehicle_params['zeta'] = (m_sF * vehicle_params['d_sF']) / \
                             (m_sF * vehicle_params['d_sF'] + m_sR * vehicle_params['d_sR'])

    vehicle_params['C_F_deg'] = vehicle_params['C_F']
    vehicle_params['C_R_deg'] = vehicle_params['C_R']

    total_roll_stiffness_deg = vehicle_params['k_F_springs'] + vehicle_params['k_R_springs']
    roll_moment_total = m_sF * vehicle_params['d_sF'] + m_sR * vehicle_params['d_sR']
    vehicle_params['phi_y'] = roll_moment_total / total_roll_stiffness_deg  # deg per m/s^2

    # --- Grid setup ---
    mu_vals = np.logspace(-1, 1.5, 200)  # Extended mu range for wider analysis
    lambda_vals = np.linspace(0.2, 0.8, 200)
    M, L = np.meshgrid(mu_vals, lambda_vals)

    # --- Calculation ---
    ku_flexible = calculate_understeer_gradient(L, M, vehicle_params)
    ku_rigid = calculate_rigid_understeer(vehicle_params)

    if abs(ku_rigid) < 1e-9:
        ku_rigid = 1e-9

    relative_error = np.abs((ku_flexible - ku_rigid) / ku_rigid)
    max_error = np.max(relative_error)
    levels = np.linspace(0, max_error, 51)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 8))
    contour_filled = ax.contourf(M, L, relative_error, levels=levels, cmap='jet')
    ax.contour(M, L, relative_error, levels=25, colors='black', linewidths=0.5)

    # --- Highlight Target Operating Range and Find Required Stiffness ---
    # Draw the specific error contour line for the threshold
    ax.contour(M, L, relative_error, levels=[relative_error_threshold],
               colors='white', linewidths=3, linestyles='solid')

    # Highlight the target lambda range with a transparent overlay
    ax.axhspan(target_lambda_range[0], target_lambda_range[1], color='white', alpha=0.15, zorder=1)

    # Find the worst-case mu required for the target lambda range
    worst_mu = 0.0
    worst_lambda = None
    for i in range(len(lambda_vals)):
        lambda_val = lambda_vals[i]
        if target_lambda_range[0] <= lambda_val <= target_lambda_range[1]:
            error_column = relative_error[i, :]
            above_indices = np.where(error_column >= relative_error_threshold)[0]
            if above_indices.size > 0:
                last_above_idx = above_indices[-1]
                if last_above_idx + 1 < len(mu_vals):
                    mu1, mu2 = mu_vals[last_above_idx], mu_vals[last_above_idx + 1]
                    err1, err2 = error_column[last_above_idx], error_column[last_above_idx + 1]
                    log_mu_interp = np.interp(relative_error_threshold, [err2, err1], [np.log(mu2), np.log(mu1)])
                    mu_interp = np.exp(log_mu_interp)
                    if mu_interp > worst_mu:
                        worst_mu = mu_interp
                        worst_lambda = lambda_val

    # Add annotation for the result
    if worst_mu > 0:
        kc_required_deg = worst_mu * total_roll_stiffness_deg

        # Draw the vertical dotted line
        ax.axvline(x=worst_mu, ymin=0, ymax=(worst_lambda - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   color='white', linestyle='--', lw=2)

        # Optional: Draw a horizontal line as well
        ax.axhline(y=worst_lambda, xmin=0, xmax=(np.log(worst_mu) - np.log(ax.get_xlim()[0])) / (
                    np.log(ax.get_xlim()[1]) - np.log(ax.get_xlim()[0])), color='white', linestyle='--', lw=2)

        # --- INTELLIGENT ANNOTATION PLACEMENT ---
        # Decide where to place the text based on the point's position
        # Get plot center for positioning logic
        x_center = np.mean(ax.get_xlim())
        y_center = np.mean(ax.get_ylim())

        # Set horizontal alignment and position
        if worst_mu > x_center:
            ha = 'right'
            x_text = worst_mu - 0.05
        else:
            ha = 'left'
            x_text = worst_mu + 0.05

        # Set vertical alignment and position
        if worst_lambda > y_center:
            va = 'top'
            y_text = worst_lambda - 0.02
        else:
            va = 'bottom'
            y_text = worst_lambda + 0.02

        # Create and place the annotation
        annotation_text = (
            f'Required μ ≥ {worst_mu:.2f}\n'
            f'($k_c$ ≥ {kc_required_deg:.0f} N·m/deg)\n'
            f'to keep error ≤ {relative_error_threshold * 100:.0f}%'
        )
        ax.annotate(
            annotation_text, xy=(worst_mu, worst_lambda), xytext=(x_text, y_text),
            color='white', ha=ha, va=va, fontsize=12,
            arrowprops=dict(arrowstyle='->', color='white', connectionstyle="arc3,rad=-0.2"))

    # --- Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ (Chassis Stiffness Ratio)', fontsize=14)
    ax.set_ylabel('λ (Roll Stiffness Distribution)', fontsize=14, rotation=90, labelpad=15)
    ax.set_box_aspect(1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.3)
    cbar_ticks = np.linspace(0, max_error, 4)
    cbar = fig.colorbar(contour_filled, cax=cax, orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xticklabels([f'{tick:.2f}' for tick in cbar_ticks])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('$|k_u - k_{u0}| / |k_{u0}|$', fontsize=16, labelpad=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
