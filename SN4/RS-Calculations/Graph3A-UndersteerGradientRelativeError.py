import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def calculate_understeer_gradient(L, M, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis.
    Uses the exact physics model and unit handling from your reference script.
    """
    g = 9.81  # m/s^2

    # --- Part 1: Standard Understeer from Tire/Mass properties ---
    tire_mass_component = (params['m'] / params['l']) * (
            (params['b_cg_rear'] / params['C_F_deg']) - (params['a_cg_front'] / params['C_R_deg'])
    )

    # --- Part 2: Roll Steer Component for a Flexible Chassis ---
    denominator = M + L - L ** 2  # Note: M is mu, L is lambda
    denominator[denominator == 0] = 1e-9  # prevent divide by zero

    phi_y = params['phi_y']  # baseline roll gradient (deg per m/s^2)

    phi_f_component = (M + (1 - L) * params['zeta']) / denominator
    phi_r_component = (M + L * (1 - params['zeta'])) / denominator

    roll_steer_component = (phi_f_component * params['eps_F'] - phi_r_component * params['eps_R']) * phi_y

    # Multiply the intermediate result by g to get the final units of deg/g
    ku_deg_per_g = (tire_mass_component - roll_steer_component) * g
    return ku_deg_per_g


def calculate_rigid_understeer(params):
    """
    Calculates the rigid understeer gradient (ku0) based on the reference script's physics.
    This is the limit of the flexible equation as mu -> infinity.
    """
    g = 9.81  # m/s^2

    # Part 1: Tire/Mass component (same as flexible case)
    tire_mass_component = (params['m'] / params['l']) * (
            (params['b_cg_rear'] / params['C_F_deg']) - (params['a_cg_front'] / params['C_R_deg'])
    )

    # Part 2: Rigid Roll steer component
    phi_y = params['phi_y']
    # As mu -> inf, the phi_f and phi_r components both become 1
    roll_steer_component_rigid = (params['eps_F'] - params['eps_R']) * phi_y

    k_u0_deg_per_g = (tire_mass_component - roll_steer_component_rigid) * g
    return k_u0_deg_per_g


def main():
    """Main function to generate the final relative error contour plot."""

    # --- Vehicle parameters (from your reference script) ---
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

    # --- Pre-calculations (from your reference script) ---
    vehicle_params['a_cg_front'] = vehicle_params['l'] * (1 - vehicle_params['front_weight_dist'])
    vehicle_params['b_cg_rear'] = vehicle_params['l'] * vehicle_params['front_weight_dist']

    # Sprung masses
    m_sF = (vehicle_params['m'] * vehicle_params['b_cg_rear'] / vehicle_params['l']) - vehicle_params['m_uF']
    m_sR = (vehicle_params['m'] * vehicle_params['a_cg_front'] / vehicle_params['l']) - vehicle_params['m_uR']

    vehicle_params['zeta'] = (m_sF * vehicle_params['d_sF']) / \
                             (m_sF * vehicle_params['d_sF'] + m_sR * vehicle_params['d_sR'])

    # Keep cornering stiffness in deg
    vehicle_params['C_F_deg'] = vehicle_params['C_F']
    vehicle_params['C_R_deg'] = vehicle_params['C_R']

    # Roll stiffness in deg
    total_roll_stiffness_deg = vehicle_params['k_F_springs'] + vehicle_params['k_R_springs']
    roll_moment_total = m_sF * vehicle_params['d_sF'] + m_sR * vehicle_params['d_sR']
    vehicle_params['phi_y'] = roll_moment_total / total_roll_stiffness_deg  # deg per m/s^2

    # --- Grid setup ---
    mu_vals = np.logspace(-1, 1, 200)
    lambda_vals = np.linspace(0.2, 0.8, 200)
    M, L = np.meshgrid(mu_vals, lambda_vals)

    # --- Calculation ---
    ku_flexible = calculate_understeer_gradient(L, M, vehicle_params)
    ku_rigid = calculate_rigid_understeer(vehicle_params)

    if abs(ku_rigid) < 1e-9:
        ku_rigid = 1e-9

    relative_error = np.abs((ku_flexible - ku_rigid) / ku_rigid)

    # --- Plotting ---
    min_error = np.min(relative_error)
    max_error = np.max(relative_error)
    levels = np.linspace(min_error, max_error, 51)

    fig, ax = plt.subplots(figsize=(8, 8))
    contour_filled = ax.contourf(M, L, relative_error, levels=levels, cmap='jet')
    ax.contour(M, L, relative_error, levels=25, colors='black', linewidths=0.8)

    # --- Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=16)
    ax.set_ylabel('λ', fontsize=16, rotation=0, labelpad=20)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.2, 0.8)
    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_box_aspect(1)

    # --- Color Bar ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.3)
    cbar_ticks = np.linspace(min_error, max_error, 4)
    cbar = fig.colorbar(contour_filled, cax=cax, orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xticklabels([f'{tick:.2f}' for tick in cbar_ticks])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('$|k_u - k_{u0}| / |k_{u0}|$', fontsize=16, labelpad=10)

    plt.show()


if __name__ == "__main__":
    main()