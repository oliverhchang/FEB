import numpy as np
import matplotlib.pyplot as plt

def calculate_understeer_gradient(lambda_vals, mu, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis.
    Returns ku in deg/g (using degrees throughout).
    """
    l = np.copy(lambda_vals)
    g = 9.81  # m/s^2

    # --- Part 1: Standard Understeer from Tire/Mass properties ---
    tire_mass_component = (params['m'] / params['l']) * (
        (params['b_cg_rear'] / params['C_F_deg']) - (params['a_cg_front'] / params['C_R_deg'])
    )

    # --- Part 2: Roll Steer Component for a Flexible Chassis ---
    denominator = mu + l - l ** 2
    denominator[denominator == 0] = 1e-9  # prevent divide by zero

    phi_y = params['phi_y']  # baseline roll gradient from springs (deg per m/s^2)

    phi_f_component = (mu + (1 - l) * params['zeta']) / denominator
    phi_r_component = (mu + l * (1 - params['zeta'])) / denominator

    roll_steer_component = (phi_f_component * params['eps_F'] - phi_r_component * params['eps_R']) * phi_y

    ku_deg_per_g = (tire_mass_component - roll_steer_component) * g  # Already in deg, multiply by g for deg/g
    return ku_deg_per_g


def main():
    """Generate Understeer Gradient vs. Roll Stiffness Distribution plot (deg units)."""

    # --- Vehicle parameters ---
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
        'C_R': 250.0,  # Tire cornering stiffness in N·m/deg
    }

    # --- Pre-calculations ---
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

    # --- Simulation ---
    lambda_range = np.linspace(0.2, 0.8, 500)
    mu_values_to_plot = [0.1, 0.2, 0.5, 1, 2, 5,10]  # μ = chassis flex ratio

    fig, ax = plt.subplots(figsize=(8, 8))
    all_ku_values = []

    for mu in mu_values_to_plot:
        ku_values = calculate_understeer_gradient(lambda_range, mu, vehicle_params)
        all_ku_values.append(ku_values)
        ax.plot(lambda_range, ku_values, label=f'μ = {mu}', linewidth=2)

    # Plot limits
    all_ku_values = np.concatenate(all_ku_values)
    y_min, y_max = np.min(all_ku_values), np.max(all_ku_values)
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Formatting
    ax.set_title("Understeer Gradient (k_u) vs Roll Stiffness Distribution", fontsize=16)
    ax.set_xlabel("λ (Roll Stiffness Distribution)", fontsize=12)
    ax.set_ylabel("k_u [deg/g]", fontsize=12)
    ax.legend(title="Chassis Stiffness μ")
    ax.grid(True, linestyle=':')
    ax.set_xlim(0.2, 0.8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
