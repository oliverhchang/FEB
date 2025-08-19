import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_understeer_gradient(mu_vals, lambda_val, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis.
    Accepts a range of mu values and a single lambda value.
    Returns ku in deg/g (using degrees throughout), based on the provided formula.
    """
    # mu is now the array, l is a scalar
    mu = np.copy(mu_vals)
    l = lambda_val
    g = 9.81  # m/s^2

    # --- Part 1: Standard Understeer from Tire/Mass properties ---
    # This part is a single scalar value
    tire_mass_component = (params['m'] / params['l']) * (
        (params['b_cg_rear'] / params['C_F_deg']) - (params['a_cg_front'] / params['C_R_deg'])
    )

    # --- Part 2: Roll Steer Component for a Flexible Chassis ---
    # NumPy will correctly broadcast the scalar 'l' with the array 'mu'
    denominator = mu + l - l ** 2
    denominator[denominator == 0] = 1e-9  # prevent divide by zero

    phi_y = params['phi_y']  # baseline roll gradient (deg per m/s^2)

    phi_f_component = (mu + (1 - l) * params['zeta']) / denominator
    phi_r_component = (mu + l * (1 - params['zeta'])) / denominator

    roll_steer_component = (phi_f_component * params['eps_F'] - phi_r_component * params['eps_R']) * phi_y

    # Multiply the intermediate result by g to get the final units of deg/g
    ku_deg_per_g = (tire_mass_component - roll_steer_component) * g
    return ku_deg_per_g


def main():
    """Generate Understeer Gradient vs Normalised Chassis Torsional Stiffness plot."""

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

    # Keep cornering stiffness in deg
    vehicle_params['C_F_deg'] = vehicle_params['C_F']
    vehicle_params['C_R_deg'] = vehicle_params['C_R']

    # Roll stiffness in deg
    total_roll_stiffness_deg = vehicle_params['k_F_springs'] + vehicle_params['k_R_springs']
    roll_moment_total = m_sF * vehicle_params['d_sF'] + m_sR * vehicle_params['d_sR']
    vehicle_params['phi_y'] = roll_moment_total / total_roll_stiffness_deg  # deg per m/s^2

    # --- Setup for the plot ---
    # Define the mu range for the x-axis
    mu_vals = np.logspace(-1, 1, 200)  # 0.1 to 10

    # Define the constant lambda values for each line series
    lambda_series = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Loop through each CONSTANT lambda value to plot a line
    for lam in lambda_series:
        # Calculate the ku values for this specific lambda across all mu values
        ku_values_for_series = calculate_understeer_gradient(mu_vals, lam, vehicle_params)

        # The function already returns the result in deg/g, so no conversion is needed
        ax.plot(mu_vals, ku_values_for_series, label=f'λ = {lam:.1f}', linewidth=2.5)

    # --- Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=16)
    ax.set_ylabel('$k_u$ [deg/g]', fontsize=16)
    ax.set_title('Understeer Gradient vs Normalised Chassis Torsional Stiffness', fontsize=16)

    # Set axis limits and ticks
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Enforce a square plot box
    ax.set_box_aspect(1)

    # Grid and Legend
    ax.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax.legend(ncol=2, fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()