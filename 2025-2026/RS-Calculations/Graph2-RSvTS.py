import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_understeer_gradient(lambda_vals, mu_vals, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis.
    This function is taken directly from your previous code.
    """
    m = params['m']
    a = params['a']
    b = params['b']
    l = params['l']
    C_F = params['C_F']
    C_R = params['C_R']
    eps_F = params['eps_F']
    eps_R = params['eps_R']
    zeta = params['zeta']
    phi_y = params['phi_y']

    # Create a 2D grid for vectorized calculation
    # L corresponds to lambda (columns), M corresponds to mu (rows)
    L, M = np.meshgrid(lambda_vals, mu_vals)

    # First term (rigid chassis contribution)
    term1 = (m / l) * (b / C_F - a / C_R)

    # Flexible chassis correction terms
    denominator = L + M - L ** 2
    denominator[denominator == 0] = 1e-9  # avoid divide by zero

    term2 = ((M + (1 - L) * zeta) / denominator) * eps_F
    term3 = ((M + L * (1 - zeta)) / denominator) * eps_R

    # ku is returned in radians/g assuming inputs are in SI units/radians
    ku = term1 - (term2 - term3) * phi_y

    return ku


def main():
    """Main function to generate the line plot."""

    # --- Vehicle Parameters (from your previous code) ---
    vehicle_params = {
        # Geometry
        'l': 2.5,  # wheelbase [m]
        'a': 1.25,  # CG to front axle [m]
        'b': 1.25,  # CG to rear axle [m]
        # Mass
        'm': 800.0,  # total mass [kg] - adjusted to match graph magnitude
        # Tire cornering stiffness [N/rad]
        'C_F': 80000.0,
        'C_R': 80000.0,
        # Compliance parameters
        'eps_F': -0.05,  # roll steer front
        'eps_R': 0.15,  # roll steer rear
        'zeta': 0.05,  # roll axis inclination param - adjusted
        'phi_y': 0.08  # roll angle per g lat acc [rad/g] - adjusted
    }

    # --- Setup for the plot ---
    # Define the mu range for the x-axis
    mu_vals = np.logspace(-1, 1, 200)  # 0.1 to 10

    # Define the constant lambda values for each line series
    lambda_series = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # --- Calculation ---
    # Calculate the entire grid of ku values once.
    # The output `ku_grid_rad` will have shape (len(mu_vals), len(lambda_series))
    ku_grid_rad = calculate_understeer_gradient(lambda_series, mu_vals, vehicle_params)

    # Convert the entire grid from rad/g to deg/g for plotting
    ku_grid_deg = ku_grid_rad * 180 / np.pi

    # --- Plotting ---
    # CHANGED: figsize is now square
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define a color cycle to distinguish lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_series)))

    # Loop through each lambda series (each column of the grid) to plot it
    for i, lam in enumerate(lambda_series):
        # Get the column of data corresponding to this lambda value
        ku_values_for_series = ku_grid_deg[:, i]
        ax.plot(mu_vals, ku_values_for_series, label=f'λ = {lam:.1f}', linewidth=2.5)

    # --- Styling to replicate the reference image ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=16)
    ax.set_ylabel('$k_u$ [deg/g]', fontsize=16)
    ax.set_title('Understeer Gradient vs Normalised Chassis Torsional Stiffness', fontsize=16)

    # Set axis limits and ticks
    ax.set_xlim(0.1, 10)
    # CHANGED: y-axis limit is now 2.0
    ax.set_ylim(0.6, 2.0)
    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # ADDED: Enforce a square plot box
    ax.set_box_aspect(1)

    # Grid and Legend
    ax.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax.legend(ncol=2, fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()