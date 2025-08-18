import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def calculate_understeer_gradient(lambda_vals, mu_vals, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis using Equation 3.16.
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

    # Broadcast meshgrid
    L, M = np.meshgrid(lambda_vals, mu_vals)

    # First term (rigid chassis contribution)
    term1 = (m / l) * (b / C_F - a / C_R)

    # Flexible chassis correction terms
    denominator = L + M - L**2
    denominator[denominator == 0] = 1e-9  # avoid divide by zero

    term2 = ((M + (1 - L) * zeta) / denominator) * eps_F
    term3 = ((M + L * (1 - zeta)) / denominator) * eps_R

    ku = term1 - (term2 - term3) * phi_y

    return ku

def main():
    """Main function to generate the Understeer Gradient contour plot."""

    # --- Vehicle Parameters ---
    vehicle_params = {
        # Geometry
        'l': 2.5,   # wheelbase [m]
        'a': 1.25,  # CG to front axle [m]
        'b': 1.25,  # CG to rear axle [m]

        # Mass
        'm': 310.0,  # total mass [kg]

        # Tire cornering stiffness [N/rad]
        'C_F': 250.0,
        'C_R': 250.0,

        # Compliance parameters
        'eps_F': -0.05,
        'eps_R': 0.15,
        'zeta': 0.5,      # front/rear distribution factor (example value)
        'phi_y': 1.0      # lateral compliance scaling factor (example value)
    }

    # --- Grid setup ---
    lambda_vals = np.linspace(0.01, 1, 200)  # λ range
    mu_vals = np.logspace(-1, 1, 200)        # μ range (0.1 to 10)

    ku = calculate_understeer_gradient(lambda_vals, mu_vals, vehicle_params)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    contour_filled = ax.contourf(mu_vals, lambda_vals, ku,
                                 levels=50, cmap='jet')
    ax.contour(mu_vals, lambda_vals, ku, levels=25,
               colors='black', linewidths=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('λ', fontsize=14, rotation=0, labelpad=15)
    ax.set_title('Understeer Gradient $k_u$', fontsize=16)
    ax.set_box_aspect(1)

    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(contour_filled, ax=ax, orientation='horizontal',
                        pad=0.15, aspect=40)
    cbar.set_label('$k_u$', fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
