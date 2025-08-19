import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def calculate_understeer_gradient(L, M, params):
    """
    Calculates the understeer gradient (ku) for a flexible chassis.
    Accepts 2D meshgrid arrays for L (lambda) and M (mu).
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

    term1 = (m / l) * (b / C_F - a / C_R)
    denominator = L + M - L ** 2
    denominator[denominator == 0] = 1e-9

    term2 = ((M + (1 - L) * zeta) / denominator) * eps_F
    term3 = ((M + L * (1 - zeta)) / denominator) * eps_R

    ku = term1 - (term2 - term3) * phi_y
    return ku


def calculate_rigid_understeer(params):
    """
    Calculates the understeer gradient (ku0) for a perfectly rigid chassis (mu -> inf).
    """
    m = params['m']
    a = params['a']
    b = params['b']
    l = params['l']
    C_F = params['C_F']
    C_R = params['C_R']
    eps_F = params['eps_F']
    eps_R = params['eps_R']
    phi_y = params['phi_y']

    term1 = (m / l) * (b / C_F - a / C_R)
    roll_comp_rigid = eps_F - eps_R

    ku_rigid = term1 - (roll_comp_rigid * phi_y)
    return ku_rigid


def main():
    """Main function to generate the final relative error contour plot."""

    # --- Vehicle Parameters ---
    vehicle_params = {
        'l': 2.5, 'a': 1.25, 'b': 1.25, 'm': 800.0,
        'C_F': 80000.0, 'C_R': 80000.0,
        'eps_F': -0.1, 'eps_R': 0.1,
        'zeta': 0.0, 'phi_y': 0.08
    }

    # --- Grid setup ---
    mu_vals = np.logspace(-1, 1, 200)  # X-axis: 0.1 to 10
    lambda_vals = np.linspace(0.2, 0.8, 200)  # Y-axis: 0.2 to 0.8
    M, L = np.meshgrid(mu_vals, lambda_vals)

    # --- Calculation ---
    ku_flexible = calculate_understeer_gradient(L, M, vehicle_params)
    ku_rigid = calculate_rigid_understeer(vehicle_params)

    if abs(ku_rigid) < 1e-9:
        ku_rigid = 1e-9

    relative_error = np.abs((ku_flexible - ku_rigid) / ku_rigid)

    # --- Determine min and max of relative error to fill colors ---
    min_error = np.min(relative_error)
    max_error = np.max(relative_error)
    levels = np.linspace(min_error, max_error, 51)

    # --- Plotting ---
    # CHANGED: figsize is now square
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

    # ADDED: Enforce a square plot box
    ax.set_box_aspect(1)

    # --- Color Bar on Top ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.3)  # Increased pad slightly

    # Use the calculated max_error for the ticks
    cbar_ticks = np.linspace(min_error, max_error, 4)
    cbar = fig.colorbar(contour_filled, cax=cax, orientation='horizontal', ticks=cbar_ticks)

    # Format tick labels to 2 decimal places
    cbar.ax.set_xticklabels([f'{tick:.2f}' for tick in cbar_ticks])

    # CHANGED: Set label position and padding to prevent overlap
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('$|k_u - k_{u0}| / |k_{u0}|$', fontsize=16, labelpad=10)

    plt.show()


if __name__ == "__main__":
    main()