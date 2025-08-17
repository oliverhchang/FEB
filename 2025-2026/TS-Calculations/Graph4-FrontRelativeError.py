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
    This is a constant value, representing a linear relationship.
    """
    term1_deriv = (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    term2_deriv = (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])

    return term1_deriv - term2_deriv


def main():
    """Main function to generate the LLTD Sensitivity contour plot."""

    # --- Step 1: Define All Vehicle Parameters in a Single Dictionary ---
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0,
    }

    # --- Step 2: Perform Initial Calculations Based on Parameters ---
    total_unsprung_mass = vehicle_params['m_uF'] * 2
    total_sprung_mass = vehicle_params['m'] - total_unsprung_mass
    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung_mass * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung_mass * vehicle_params['a_s'] / vehicle_params['wheelbase']

    # --- Step 3: Set up the Grid for the Contour Plot ---
    lambda_vals = np.linspace(0.01, 1, 200)
    mu_vals = np.logspace(-1, 1, 200)  # 0.1 to 10
    L, M = np.meshgrid(lambda_vals, mu_vals)

    # --- Step 4: Calculate the Sensitivity Error (Non-Linearity Error) ---
    # Calculate chi for the flexible case
    chi_flexible = calculate_LLTD(L, M, vehicle_params)

    # Calculate the sensitivity (d(chi)/d(lambda)) for the flexible case numerically
    sensitivity_flexible = np.gradient(chi_flexible, lambda_vals, axis=1)

    # Calculate the constant sensitivity for the rigid case analytically
    sensitivity_rigid = calculate_rigid_sensitivity(vehicle_params)

    # Avoid division by zero in the sensitivity calculation
    if sensitivity_rigid == 0:
        sensitivity_rigid = 1e-9

    # Calculate the relative error of the sensitivity
    sensitivity_error = np.abs(sensitivity_flexible - sensitivity_rigid) / np.abs(sensitivity_rigid)

    # Dynamically set the color limit based on the maximum sensitivity in the data
    max_error_val = np.max(sensitivity_error)

    # --- Step 5: Generate the Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    contour_filled = ax.contourf(M, L, sensitivity_error, levels=np.linspace(0, max_error_val, 50), cmap='jet')
    ax.contour(M, L, sensitivity_error, levels=np.linspace(0, max_error_val, 25), colors='black', linewidths=0.5)

    # --- Step 6: Styling ---
    ax.set_xscale('log')
    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('λ', fontsize=14, rotation=0, labelpad=15)
    ax.set_title('LLTD Sensitivity Error (Non-Linearity)', fontsize=16)
    ax.set_box_aspect(1)

    ax.set_xticks([0.1, 1, 10])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(contour_filled, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Sensitivity Error', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
