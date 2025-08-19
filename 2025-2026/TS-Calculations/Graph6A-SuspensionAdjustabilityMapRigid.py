import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon


def calculate_LLTD_rigid(l, params):
    """
    Calculates the lateral load transfer proportion (chi_0) for a perfectly RIGID chassis.
    This is the limit of the flexible equation as mu -> infinity.
    """
    # This is a linear relationship with respect to l (lambda)
    # The terms C1, C2, C3, C4 are constants based on vehicle geometry and mass
    C1 = (params['d_sF'] * params['m_sF']) / (params['h_G'] * params['m'])
    C2 = (params['d_sR'] * params['m_sR']) / (params['h_G'] * params['m'])
    C3 = (params['z_F'] * params['m_sF']) / (params['h_G'] * params['m'])
    C4 = (params['z_uF'] * params['m_uF']) / (params['h_G'] * params['m'])

    # The rigid LLTD is chi_0 = l * (C1 + C2) + C3 + C4
    return l * (C1 + C2) + C3 + C4


def main():
    """Main function to generate the Rigid Body Suspension Adjustability Map."""

    # --- Step 1: Define Base Vehicle Parameters ---
    # These are the same parameters used for our previous LLTD calculations
    vehicle_params = {
        # Geometric Parameters [m]
        'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279, 'z_F': 0.115,
        'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
        # Mass Parameters [kg]
        'm': 310.0, 'm_uF': 12.0, 'm_uR': 12.0,
    }

    # --- Step 2: Perform Initial Calculations ---
    total_unsprung_mass = vehicle_params['m_uF'] + vehicle_params['m_uR']
    total_sprung_mass = vehicle_params['m'] - total_unsprung_mass
    b_s = vehicle_params['wheelbase'] - vehicle_params['a_s']
    vehicle_params['m_sF'] = total_sprung_mass * b_s / vehicle_params['wheelbase']
    vehicle_params['m_sR'] = total_sprung_mass * vehicle_params['a_s'] / vehicle_params['wheelbase']

    # --- Step 3: Define Your Team's Suspension Stiffness Parameters (SN4) ---
    K_spring_F = 240.85  # Nm/deg
    K_spring_R = 176.05  # Nm/deg
    K_ARB_F_soft = 61.64  # Nm/deg
    K_ARB_F_stiff = 305.47  # Nm/deg
    K_ARB_R_soft = 109.22  # Nm/deg
    K_ARB_R_stiff = 320.03  # Nm/deg

    # --- Step 4: Calculate the Coordinates of the Setup Window ---
    # This section is identical to the previous script
    k_total_no_arb = K_spring_F + K_spring_R
    rsd_no_arb = K_spring_F / k_total_no_arb
    k_total_rear_off_soft = K_spring_F + K_ARB_F_soft + K_spring_R
    rsd_rear_off_soft = (K_spring_F + K_ARB_F_soft) / k_total_rear_off_soft
    k_total_rear_off_stiff = K_spring_F + K_ARB_F_stiff + K_spring_R
    rsd_rear_off_stiff = (K_spring_F + K_ARB_F_stiff) / k_total_rear_off_stiff
    k_total_front_off_soft = K_spring_F + K_spring_R + K_ARB_R_soft
    rsd_front_off_soft = K_spring_F / k_total_front_off_soft
    k_total_front_off_stiff = K_spring_F + K_spring_R + K_ARB_R_stiff
    rsd_front_off_stiff = K_spring_F / k_total_front_off_stiff
    k_total_A = K_spring_F + K_ARB_F_soft + K_spring_R + K_ARB_R_soft
    rsd_A = (K_spring_F + K_ARB_F_soft) / k_total_A
    k_total_B = K_spring_F + K_ARB_F_stiff + K_spring_R + K_ARB_R_soft
    rsd_B = (K_spring_F + K_ARB_F_stiff) / k_total_B
    k_total_C = K_spring_F + K_ARB_F_stiff + K_spring_R + K_ARB_R_stiff
    rsd_C = (K_spring_F + K_ARB_F_stiff) / k_total_C
    k_total_D = K_spring_F + K_ARB_F_soft + K_spring_R + K_ARB_R_stiff
    rsd_D = (K_spring_F + K_ARB_F_soft) / k_total_D
    box_coords = np.array([[rsd_A, k_total_A], [rsd_B, k_total_B], [rsd_C, k_total_C], [rsd_D, k_total_D]])

    # --- Step 5: Generate the Background Contour Map (Rigid Body) ---
    rsd_grid_vals = np.linspace(0.3, 0.7, 100)
    k_total_grid_vals = np.linspace(400, 1000, 100)
    RSD, K_TOTAL = np.meshgrid(rsd_grid_vals, k_total_grid_vals)

    # Calculate the rigid LLTD. It only depends on RSD (lambda), not total stiffness.
    LLTD_grid_rigid = calculate_LLTD_rigid(RSD, vehicle_params)

    # --- Step 6: Create the Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the background contour
    contour = ax.contourf(rsd_grid_vals, k_total_grid_vals, LLTD_grid_rigid, levels=20, cmap='viridis')
    cbar = fig.colorbar(contour)
    cbar.set_label('LLTD (Front %) - Rigid Chassis', fontsize=12)

    # Plot the setup window (same as before)
    poly = Polygon(box_coords, closed=True, facecolor='red', edgecolor='white', alpha=0.4, linewidth=2,
                   label='Usable Adjustability Range')
    ax.add_patch(poly)
    ax.plot([rsd_rear_off_soft, rsd_rear_off_stiff], [k_total_rear_off_soft, k_total_rear_off_stiff],
            'm-o', linewidth=3, label='Front ARB Only')
    ax.plot([rsd_front_off_soft, rsd_front_off_stiff], [k_total_front_off_soft, k_total_front_off_stiff],
            'c-o', linewidth=3, label='Rear ARB Only')
    ax.plot(rsd_no_arb, k_total_no_arb, 'w*', markersize=15, label='Springs Only (No ARBs)')

    # --- Step 7: Styling ---
    ax.set_title('SN4 Suspension Adjustability Map (Rigid Body Model)', fontsize=18)
    ax.set_xlabel('λ, Roll Stiffness Distribution (Front %)', fontsize=14)
    ax.set_ylabel('Total Roll Stiffness (Nm/deg)', fontsize=14)
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(400, 1000)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
