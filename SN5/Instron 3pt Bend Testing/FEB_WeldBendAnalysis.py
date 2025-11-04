import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# User settings - update these if your CSV format is different
COL_FORCE = 'Force'
COL_DISP = 'Displacement'
FORCE_MULTIPLIER_TO_NEWTONS = 1000  # 1000 for kN, 1 for N

# Material properties for 4130 Steel with ER70S filler
THEORETICAL_YIELD_STRENGTH_MPA = 414.0
THEORETICAL_MODULUS_GPA = 200.0

# Specimen geometry
L_mm = 100.0  # support span
b_mm = 76.2  # width (3 inches)
h1_mm = 1.5875  # 1/16"
h2_mm = 3.175  # 1/8"

# Calculate theoretical yield forces
F_TH_1_16_kN = (2 * THEORETICAL_YIELD_STRENGTH_MPA * b_mm * h1_mm**2) / (3 * L_mm * 1000)
F_TH_1_8_kN = (2 * THEORETICAL_YIELD_STRENGTH_MPA * b_mm * h2_mm**2) / (3 * L_mm * 1000)

metadata = {
    1: ("Tommy 1/16", "1/16", h1_mm, h1_mm**2, F_TH_1_16_kN),
    2: ("Spencer 1/16", "1/16", h1_mm, h1_mm**2, F_TH_1_16_kN),
    3: ("Arnav 1/16", "1/16", h1_mm, h1_mm**2, F_TH_1_16_kN),
    4: ("Arnav 1/16", "1/16", h1_mm, h1_mm**2, F_TH_1_16_kN),
    5: ("Jessica 1/16", "1/16", h1_mm, h1_mm**2, F_TH_1_16_kN),
    6: ("Oliver /16", "1/16", h1_mm, h1_mm**2, F_TH_1_16_kN),
    7: ("Jessica 1/8", "1/8", h2_mm, h2_mm**2, F_TH_1_8_kN),
    8: ("Oliver 1/8", "1/8", h2_mm, h2_mm**2, F_TH_1_8_kN),
    9: ("Tommy 1/8", "1/8", h2_mm, h2_mm**2, F_TH_1_8_kN),
    10: ("Spencer 1/8", "1/8", h2_mm, h2_mm**2, F_TH_1_8_kN),
    11: ("Arnav 1/8", "1/8", h2_mm, h2_mm**2, F_TH_1_8_kN),
}


def calculate_stress_strain(df, L, b, h, h2):
    if COL_FORCE not in df.columns or COL_DISP not in df.columns:
        print(f"  ERROR: Can't find '{COL_FORCE}' or '{COL_DISP}' columns")
        print(f"  Available columns: {list(df.columns)}")
        raise KeyError("Missing required columns")

    # Calculate flexural stress: sigma = (3*F*L)/(2*b*h^2)
    stress_const = (3 * FORCE_MULTIPLIER_TO_NEWTONS * L) / (2 * b)
    df['Flexural Stress (MPa)'] = (stress_const * df[COL_FORCE]) / h2

    # Calculate flexural strain: epsilon = (6*delta*h)/L^2
    strain_const = (6 * h) / (L ** 2)
    df['Flexural Strain'] = strain_const * df[COL_DISP]

    return df


def get_flexural_modulus(df):
    try:
        max_stress = df['Flexural Stress (MPa)'].max()
        linear_region = df[
            (df['Flexural Stress (MPa)'] > 0.4 * max_stress) &
            (df['Flexural Stress (MPa)'] < 0.6 * max_stress)
        ]

        if len(linear_region) < 2:
            print("  WARNING: Not enough points for modulus calculation")
            return np.nan

        slope, _, _, _, _ = linregress(
            linear_region['Flexural Strain'],
            linear_region['Flexural Stress (MPa)']
        )

        return slope / 1000.0  # convert MPa to GPa
    except Exception as e:
        print(f"  WARNING: Modulus calculation failed: {e}")
        return np.nan


def get_energy_to_failure(df):
    force_in_kN = df[COL_FORCE] if FORCE_MULTIPLIER_TO_NEWTONS == 1000 else df[COL_FORCE] / 1000.0
    return np.trapezoid(force_in_kN, x=df[COL_DISP])


def main():
    print("Starting weld bend test analysis...")
    print(f"Theoretical yield force (1/16\"): {F_TH_1_16_kN:.3f} kN")
    print(f"Theoretical yield force (1/8\"):  {F_TH_1_8_kN:.3f} kN")

    data_folder = 'bend_test_data'
    all_results = []

    fig_fd, (ax_fd_1, ax_fd_2) = plt.subplots(2, 1, figsize=(12, 16))
    fig_ss, (ax_ss_1, ax_ss_2) = plt.subplots(2, 1, figsize=(12, 16))

    for spec_id, (note, thickness_str, h_val, h2_val, F_th_val) in metadata.items():
        filename = f"{spec_id}.csv"
        filepath = os.path.join(data_folder, filename)

        if not os.path.exists(filepath):
            print(f"WARNING: Skipping missing file: {filepath}")
            continue

        print(f"Processing: {filepath} ({note})")

        try:
            # Read CSV (skip first 3 rows, no header)
            df = pd.read_csv(filepath, header=None, skiprows=3, encoding='utf-8-sig')
            df = df.iloc[:, :3]
            df.columns = ['Time', 'Displacement', 'Force']

            df[COL_FORCE] = pd.to_numeric(df[COL_FORCE], errors='coerce')
            df[COL_DISP] = pd.to_numeric(df[COL_DISP], errors='coerce')
            df = df.dropna(subset=[COL_FORCE, COL_DISP])

            if df.empty:
                print(f"  WARNING: No valid data in {filepath}, skipping")
                continue

            print(f"  Loaded {len(df)} data points")

            df = calculate_stress_strain(df, L_mm, b_mm, h_val, h2_val)

            peak_force_kN = df[COL_FORCE].max()
            disp_at_peak_mm = df.loc[df[COL_FORCE].idxmax(), COL_DISP]
            peak_stress_MPa = df['Flexural Stress (MPa)'].max()
            strain_at_peak = df.loc[df['Flexural Stress (MPa)'].idxmax(), 'Flexural Strain']
            modulus_GPa = get_flexural_modulus(df)
            energy_J = get_energy_to_failure(df)
            force_efficiency = (peak_force_kN / F_th_val) * 100 if peak_force_kN > 0 else 0.0

            summary = {
                "Specimen ID": spec_id,
                "Note": note,
                "Thickness (in)": thickness_str,
                "Peak Force (kN)": peak_force_kN,
                "Theoretical Yield Force (kN)": F_th_val,
                "Force Efficiency (%)": force_efficiency,
                "Displacement at Peak Force (mm)": disp_at_peak_mm,
                "Flexural Strength (MPa)": peak_stress_MPa,
                "Strain at Peak Strength": strain_at_peak,
                "Flexural Modulus (GPa)": modulus_GPa,
                "Theoretical Modulus (GPa)": THEORETICAL_MODULUS_GPA,
                "Energy to Failure (J)": energy_J
            }
            all_results.append(summary)

            label = f"ID {spec_id}: {note}"
            if thickness_str == "1/16":
                ax_fd_1.plot(df[COL_DISP], df[COL_FORCE], label=label, alpha=0.8)
                ax_ss_1.plot(df['Flexural Strain'], df['Flexural Stress (MPa)'], label=label, alpha=0.8)
            else:
                ax_fd_2.plot(df[COL_DISP], df[COL_FORCE], label=label, alpha=0.8)
                ax_ss_2.plot(df['Flexural Strain'], df['Flexural Stress (MPa)'], label=label, alpha=0.8)

        except Exception as e:
            print(f"  ERROR: Failed to process {filepath}: {e}")

    print("\nSaving results...")

    if not all_results:
        print("WARNING: No data processed")

    column_order = [
        "Specimen ID", "Note", "Thickness (in)",
        "Peak Force (kN)", "Theoretical Yield Force (kN)", "Force Efficiency (%)",
        "Displacement at Peak Force (mm)",
        "Flexural Strength (MPa)", "Strain at Peak Strength",
        "Flexural Modulus (GPa)", "Theoretical Modulus (GPa)",
        "Energy to Failure (J)"
    ]

    summary_df = pd.DataFrame(all_results, columns=column_order)
    summary_df = summary_df.round(3)
    summary_df.to_csv('cumulative_bend_test_results.csv', index=False)
    print(f"Saved {len(summary_df)} results to cumulative_bend_test_results.csv")

    # Format force-displacement plots
    ax_fd_1.set_title('Force vs. Displacement (1/16" Specimens)', fontsize=16)
    ax_fd_1.set_xlabel('Displacement (mm)')
    ax_fd_1.set_ylabel('Force (kN)')
    ax_fd_1.axhline(y=F_TH_1_16_kN, color='r', linestyle=':', linewidth=2, label=f"Theoretical Yield ({F_TH_1_16_kN:.2f} kN)")
    if ax_fd_1.has_data():
        ax_fd_1.legend(loc='best')
    ax_fd_1.grid(True, linestyle='--')

    ax_fd_2.set_title('Force vs. Displacement (1/8" Specimens)', fontsize=16)
    ax_fd_2.set_xlabel('Displacement (mm)')
    ax_fd_2.set_ylabel('Force (kN)')
    ax_fd_2.axhline(y=F_TH_1_8_kN, color='r', linestyle=':', linewidth=2, label=f"Theoretical Yield ({F_TH_1_8_kN:.2f} kN)")
    if ax_fd_2.has_data():
        ax_fd_2.legend(loc='best')
    ax_fd_2.grid(True, linestyle='--')

    fig_fd.tight_layout()
    fig_fd.savefig('force_vs_displacement_plots.png')
    print("Saved force_vs_displacement_plots.png")

    # Format stress-strain plots
    ax_ss_1.set_title('Stress vs. Strain (1/16" Specimens)', fontsize=16)
    ax_ss_1.set_xlabel('Strain (mm/mm)')
    ax_ss_1.set_ylabel('Stress (MPa)')
    ax_ss_1.axhline(y=THEORETICAL_YIELD_STRENGTH_MPA, color='r', linestyle=':', linewidth=2,
                    label=f"Theoretical Yield ({THEORETICAL_YIELD_STRENGTH_MPA} MPa)")
    if ax_ss_1.has_data():
        ax_ss_1.legend(loc='best')
    ax_ss_1.grid(True, linestyle='--')

    ax_ss_2.set_title('Stress vs. Strain (1/8" Specimens)', fontsize=16)
    ax_ss_2.set_xlabel('Strain (mm/mm)')
    ax_ss_2.set_ylabel('Stress (MPa)')
    ax_ss_2.axhline(y=THEORETICAL_YIELD_STRENGTH_MPA, color='r', linestyle=':', linewidth=2,
                    label=f"Theoretical Yield ({THEORETICAL_YIELD_STRENGTH_MPA} MPa)")
    if ax_ss_2.has_data():
        ax_ss_2.legend(loc='best')
    ax_ss_2.grid(True, linestyle='--')

    fig_ss.tight_layout()
    fig_ss.savefig('stress_vs_strain_plots.png')
    print("Saved stress_vs_strain_plots.png")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()