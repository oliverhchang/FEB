import json
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "chassis_results.json"

# 1. Load Data
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Run the solver first.")
    exit()

results = data["results"]

# Filter out construction/unmatched lines for analysis
clean_results = [r for r in results if r['desc'] != 'CONSTRUCTION / UNMATCHED']

# Extract data for plotting
ids = [r['id'] for r in clean_results]
fos = [r['FOS'] for r in clean_results]
stress = [abs(r['sigma_MPa']) for r in clean_results]
modes = [r['mode'] for r in clean_results]
od = [r['OD_mm'] for r in clean_results]

# 2. Setup Figure
plt.figure(figsize=(10, 8))

# --- PLOT 1: Top 15 Weakest Tubes (Lowest FOS) ---
# Sorting to find the most critical tubes as per your planning
sorted_indices = np.argsort(fos)[:15]
top_ids = [str(ids[i]) for i in sorted_indices]
top_fos = [fos[i] for i in sorted_indices]
top_modes = [modes[i] for i in sorted_indices]

plt.subplot(2, 1, 1)
colors = ['#d63031' if m == 'COMP' else '#0984e3' for m in top_modes]
bars = plt.bar(top_ids, top_fos, color=colors, edgecolor='black', alpha=0.8)

plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Failure (1.0)')
plt.title("Critical Members: Lowest Factor of Safety", fontsize=12, fontweight='bold')
plt.ylabel("FOS Value")
plt.xlabel("Tube ID")
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend(['Failure Limit', 'Compression (Buckling)', 'Tension (Yield)'], loc='upper right')

# --- PLOT 2: Axial Stress vs. FOS ---
# Visualizing the spread to identify the "elbow" of the design
plt.subplot(2, 1, 2)
plt.scatter(stress, fos, c=od, cmap='viridis', s=50, edgecolors='black', alpha=0.7)

plt.yscale('log') # FOS spread is exponential; log scale shows detail better
plt.axhline(y=2.0, color='orange', linestyle='--', label='Design Target (2.0)')
plt.title("Design Spread: Axial Stress vs. FOS", fontsize=12, fontweight='bold')
plt.xlabel("Absolute Axial Stress (MPa)")
plt.ylabel("FOS (Log Scale)")
plt.colorbar(label='Tube OD (mm)')
plt.grid(True, which="both", ls="-", alpha=0.2)

# Final Layout
plt.tight_layout()
print("Displaying graphs in PyCharm...")
plt.show()

# Print Text Summary to Console
print("\n" + "-"*30)
print("QUICK CHASSIS AUDIT")
print("-"*30)
min_val = min(fos)
min_idx = ids[fos.index(min_val)]
print(f"Absolute Minimum FOS: {min_val} (Tube {min_idx})")
print(f"Max Stress Encountered: {max(stress):.2f} MPa")