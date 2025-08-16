import matplotlib.pyplot as plt
import numpy as np

measurement_data = [
    {"tube1": "Front Bulkhead top", "tube2": "Front Bulkhead Bottom", "cad_length": 13.94, "actual_length": 13.875},
    {"tube1": "Front Bulkhead Left", "tube2": "Front Bulkhead Right", "cad_length": 15.78, "actual_length": 15.75},
    {"tube1": "Front Bulkhead", "tube2": "Front Hoop", "cad_length": 26.87, "actual_length": 26.9375},
    {"tube1": "Front Bulkhead", "tube2": "Front Hoop", "cad_length": 30.32, "actual_length": 30.1875},
    {"tube1": "OBS", "tube2": "Rear Bulkhead (Top)", "cad_length": 9.38, "actual_length": 9.4375},
    {"tube1": "OBS", "tube2": "Rear Bulkhead (Bottom)", "cad_length": 8.58, "actual_length": 8.5625},
    {"tube1": "Main Roll Hoop", "tube2": "Front Hoop", "cad_length": 26.817, "actual_length": 26.8125},
    {"tube1": "FBH Support Triangulation Node Left", "tube2": "FBH Support Triangulation Node Right", "cad_length": 19.402, "actual_length": 19.1875},
    {"tube1": "Front Hoop Left", "tube2": "Front Hoop Right", "cad_length": 20.926, "actual_length": 21.0625},
    {"tube1": "Main Roll Hoop Left", "tube2": "Main Roll Hoop Right", "cad_length": 25.805, "actual_length": 25.75},
    {"tube1": "RIS Top Left", "tube2": "RIS Top Right", "cad_length": 20.7583, "actual_length": 20.5625},
    {"tube1": "Main Roll Hoop Bracing Left (TOP)", "tube2": "Main Roll Hoop Bracing Right (TOP)", "cad_length": 12.95, "actual_length": 12.875},
    {"tube1": "Accumulator Protection Left", "tube2": "Accumulator Protection Right", "cad_length": 23.4263, "actual_length": 23.5},
    {"tube1": "Rear Bulkhead Top", "tube2": "Rear Bulkhead Bottom", "cad_length": 12.64, "actual_length": 12.625},
    {"tube1": "Rear Bulkhead Left (Bottom)", "tube2": "Rear Bulkhead Right (Bottom)", "cad_length": 21.614, "actual_length": 21.4375},
    {"tube1": "Shoulder Harness", "tube2": "Main Hoop Cross Bracing", "cad_length": 12.98, "actual_length": 13.3125},
    {"tube1": "Shoulder Harness", "tube2": "Floor", "cad_length": 23.6, "actual_length": 23.6875}
]

# Recalculate differences (actual - CAD)
for data in measurement_data:
    data["difference"] = data["actual_length"] - data["cad_length"]

# Extract data for plotting
differences = [data["difference"] for data in measurement_data]
measurements = [f"{data['tube1']} - {data['tube2']}" for data in measurement_data]

# Plotting code (same as your original)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

x_pos = np.arange(len(differences))
colors = ['red' if abs(x) > 0.1 else 'orange' if abs(x) > 0.05 else 'green' for x in differences]
bars = ax.bar(x_pos, differences, color=colors, alpha=0.7)

ax.set_xlabel('Measurements', fontsize=12)
ax.set_ylabel('Difference (inches)', fontsize=12)
ax.set_title('SN4 Chassis Measurement Differences (Actual - CAD)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([])  # Remove x-axis labels
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
ax.grid(True, alpha=0.3)

# Tolerance bands
ax.axhspan(-0.05, 0.05, alpha=0.2, color='green', label='±0.05" tolerance')
ax.axhspan(-0.1, -0.05, alpha=0.15, color='orange')
ax.axhspan(0.05, 0.1, alpha=0.15, color='orange', label='±0.05-0.1" tolerance')
ax.axhspan(-1, -0.1, alpha=0.1, color='red')
ax.axhspan(0.1, 1, alpha=0.1, color='red', label='>±0.1" tolerance')

# Value Bars
for i, (bar, val) in enumerate(zip(bars, differences)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
            f'{val:+.3f}"', ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=8, rotation=90)

ax.legend(loc='upper right')
plt.show()
