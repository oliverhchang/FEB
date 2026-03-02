"""
Chassis FOS 3D Heatmap Visualizer (Dynamic Scaling)
Reads chassis_results.json and generates an interactive Plotly 3D model.
"""
import json
import plotly.graph_objects as go

INPUT_FILE = "chassis_results.json"

print(f"Loading {INPUT_FILE}...")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Run the stiffness solver first.")
    exit()

nodes = data["nodes"]
results = data["results"]

# 1. Analyze the FOS Spread to Create Dynamic Thresholds
# Filter out tubes that are essentially unloaded to avoid skewing the data
active_fos = [r.get("FOS", 999) for r in results if r.get("FOS", 999) < 100.0]

if active_fos:
    min_fos = min(active_fos)
    max_fos = max(active_fos)
else:
    min_fos = 1.0
    max_fos = 10.0

fos_range = max_fos - min_fos

# Create dynamic tiers at 10%, 25%, 50%, and 75% of the active FOS range
tier_1 = min_fos + (fos_range * 0.10)
tier_2 = min_fos + (fos_range * 0.25)
tier_3 = min_fos + (fos_range * 0.50)
tier_4 = min_fos + (fos_range * 0.75)

print(f"Dynamic FOS Spread Detected:")
print(f"  Absolute Min FOS: {min_fos}")
print(f"  Bottom 10% cutoff: {tier_1:.2f}")
print(f"  Bottom 25% cutoff: {tier_2:.2f}")

# 2. Dynamic Color Map Function
def get_dynamic_style(fos):
    if fos < 1.0:
        return "#ff0000", 7  # Red / Thick (Absolute Failure)
    elif fos <= tier_1:
        return "#ff5500", 6  # Dark Orange (Relative Weakest Links)
    elif fos <= tier_2:
        return "#ffaa00", 5  # Light Orange (Bottom 25%)
    elif fos <= tier_3:
        return "#fadb14", 4  # Yellow (Below Average)
    elif fos <= tier_4:
        return "#52c41a", 3  # Green (Above Average)
    elif fos < 100.0:
        return "#00d2ff", 2  # Light Blue (Very low stress)
    else:
        return "#0044ff", 2  # Deep Blue (Negligible load)

# 3. Build 3D Plot
print(f"Plotting {len(results)} members...")
fig = go.Figure()

for r in results:
    n1_key = str(r["n1"])
    n2_key = str(r["n2"])

    n1 = nodes[n1_key]
    n2 = nodes[n2_key]

    x_coords = [n1[0], n2[0]]
    y_coords = [n1[1], n2[1]]
    z_coords = [n1[2], n2[2]]

    fos = r.get("FOS", 999)
    color, width = get_dynamic_style(fos)

    # Build the hover text panel
    hover_text = (
        f"<b>Tube ID:</b> {r.get('id', 'N/A')}<br>"
        f"<b>Desc:</b> {r.get('desc', 'N/A')}<br>"
        f"<b>Length:</b> {round(r.get('L_true_mm', 0), 1)} mm<br>"
        f"<b>Force:</b> {r.get('F_N', 0)} N<br>"
        f"<b>Stress:</b> {r.get('sigma_MPa', 0)} MPa ({r.get('mode', 'N/A')})<br>"
        f"<b>FOS:</b> {fos}"
    )

    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color=color, width=width),
        hoverinfo='text',
        text=hover_text,
        name=f"T{r.get('id', '')}",
        showlegend=False
    ))

# 4. Layout and Formatting
fig.update_layout(
    title=f"FSAE Spaceframe FOS Heatmap (Dynamic Spread: Min FOS {min_fos})",
    scene=dict(
        aspectmode='data',
        xaxis=dict(title="X (mm)", showbackground=False),
        yaxis=dict(title="Y (mm)", showbackground=False),
        zaxis=dict(title="Z (mm)", showbackground=False),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=1.5, y=-1.5, z=1)
        )
    ),
    template="plotly_dark",
    margin=dict(l=0, r=0, b=0, t=40)
)

print("Opening visualizer in browser...")
fig.show()

print("Exporting to HTML...")
fig.write_html("FEB_SN5_Chassis_Analysis.html")

# Still open it locally for you to see
print("Opening visualizer in browser...")
fig.show()