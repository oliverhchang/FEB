"""
Chassis Geometry 3D Visualizer
Reads chassis_analysis.json and generates an interactive Plotly 3D model of the raw tubes.
"""
import json
import plotly.graph_objects as go

INPUT_FILE = "chassis_analysis.json"

print(f"Loading {INPUT_FILE}...")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Run the extraction script first.")
    exit()

nodes = data["nodes"]
members = data["members"]

print(f"Plotting {len(members)} tubes...")
fig = go.Figure()

for mem in members:
    n1 = nodes[str(mem["n1"])]
    n2 = nodes[str(mem["n2"])]

    x_coords = [n1[0], n2[0]]
    y_coords = [n1[1], n2[1]]
    z_coords = [n1[2], n2[2]]

    hover_text = (
        f"<b>Tube ID:</b> {mem['id']}<br>"
        f"<b>OD:</b> {mem['OD_mm']} mm<br>"
        f"<b>Length:</b> {mem['L_mm']} mm"
    )

    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color='#00d2ff', width=4),
        hoverinfo='text',
        text=hover_text,
        name=f"T{mem['id']}",
        showlegend=False
    ))

fig.update_layout(
    title="FSAE Spaceframe Raw Geometry",
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