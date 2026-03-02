import json
import matplotlib.pyplot as plt

INPUT_FILE = "chassis_results.json"

# 1. Load and prepare data
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Run the solver script first.")
    exit()

results = data["results"]

# Remove construction/unmatched lines to focus on real tubes [cite: 49, 64]
clean_results = [r for r in results if r['desc'] != 'CONSTRUCTION / UNMATCHED']

# Sort to identify the 20 weakest tubes [cite: 63, 210]
sorted_results = sorted(clean_results, key=lambda x: x['FOS'])[:20]

ids = [str(r['id']) for r in sorted_results]
fos_values = [r['FOS'] for r in sorted_results]
modes = [r['mode'] for r in sorted_results]

# 2. Plain Formatting Plot
plt.figure(figsize=(10, 6))

# Assign colors based on failure mode: Red for Compression/Buckling, Blue for Tension [cite: 40, 41, 203]
colors = []
for m in modes:
    if m == 'COMP':
        colors.append('red')
    else:
        colors.append('blue')

plt.bar(ids, fos_values, color=colors)

# Standard labeling
plt.title("Factor of Safety per Tube ID (Top 20 Critical)")
plt.xlabel("Tube ID")
plt.ylabel("Factor of Safety")
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

# Console output summary [cite: 204, 212]
print(f"Minimum Chassis FOS: {min(fos_values)}")