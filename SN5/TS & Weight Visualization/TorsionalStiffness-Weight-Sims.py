import pandas as pd
import plotly.express as px

# Combine all data into a single list of dictionaries
# This is similar to how you would fetch and process data in a real application.
all_data = [
    # SN4 Data
    {'iteration': 'SN4-1', 'weight': 30.27, 'ts': 1228.517641, 'name': 'Baseline', 'version': 'SN4'},

    # Version 5.1 Data
    {'iteration': '1', 'weight': 25.51, 'ts': 797.388179, 'name': 'Baseline', 'version': '5.1'},
    {'iteration': '2', 'weight': 25.30, 'ts': 798.9367227, 'name': 'Shoulder Harness Brace', 'version': '5.1'},

    # Version 5.2 Data
    {'iteration': '3', 'weight': 32.78, 'ts': 1101.447197, 'name': 'Baseline Geometry 10/8', 'version': '5.2'},
    {'iteration': '4', 'weight': 31.80, 'ts': 995.0880059, 'name': 'No seat mounting or FRH bracing', 'version': '5.2'},
    {'iteration': '5', 'weight': 32.32, 'ts': 1063.609587, 'name': 'No seat mounting tube', 'version': '5.2'},
    {'iteration': '6', 'weight': 32.75, 'ts': 1087.925136, 'name': '1.25 --> 1 Seat Belt Tube', 'version': '5.2'},
    {'iteration': '7', 'weight': 33.04, 'ts': 1131.966514, 'name': 'Top X Brace | Damper to FRH', 'version': '5.2'},
    {'iteration': '8', 'weight': 33.03, 'ts': 907.575528, 'name': 'Bot X Brace | Steering to FRH', 'version': '5.2'},
    {'iteration': '9', 'weight': 34.43, 'ts': 903.8934086, 'name': 'Double Accumulator Protection Brace',
     'version': '5.2'},
    {'iteration': '10', 'weight': 33.09, 'ts': 1114.476689, 'name': 'Bot X Brace | Seat Tube to FRH', 'version': '5.2'},
    {'iteration': '11', 'weight': 33.30, 'ts': 1326.174884, 'name': 'Bot X Brace | Front', 'version': '5.2'},
    {'iteration': '12', 'weight': 33.31, 'ts': 1294.845019, 'name': 'Top X Brace | Front', 'version': '5.2'},
    {'iteration': '13', 'weight': 33.31, 'ts': 522.0569406, 'name': 'Top X Brace | Front', 'version': '5.2'},

    # Version 5.4 Data
    {'iteration': '15', 'weight': 30.08, 'ts': 1201.814994, 'name': '10/19 Geometry Baseline', 'version': '5.4'},
    {'iteration': '16', 'weight': 30.49, 'ts': 1233.201048, 'name': 'Baseline w/ previous damper lateral tube',
     'version': '5.4'},
    {'iteration': '17', 'weight': 30.02, 'ts': 1220.372088, 'name': 'Front bulkhead vertical thin', 'version': '5.4'},
    {'iteration': '18', 'weight': 30.02, 'ts': 1191.080371, 'name': 'Front a-arm vertical thin', 'version': '5.4'},
    {'iteration': '19', 'weight': 29.77, 'ts': 1173.129438, 'name': 'FRH Bottom Lateral Tube', 'version': '5.4'},
    {'iteration': '20', 'weight': 30.00, 'ts': 1222.792217, 'name': 'Steering mounting tube', 'version': '5.4'},
    {'iteration': '21', 'weight': 29.98, 'ts': 1207.83663, 'name': 'Driver restraint harness tube', 'version': '5.4'},
    {'iteration': '22', 'weight': 30.68, 'ts': 1466.882945, 'name': 'Top X', 'version': '5.4'},
    {'iteration': '23a', 'weight': 30.42, 'ts': 1420.981055, 'name': 'Top Diagonal Direction 1', 'version': '5.4'},
    {'iteration': '23b', 'weight': 30.42, 'ts': 1430.928425, 'name': 'Top Diagonal Direction 2', 'version': '5.4'},
    {'iteration': '24', 'weight': 30.41, 'ts': 1209.206928, 'name': 'Bot Driver Harness to FRH X', 'version': '5.4'},
    {'iteration': '25a', 'weight': 30.42, 'ts': 1252.842611, 'name': 'Bot FRH to Steering X', 'version': '5.4'},
    {'iteration': '25b', 'weight': 30.61, 'ts': 1406.51548, 'name': 'Front Bulkhead to Steering X', 'version': '5.4'},
    {'iteration': '26', 'weight': 30.02, 'ts': 1202.841898, 'name': 'Rear triangulation', 'version': '5.4'},
    {'iteration': '27', 'weight': 29.60, 'ts': 1122.104198, 'name': 'No rear triangulation', 'version': '5.4'},
]

# Create a pandas DataFrame from the data
df = pd.DataFrame(all_data)

# Create the scatter plot
# The 'color' argument automatically groups data by the 'version' column and creates a legend.
fig = px.scatter(
    df,
    x='weight',
    y='ts',
    color='version',
    title='Torsional Stiffness vs Weight',
    # Pass extra data columns to be used in the hover template
    custom_data=['name', 'iteration', 'version']
)

# Customize the appearance and hover tooltip to match the React example
fig.update_layout(
    xaxis_title='Weight (lbs)',
    yaxis_title='Torsional Stiffness (lb-ft/deg)',
    legend_title_text='Version'  # Renames the legend title
)

# Update the hover template for a custom tooltip experience
# <br> is HTML for a line break
# %{x} and %{y} are the x and y values
# %{customdata[i]} refers to the columns passed in the custom_data list
# :.1f formats the y-value to one decimal place
fig.update_traces(
    hovertemplate="<b>Iteration %{customdata[1]} (v%{customdata[2]})</b><br><br>" +
                  "Name: %{customdata[0]}<br>" +
                  "Weight: %{x} lbs<br>" +
                  "TS: %{y:.1f} lb-ft/deg<extra></extra>"  # <extra></extra> hides the secondary box
)

# Display the interactive chart
fig.show()