import pandas as pd
import plotly.express as px

# --- 1. Define File Names ---
csv_file_name = 'FEB _ SN5 Chassis _ Torsional Stiffness Tracker - Mass.csv'
output_html_file = 'chassis_weight_vs_iteration.html'

# --- 2. Load and Process Data ---
try:
    # Load the CSV file into a pandas DataFrame, skipping the empty first row
    df = pd.read_csv(csv_file_name, skiprows=1)

    # --- 3. Create the Plot ---
    fig = px.scatter(
        df,
        x='Iteration',
        y='Mass',
        color='Name',
        title='Chassis Tube Weight vs. Iteration',
        custom_data=['Name', 'Iteration']
    )

    # Add lines to connect the markers for each 'Name' group
    fig.update_traces(mode='lines+markers')

    # Customize the appearance
    fig.update_layout(
        xaxis_title='Iteration Number',
        yaxis_title='Weight (lbs)',
        legend_title_text='Concept / Name'
    )

    # Customize the hover tooltip
    fig.update_traces(
        hovertemplate="<b>Iteration %{x}</b><br><br>" +
                      "Name: %{customdata[0]}<br>" +
                      "Weight: %{y:.2f} lbs<extra></extra>"
    )

    # --- 4. Save to a Single HTML File ---
    # This bundles all the data and interactive code into one file
    fig.write_html(output_html_file)

    print(f"\nSuccessfully created single-file visualization!")
    print(f"You can now open '{output_html_file}' in any web browser.")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")