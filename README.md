# Climate Change Impact on Agriculture

## Project Overview:
This project analyzes the impact of climate change on agriculture, examining factors like rising temperatures, extreme weather, and soil health. The analysis is based on a cleaned dataset and aims to explore how climate variations affect crop yields, economic outcomes, and farming practices.

## Course Code:
Course code:
1.Data Loading and Cleaning Script (data_cleaning.py)

import pandas as pd

# Load the dataset
df = pd.read_csv('climate_agriculture_raw_data.csv')

# Display basic information
print(df.info())
print(df.head())

# Drop duplicates (if any)
df.drop_duplicates(inplace=True)

# Fill or drop missing values
df.fillna({'column_name': 'value'}, inplace=True)  # Example, modify for each column
# df.dropna(inplace=True)  # Optionally, if you prefer to drop rows with missing data

# Convert columns to appropriate data types
df['year'] = pd.to_datetime(df['year'], format='%Y')

# Save cleaned data to a new file
df.to_csv('cleaned_climate_agriculture.csv', index=False)

# Display cleaned dataset info
print(df.info())

2. Data Analysis Script (data_analysis.py)

3. import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('cleaned_climate_agriculture.csv')

# Basic statistics
print(df.describe())

# Correlation analysis
corr_matrix = df.corr()
print(corr_matrix)

# Plotting
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='average_temperature_c', y='economic_impact_million_usd', hue='crop_type')
plt.title('Average Temperature vs Economic Impact')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Economic Impact (Million USD)')
plt.show()

3. Visualization Script (visualizations.py)

4. import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature_vs_yield(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='average_temperature_c', y='crop_yield', hue='crop_type')
    plt.title('Average Temperature vs Crop Yield')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Crop Yield (tons per hectare)')
    plt.show()

def plot_extreme_weather_impact(df):
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='extreme_weather_events', y='economic_impact_million_usd')
    plt.title('Extreme Weather Events vs Economic Impact')
    plt.xlabel('Number of Extreme Weather Events')
    plt.ylabel('Economic Impact (Million USD)')
    plt.show()

4. Dashboard Script (dashboard.py) using Dash

5. import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load the cleaned data
df = pd.read_csv('cleaned_climate_agriculture.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Create a figure for the plot
fig = px.scatter(df, x="average_temperature_c", y="economic_impact_million_usd", color="crop_type",
                 title="Average Temperature vs Economic Impact")

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Climate Change Impact on Agriculture"),
    dcc.Dropdown(
        id="region-dropdown",
        options=[{"label": region, "value": region} for region in df['region'].unique()],
        value="Punjab",  # default region
    ),
    dcc.Graph(id='temperature-impact-graph', figure=fig),
])

# Define callback to update graph based on region selection
@app.callback(
    dash.dependencies.Output('temperature-impact-graph', 'figure'),
    [dash.dependencies.Input('region-dropdown', 'value')]
)
def update_graph(selected_region):
    filtered_df = df[df['region'] == selected_region]
    fig = px.scatter(filtered_df, x="average_temperature_c", y="economic_impact_million_usd", color="crop_type",
                     title=f"Average Temperature vs Economic Impact in {selected_region}")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


## Dashboard:
The interactive dashboard showcasing the analysis can be accessed [here](https://dash.plotly.com/dash-in-jupyter).

## Files:
- `climate_agriculture_report.pdf` - A detailed report on the project findings.
- `cleaned_data.csv` - The cleaned dataset used for analysis.
- `analysis_code.ipynb` - Jupyter Notebook with code for the data analysis.

## Technologies Used:
- Python
- Pandas
- Seaborn, Matplotlib
- Plotly (for interactive visualizations)
- Dash (for dashboard development)

## Installation:
To run the analysis locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/PravalikaJedla77/Climate-Change-Agriculture.git

## Project Overview:
This project explores the effects of climate change on agriculture, focusing on how temperature changes, extreme weather, and soil health impact crop yields, farming practices, and economic outcomes. The dataset includes data on various regions, crop types, and environmental factors.

