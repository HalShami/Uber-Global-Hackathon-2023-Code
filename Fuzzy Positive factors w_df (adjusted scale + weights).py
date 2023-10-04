import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd  # Add the pandas library

# Load the dataset
df = pd.read_csv('scaled_data.csv', encoding='Latin')

# Rename columns to match the membership functions
df = df.rename(columns={'total_accidents': 'accidents', 'bad_air_quality_count': 'pollution'})

# New Antecedent/Consequent objects hold universe variables and membership functions
accidents = ctrl.Antecedent(np.arange(0, 101, 1), 'accidents')
pollution = ctrl.Antecedent(np.arange(0, 101, 1), 'pollution')
change = ctrl.Consequent(np.arange(0, 11, 1), 'change')

# Custom membership functions for 'accidents' based on the data
accidents['verylow'] = fuzz.trimf(accidents.universe, [0, 5, 10])
accidents['low'] = fuzz.trimf(accidents.universe, [0, 10, 20])
accidents['medium'] = fuzz.trimf(accidents.universe, [15, 30, 40])
accidents['high'] = fuzz.trimf(accidents.universe, [35, 50, 60])
accidents['veryhigh'] = fuzz.trimf(accidents.universe, [55, 80, 100])

# Custom membership functions for 'pollution' to cover the entire range
pollution['verylow'] = fuzz.trimf(pollution.universe, [0, 10, 20])
pollution['low'] = fuzz.trimf(pollution.universe, [10, 30, 40])
pollution['medium'] = fuzz.trimf(pollution.universe, [30, 50, 60])
pollution['high'] = fuzz.trimf(pollution.universe, [50, 70, 80])
pollution['veryhigh'] = fuzz.trimf(pollution.universe, [75, 90, 100])

# Custom membership functions for 'change'
change['verylow'] = fuzz.trimf(change.universe, [0, 0, 0.5])
change['low'] = fuzz.trimf(change.universe, [0.5, 2, 3])
change['medium'] = fuzz.trimf(change.universe, [3, 4, 5])
change['high'] = fuzz.trimf(change.universe, [5, 6, 7])
change['veryhigh'] = fuzz.trimf(change.universe, [7, 9, 10])

# Define rules based on the data and membership functions
rule1 = ctrl.Rule(accidents['verylow'] | pollution['verylow'], change['verylow'])
rule2 = ctrl.Rule(accidents['low'] | pollution['low'], change['low'])
rule3 = ctrl.Rule(accidents['medium'] | pollution['medium'], change['medium'])
rule4 = ctrl.Rule(accidents['high'] | pollution['high'], change['high'])
rule5 = ctrl.Rule(accidents['veryhigh'] | pollution['veryhigh'], change['veryhigh'])


# Create the control system and simulation
change_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
changing = ctrl.ControlSystemSimulation(change_ctrl)

# Create a new 'change' column in the DataFrame and calculate values
df['change'] = np.nan  # Create an empty 'change' column
for index, row in df.iterrows():
    changing.input['accidents'] = row['accidents']
    changing.input['pollution'] = row['pollution']
    changing.compute()
    df.at[index, 'change'] = changing.output['change']

# Save the updated DataFrame to a new .csv file
df.to_csv('Postive factors output - adjusted scale + weights.csv', index=False, encoding='utf-8')

# Print and view the result
print(df)