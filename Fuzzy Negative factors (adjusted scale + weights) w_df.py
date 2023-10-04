import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

# Load the dataset
df = pd.read_csv('negative_scaled.csv', encoding='Latin')

# Rename columns to match the membership functions
df = df.rename(columns={'Senior total': 'seniors', 'total_transports': 'transport', 'Number of Immigrants': 'immigrants'})

# New Antecedent/Consequent objects hold universe variables and membership functions
seniors = ctrl.Antecedent(np.arange(0, 101, 1), 'seniors')  # Updated for seniors
transport = ctrl.Antecedent(np.arange(0, 101, 1), 'transport')  # Updated for transport
immigrants = ctrl.Antecedent(np.arange(0, 101, 1), 'immigrants')  # Unchanged for immigrants
change = ctrl.Consequent(np.arange(0, 11, 1), 'change')

# Custom membership functions for 'seniors' based on the data
seniors['verylow'] = fuzz.trimf(seniors.universe, [0, 5, 12])
seniors['low'] = fuzz.trimf(seniors.universe, [10, 15, 25])
seniors['medium'] = fuzz.trimf(seniors.universe, [20, 30, 50])
seniors['high'] = fuzz.trimf(seniors.universe, [45, 50, 75])
seniors['veryhigh'] = fuzz.trimf(seniors.universe, [70, 80, 100])

# Custom membership functions for 'transport' to cover the entire range
transport['verylow'] = fuzz.trimf(transport.universe, [0, 10, 20])
transport['low'] = fuzz.trimf(transport.universe, [10, 30, 40])
transport['medium'] = fuzz.trimf(transport.universe, [30, 50, 60])
transport['high'] = fuzz.trimf(transport.universe, [50, 70, 80])
transport['veryhigh'] = fuzz.trimf(transport.universe, [80, 90, 100])

# Custom membership functions for 'immigrants' based on the data
immigrants['verylow'] = fuzz.trimf(immigrants.universe, [0, 5, 10]) * 0.8 # Custom weight of 0.8
immigrants['low'] = fuzz.trimf(immigrants.universe, [0, 10, 20]) * 0.8 # Custom weight of 0.8
immigrants['medium'] = fuzz.trimf(immigrants.universe, [15, 30, 40]) * 0.8 # Custom weight of 0.8
immigrants['high'] = fuzz.trimf(immigrants.universe, [35, 50, 60]) * 0.8 # Custom weight of 0.8
immigrants['veryhigh'] = fuzz.trimf(immigrants.universe, [55, 80, 100]) * 0.8 # Custom weight of 0.8

# Custom membership functions for 'change'
change['verylow'] = fuzz.trimf(change.universe, [0, 0, 2])
change['low'] = fuzz.trimf(change.universe, [1.5, 3, 4])
change['medium'] = fuzz.trimf(change.universe, [3.5, 5, 6])
change['high'] = fuzz.trimf(change.universe, [5.5, 7, 8])
change['veryhigh'] = fuzz.trimf(change.universe, [7.5, 9, 10])

## Define rules based on the data and membership functions, including 'immigrants'
rule1 = ctrl.Rule(seniors['verylow'] | transport['verylow'] | immigrants['verylow'], change['verylow'])
rule2 = ctrl.Rule(seniors['low'] | transport['low'] | immigrants['low'], change['low'])
rule3 = ctrl.Rule(seniors['medium'] | transport['medium'] | immigrants['medium'], change['medium'])
rule4 = ctrl.Rule(seniors['high'] | transport['high'] | immigrants['high'], change['high'])
rule5 = ctrl.Rule(seniors['veryhigh'] | transport['veryhigh'] | immigrants['veryhigh'], change['veryhigh'])

# Create the control system and simulation
change_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
changing = ctrl.ControlSystemSimulation(change_ctrl)

# Create a new 'change' column in the DataFrame and calculate values
df['change'] = np.nan  # Create an empty 'change' column
for index, row in df.iterrows():
    changing.input['seniors'] = row['seniors']  # Updated for seniors
    changing.input['transport'] = row['transport']  # Updated for transport
    changing.input['immigrants'] = row['immigrants']  # Unchanged for immigrants
    changing.compute()
    df.at[index, 'change'] = changing.output['change']

# Save the updated DataFrame to a new .csv file
df.to_csv('Negative changes output.csv', index=False, encoding='utf-8')

# Print and view the result
print(df)






