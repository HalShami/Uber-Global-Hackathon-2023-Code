import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file into a DataFrame
df = pd.read_csv('Negative attributes.csv')

# Select the subset of data, excluding the first column and the first row
subset_df = df.iloc[0:, 1:]

# Create a MinMaxScaler object with the desired range [0, 100]
scaler = MinMaxScaler(feature_range=(0, 100))

# Fit the scaler to the subset_df and transform the subset_df
scaled_subset_df = scaler.fit_transform(subset_df)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_subset_df, columns=subset_df.columns)

# Write the scaled data to a new CSV file
scaled_df.to_csv('negative_scaled.csv', index=False)