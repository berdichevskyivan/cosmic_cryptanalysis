import pandas as pd

# Load the digit data
with open('pi_digits.txt', 'r') as file:
    pi_digits = file.read().replace('\n', '')

# Convert string of digits into a list of integers
digits = [int(d) for d in pi_digits]

# Create a pandas Series from the list of digits
digit_series = pd.Series(digits)

# Calculate and print autocorrelation for different lags
for lag in range(1, 11):  # Checking the first 10 lags
    autocorr = digit_series.autocorr(lag)
    print(f"Autocorrelation for lag {lag}: {autocorr:.4f}")
