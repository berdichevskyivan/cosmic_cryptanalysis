import pandas as pd
import matplotlib.pyplot as plt

# Load Pi digits from a text file
with open('pi_digits.txt', 'r') as file:
    pi_digits = file.read().replace('\n', '')  # Read the file and remove newlines

# Convert string of digits into a list of pairs
pairs = [pi_digits[i:i+2] for i in range(len(pi_digits) - 1)]

# Create a DataFrame from pairs
df = pd.DataFrame(pairs, columns=['Pair'])

# Calculate frequencies
pair_counts = df['Pair'].value_counts()

# Normalize to get probabilities
pair_probs = pair_counts / pair_counts.sum()

# Print the result
print(pair_probs.head(10))  # Display top 10 most frequent pairs

# Plotting the frequencies
pair_probs.head(20).plot(kind='bar', figsize=(10, 5))  # Adjust the number to display more or fewer pairs
plt.title('Frequency of Digit Pairs in Pi')
plt.xlabel('Digit Pairs')
plt.ylabel('Frequency')
plt.show()
