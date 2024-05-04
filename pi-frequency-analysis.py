import numpy as np
import matplotlib.pyplot as plt

# Sample data: Replace 'pi_digits.txt' with the path to your Pi digits file
with open('pi_digits.txt', 'r') as file:
    pi_digits = file.read().replace('\n', '')

# Count the frequency of each digit
digit_counts = {str(digit): pi_digits.count(str(digit)) for digit in range(10)}

# Plotting the frequency of each digit
fig, ax = plt.subplots()
ax.bar(digit_counts.keys(), digit_counts.values())
ax.set_xlabel('Digits')
ax.set_ylabel('Frequency')
ax.set_title('Frequency Distribution of Pi Digits')
plt.show()
