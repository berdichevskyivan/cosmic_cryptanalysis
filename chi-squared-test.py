from scipy.stats import chisquare

# Example data: replace these with your actual frequencies
frequencies = [810, 800, 790, 805, 785, 815, 800, 830, 795, 810]

# Perform the chi-squared test
chi2_stat, p_value = chisquare(frequencies)

print(f"Chi-Squared Statistic: {chi2_stat}")
print(f"P-value: {p_value}")
