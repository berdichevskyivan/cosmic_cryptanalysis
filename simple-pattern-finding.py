def find_repeating_sequences(pi_digits, sequence_length=4):
    repeating_sequences = set()
    for i in range(len(pi_digits) - sequence_length + 1):
        sequence = pi_digits[i:i + sequence_length]
        if sequence in pi_digits[i + sequence_length:]:
            repeating_sequences.add(sequence)
    return repeating_sequences

def find_prime_number_occurrences(pi_digits, primes):
    prime_occurrences = {}
    for prime in primes:
        prime_str = str(prime)
        if prime_str in pi_digits:
            occurrences = [pos for pos in range(len(pi_digits)) if pi_digits.startswith(prime_str, pos)]
            prime_occurrences[prime] = occurrences
    return prime_occurrences

def find_mathematical_constants(pi_digits, constants):
    constant_occurrences = {}
    for constant in constants:
        if constant in pi_digits:
            occurrences = [pos for pos in range(len(pi_digits)) if pi_digits.startswith(constant, pos)]
            constant_occurrences[constant] = occurrences
    return constant_occurrences

def find_palindrome_sequences(pi_digits, sequence_length=4):
    palindrome_sequences = set()
    for i in range(len(pi_digits) - sequence_length + 1):
        sequence = pi_digits[i:i + sequence_length]
        if sequence == sequence[::-1]:
            palindrome_sequences.add(sequence)
    return palindrome_sequences

def preprocess_digits(file_path):
    with open(file_path, 'r') as file:
        pi_digits = file.read().strip()  # Read the file and remove leading/trailing whitespace

    # Remove any unwanted characters
    pi_digits = ''.join(filter(str.isdigit, pi_digits))

    return pi_digits

# Define the mathematical constants and primes to search for
math_constants = ['314', '2718', '1.618', '1.414']
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# File path to the digits of Pi
file_path = 'pi_digits.txt'

# Preprocess the digits (if needed)
pi_digits = preprocess_digits(file_path)

# Find repeating sequences
repeating_sequences = find_repeating_sequences(pi_digits)
print("Repeating Sequences:", repeating_sequences)

# Find prime number occurrences
prime_occurrences = find_prime_number_occurrences(pi_digits, primes)
print("Prime Number Occurrences:", prime_occurrences)

# Find mathematical constants occurrences
constant_occurrences = find_mathematical_constants(pi_digits, math_constants)
print("Mathematical Constants Occurrences:", constant_occurrences)

# Find palindrome sequences
palindrome_sequences = find_palindrome_sequences(pi_digits)
print("Palindrome Sequences:", palindrome_sequences)
