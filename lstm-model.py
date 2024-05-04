import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to load data
def load_data(file_path):
    with open(file_path, 'r') as file:
        pi_digits = file.read().replace('\n', '').strip()  # Read and clean the file
    return pi_digits

# Function to create subsequences from the digits
def create_sequences(digits, sequence_length):
    sequences = []
    for i in range(len(digits) - sequence_length):
        sequence = digits[i:i+sequence_length + 1]
        sequences.append([int(char) for char in sequence])
    return sequences

# Load the data
file_path = 'pi_digits.txt'
pi_digits = load_data(file_path)

# Create sequences (choose an appropriate sequence length)
sequence_length = 10  # This is an arbitrary length; adjust as needed
sequences = create_sequences(pi_digits, sequence_length)

# Prepare the data for LSTM
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM [samples, time steps, features]

#### Step 2: Define and Compile the LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#### Step 3: Fit the Model
model.fit(X, y, epochs=300, verbose=1)

# This setup assumes you are predicting the next digit in the sequence
# Adjust the complexity and the parameters as needed based on the performance and available computational resources
