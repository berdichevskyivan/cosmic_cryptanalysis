import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Data Preparation
def load_data(file_path, sequence_length=10):
    with open(file_path, 'r') as file:
        pi_digits = file.read().replace('\n', '').strip()

    # Map each character to its corresponding numerical value
    char_to_num = {char: int(char) for char in set(pi_digits)}

    # Convert sequences of digits to numerical arrays
    sequences = [[char_to_num[char] for char in seq] for seq in [pi_digits[i:i + sequence_length] for i in range(0, len(pi_digits) - sequence_length + 1, sequence_length)]]
    return sequences

# Model Architecture
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

# Training
def train_autoencoder(autoencoder, sequences, epochs=50, batch_size=32):
    sequences = np.array(sequences)
    autoencoder.fit(sequences, sequences, epochs=epochs, batch_size=batch_size)

# Analysis of Learned Representations
def analyze_representation(encoder, sequences):
    encoded_seqs = encoder.predict(np.array(sequences))
    return encoded_seqs

# Visualize the Encoded Representations using t-SNE
def visualize_encoded_representations(encoded_seqs):
    tsne = TSNE(n_components=2, random_state=42)
    encoded_tsne = tsne.fit_transform(encoded_seqs)

    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization of Encoded Representations')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# Analyze Patterns in the Latent Space
def analyze_patterns_in_latent_space(encoded_seqs, sequences):
    # Find sequences that are close together in the latent space
    close_sequences = []
    for i in range(len(encoded_seqs)):
        for j in range(i+1, len(encoded_seqs)):
            if np.linalg.norm(encoded_seqs[i] - encoded_seqs[j]) < 5:  # Adjust the threshold as needed
                close_sequences.append((sequences[i], sequences[j]))

    # Print the close sequences
    for seq_pair in close_sequences:
        print("Close Sequences:")
        print("Sequence 1:", seq_pair[0])
        print("Sequence 2:", seq_pair[1])
        print()

# File path to the digits of Pi
file_path = 'pi_digits.txt'

# Load and prepare data
sequences = load_data(file_path, sequence_length=10)

# Define autoencoder parameters
input_dim = len(sequences[0])
encoding_dim = 2  # You can adjust the encoding dimension

# Build autoencoder
autoencoder, encoder, decoder = build_autoencoder(input_dim, encoding_dim)

# Train autoencoder
train_autoencoder(autoencoder, sequences)

# Analyze learned representations
encoded_seqs = analyze_representation(encoder, sequences)
print("Encoded representations:", encoded_seqs)

# Visualize the encoded representations
visualize_encoded_representations(encoded_seqs)

# Analyze patterns in the latent space
analyze_patterns_in_latent_space(encoded_seqs, sequences)
