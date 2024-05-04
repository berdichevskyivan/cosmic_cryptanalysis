import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Function to load data and create sequences
def load_data_and_create_sequences(file_path, sequence_length=10):
    with open(file_path, 'r') as file:
        pi_digits = file.read().replace('\n', '').strip()  # Read and clean the file

    # Split the digits into sequences of fixed length
    sequences = [pi_digits[i:i + sequence_length] for i in range(0, len(pi_digits) - sequence_length + 1, sequence_length)]
    return sequences

# Load the data and prepare sequences
file_path = 'pi_digits.txt'  # Make sure this path is correct
sequences = load_data_and_create_sequences(file_path, sequence_length=10)  # You can adjust the sequence length

# Vectorize the sequences
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,1))  # Character-level vectorization
X = vectorizer.fit_transform(sequences)

# Clustering
num_clusters = 5  # Adjust based on analysis
km = KMeans(n_clusters=num_clusters)
km.fit(X)
clusters = km.labels_

# Analyze clusters
print(f"Total sequences processed: {len(sequences)}")
for i in range(num_clusters):
    cluster_sequences = [sequences[j] for j in range(len(sequences)) if clusters[j] == i]
    print(f"Cluster {i}: {cluster_sequences[:5]}")  # Print first 5 sequences of each cluster
