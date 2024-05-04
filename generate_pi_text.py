import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load digits of Pi text from file
def load_pi_digits(file_path):
    with open(file_path, 'r') as file:
        pi_digits_text = file.read().replace('\n', '').strip()
    return pi_digits_text

# Define character-to-index and index-to-character mappings
def create_char_mappings(text):
    chars = sorted(list(set(text)))
    char_indices = {char: i for i, char in enumerate(chars)}
    indices_char = {i: char for i, char in enumerate(chars)}
    return char_indices, indices_char

# Data Preparation
def prepare_data(text, maxlen, step):
    sequences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sequences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    return sequences, next_chars

# Model Architecture
def build_model():
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer, model

# Text Generation
def sample(preds, temperature=1.0):
    preds = np.nan_to_num(preds)  # Handle NaN values
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed_text, maxlen, char_indices, indices_char, temperature=1.0, num_chars=400):
    generated_text = seed_text
    for i in range(num_chars):
        sampled = np.zeros((1, len(seed_text), len(char_indices)))
        for t, char in enumerate(seed_text):
            sampled[0, t, char_indices[char]] = 1.0
        preds = model.predict(sampled, verbose=0)[0][-1]  # Select the last prediction for the next character
        
        # Handle NaN values
        preds = np.nan_to_num(preds)
        
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

# def softmax(x, axis=None):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
#     return e_x / e_x.sum(axis=axis, keepdims=True)

# Main
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'pi_digits.txt'
    pi_digits_text = load_pi_digits(file_path)
    maxlen = 40
    step = 3
    sequences, next_chars = prepare_data(pi_digits_text, maxlen, step)
    char_indices, indices_char = create_char_mappings(pi_digits_text)

    # Build and train model
    tokenizer, model = build_model()
    # Train model with sequences and next_chars

    # Generate text
    seed_text = '314159265358979323846264338327950288'
    generated_text = generate_text(model, seed_text, maxlen, char_indices, indices_char)
    print(generated_text)
