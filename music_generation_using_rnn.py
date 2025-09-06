
#Importing Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam

# For Transformers
from tensorflow.keras import layers

# For MIDI processing
import pretty_midi
import glob

import kagglehub
from google.colab import files
import zipfile
import os

import zipfile
with zipfile.ZipFile("Antonello_Venditti.zip", 'r') as zip_ref:
    zip_ref.extractall("Antonello_Venditti")

midi_files = glob.glob("/content/Antonello_Venditti/Antonello_Venditti/*.mid")  # folder with MIDI songs

notes = []
for file in midi_files:
    midi = pretty_midi.PrettyMIDI(file)
    # Example: get piano roll or note sequence
    for instrument in midi.instruments:
        if not instrument.is_drum:   # ignore drums
            for note in instrument.notes:
                notes.append(note.pitch)  # store pitches only

# Convert notes into integers
unique_notes = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(unique_notes)}
int_to_note = {num: note for num, note in enumerate(unique_notes)}

encoded_notes = [note_to_int[n] for n in notes]
print("Total MIDI files:", len(midi_files))
print(midi_files[:5])  # show first 5

SEQ_LENGTH=50   # how many notes per training example

X, y = [], []
for i in range(len(encoded_notes) - SEQ_LENGTH):
    seq = encoded_notes[i:i+SEQ_LENGTH]
    target = encoded_notes[i+SEQ_LENGTH]
    X.append(seq)
    y.append(target)

X = np.array(X)
y = np.array(y)

# Train/test split
split = int(0.9* len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

rnn_model = Sequential([
    Embedding(len(unique_notes), 128),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(128, activation="relu"),
    Dense(len(unique_notes), activation="softmax")
])

rnn_model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adam(0.001),
                  metrics=["accuracy"])

rnn_model.summary()

def transformer_model(seq_len, vocab_size):
    inputs = layers.Input(shape=(seq_len,))
    x = layers.Embedding(vocab_size, 128)(inputs)

    # Transformer block
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    x = layers.LayerNormalization()(x + attention)
    ffn = layers.Dense(256, activation="relu")(x)
    ffn = layers.Dense(128)(ffn)
    x = layers.LayerNormalization()(x + ffn)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

transformer = transformer_model(SEQ_LENGTH, len(unique_notes))
transformer.compile(loss="sparse_categorical_crossentropy",
                    optimizer=Adam(0.001),
                    metrics=["accuracy"])

transformer.summary()

# Train RNN
rnn_model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=1, batch_size=2048)

# Train Transformer
transformer.fit(X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1, batch_size=2048)

def generate_music(model, seed_sequence, length=200):
    generated = seed_sequence.copy()

    for _ in range(length):
        X_input = np.array(generated[-SEQ_LENGTH:]).reshape(1, -1)
        preds = model.predict(X_input, verbose=0)[0]
        next_note = np.argmax(preds)
        generated.append(next_note)

    return generated

# Example usage
seed = X[100].tolist()
gen_notes_rnn = generate_music(rnn_model, seed, 200)
gen_notes_trf = generate_music(transformer, seed, 200)

def notes_to_midi(note_sequence, filename="output.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    start = 0
    for n in note_sequence:
        note = int_to_note[n]
        note_obj = pretty_midi.Note(
            velocity=100, pitch=note,
            start=start, end=start+0.5
        )
        instrument.notes.append(note_obj)
        start += 0.5

    midi.instruments.append(instrument)
    midi.write(filename)

# Save results
notes_to_midi(gen_notes_rnn, "generated_rnn.mid")
notes_to_midi(gen_notes_trf, "generated_transformer.mid")