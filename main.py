import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import json
import tensorflow as tf


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        batch_input_shape=[batch_size, None]
    ))

    model.add(tf.keras.layers.LSTM(
        units=rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))

    model.add(tf.keras.layers.Dense(vocab_size))

    return model


with open('char2index.json') as json_file:
    char2index = Counter(json.load(json_file))
char2index = dict(char2index)

index2char = [word for word, index in char2index.items()]
#model = build_model(66, 256, 1024, 16)
# model.load_weights("modelis.h5")
model = tf.keras.models.load_model("modelis.h5")
model.build(tf.TensorShape([1, None]))


st.title('Lietuviškų Poemų generavimas')


def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    # Evaluation step (generating text using the learned model)

    # Converting our start string to numbers (vectorizing).
    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return (start_string + ''.join(text_generated))


user_input = st.text_input('Įveskite peomos pradžią: ')


if st.button('Generuoti poemą'):
    generated_text = generate_text(model, user_input.encode().decode())
    st.write(generated_text)
