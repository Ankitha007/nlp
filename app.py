# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from keras.models import load_model

# Load the model and vectorizers
transformer = load_model('transformer_model.h5', compile=False)

with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

# Load the target vocabulary and lookup dictionary
target_vocab = target_vectorization.get_vocabulary()
target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))

max_decoded_sentence_length = 30

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

# Streamlit app
st.title("English to German Translation")

input_text = st.text_area("Enter English text here:")
if st.button("Translate"):
    if input_text:
        translation = decode_sequence(input_text)
        st.write("Translation:")
        st.write(translation)
    else:
        st.write("Please enter text to translate.")
