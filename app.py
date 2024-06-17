import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from transformers import pipeline

# Load the trained Transformer model
transformer = keras.models.load_model('transformer_model.h5',
                                      custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                      'TransformerEncoder': TransformerEncoder,
                                                      'TransformerDecoder': TransformerDecoder,
                                                      'MultiHeadAttention': MultiHeadAttention})

# Load the vectorization layers
with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

target_vocab = target_vectorization.get_vocabulary()
target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
max_decoded_sentence_length = 30

# Function to decode sequence
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

    # Remove [start] and [end] tokens
    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

# Initialize sentiment analysis pipeline for German language
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Streamlit app layout
st.title('English to German Translation App')

st.write("""
This app translates English sentences to German using a Transformer model.
Additionally, it performs sentiment analysis on the translated German sentences.
""")

# Input text box
input_sentence = st.text_input("Enter an English sentence:")

if st.button("Translate"):
    if input_sentence:
        translated_sentence = decode_sequence(input_sentence)
        sentiment = sentiment_pipeline(translated_sentence)

        st.write("**English:**", input_sentence)
        st.write("**German:**", translated_sentence)
        st.write("**Sentiment:**", sentiment[0]['label'], "with score:", sentiment[0]['score'])
    else:
        st.write("Please enter a sentence to translate.")
