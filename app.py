import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import pipeline
from tensorflow import keras
import pickle
from custom_layers import PositionalEmbedding, MultiHeadAttention, TransformerEncoder, TransformerDecoder

# Load the saved Transformer model
transformer = keras.models.load_model('transformer_model.h5', custom_objects={
    'PositionalEmbedding': PositionalEmbedding,
    'MultiHeadAttention': MultiHeadAttention,
    'TransformerEncoder': TransformerEncoder,
    'TransformerDecoder': TransformerDecoder
})

# Load the vectorization layers
with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

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
st.title("English to German Translation and Sentiment Analysis")
st.write("Enter an English sentence and get its German translation along with sentiment analysis.")

input_sentence = st.text_input("Enter English sentence:")
if st.button("Translate and Analyze"):
    if input_sentence:
        translated_sentence = decode_sequence(input_sentence)
        # Initialize sentiment analysis pipeline for German language
        sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
        sentiment = sentiment_pipeline(translated_sentence)
        st.write("**German Translation:**", translated_sentence)
        st.write("**Sentiment Analysis:**", sentiment)
    else:
        st.write("Please enter a sentence.")
