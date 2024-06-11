import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import pipeline
from tensorflow import keras
import pickle
import os
import re
import string
from custom_layers import PositionalEmbedding, MultiHeadAttention, TransformerEncoder, TransformerDecoder
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import register_keras_serializable
import gdown

# Define and register the custom standardization function
@register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Google Drive file ID for the model
model_file_id = '1ib-A2IFcrlX-HN-QsGhHBjHQh5Ue5Zsm'
model_path = 'transformer_model.h5'

# Download the model file if it does not exist
if not os.path.exists(model_path):
    st.write(f"Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)

# Verify if the model file exists and is not corrupted
model_loaded = False
if os.path.exists(model_path):
    try:
        transformer = keras.models.load_model(model_path, custom_objects={
            'PositionalEmbedding': PositionalEmbedding,
            'MultiHeadAttention': MultiHeadAttention,
            'TransformerEncoder': TransformerEncoder,
            'TransformerDecoder': TransformerDecoder,
            'custom_standardization': custom_standardization
        })
        st.success("Model loaded successfully")
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.error("Model file not found or is corrupted.")

# Load the vectorization layers
source_vectorization_loaded = False
target_vectorization_loaded = False

try:
    with open('source_vectorization.pkl', 'rb') as f:
        source_vectorization = pickle.load(f)
    source_vectorization_loaded = True
except Exception as e:
    st.error(f"Error loading source vectorization: {e}")

try:
    with open('target_vectorization.pkl', 'rb') as f:
        target_vectorization = pickle.load(f)
    target_vectorization_loaded = True
except Exception as e:
    st.error(f"Error loading target vectorization: {e}")

if model_loaded and source_vectorization_loaded and target_vectorization_loaded:
    target_vocab = target_vectorization.get_vocabulary()
    target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
    max_decoded_sentence_length = 30

    def decode_sequence(input_sentence):
        tokenized_input_sentence = source_vectorization([input_sentence])
        st.write(f"Tokenized input sentence shape: {tokenized_input_sentence.shape}")
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
            st.write(f"Tokenized target sentence shape at step {i}: {tokenized_target_sentence.shape}")
            predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
            st.write(f"Predictions shape at step {i}: {predictions.shape}")
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = target_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token
            if sampled_token == "[end]":
                break
        decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
        return decoded_sentence

    st.title("English to German Translation and Sentiment Analysis")
    st.write("Enter an English sentence and get its German translation along with sentiment analysis.")

    input_sentence = st.text_input("Enter English sentence:")
    if st.button("Translate and Analyze"):
        if input_sentence:
            try:
                translated_sentence = decode_sequence(input_sentence)
                sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
                sentiment = sentiment_pipeline(translated_sentence)
                st.write("**German Translation:**", translated_sentence)
                st.write("**Sentiment Analysis:**", sentiment)
            except Exception as e:
                st.error(f"Error during translation or analysis: {e}")
else:
    st.error("Model or vectorization layers are not loaded correctly.")
