from flask import Flask, request, jsonify
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

app = Flask(__name__)

# Define and register the custom standardization function
@register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Google Drive file ID for the model
model_file_id = '1jkeJL-gExfzMa2oWvno5PgRb465RhLUl'
model_path = 'transformer_model.h5'

# Function to verify file integrity
def verify_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            f.read()
        return True
    except Exception as e:
        return False

# Download the model file if it does not exist or is corrupted
if not os.path.exists(model_path) or not verify_file(model_path):
    gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)

# Verify if the model file exists and is not corrupted
model_loaded = False
if os.path.exists(model_path) and verify_file(model_path):
    try:
        transformer = keras.models.load_model(model_path, custom_objects={
            'PositionalEmbedding': PositionalEmbedding,
            'MultiHeadAttention': MultiHeadAttention,
            'TransformerEncoder': TransformerEncoder,
            'TransformerDecoder': TransformerDecoder,
            'custom_standardization': custom_standardization
        })
        model_loaded = True
        print("Transformer model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")

# Load the vectorization layers
source_vectorization_loaded = False
target_vectorization_loaded = False

try:
    with open('source_vectorization.pkl', 'rb') as f:
        source_vectorization = pickle.load(f)
    source_vectorization_loaded = True
    print("Source vectorization loaded successfully.")
except Exception as e:
    print(f"Error loading source vectorization: {e}")

try:
    with open('target_vectorization.pkl', 'rb') as f:
        target_vectorization = pickle.load(f)
    target_vectorization_loaded = True
    print("Target vectorization loaded successfully.")
except Exception as e:
    print(f"Error loading target vectorization: {e}")

# Define the maximum length for decoded sentences
max_decoded_sentence_length = 30

# Function to decode input sentence using the loaded transformer model
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

# Route for translation endpoint
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    input_sentence = data.get("sentence")
    if input_sentence:
        try:
            translated_sentence = decode_sequence(input_sentence)
            sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
            sentiment = sentiment_pipeline(translated_sentence)
            return jsonify({
                "translated_sentence": translated_sentence,
                "sentiment": sentiment
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No sentence provided"}), 400

if __name__ == '__main__':
    if model_loaded and source_vectorization_loaded and target_vectorization_loaded:
        target_vocab = target_vectorization.get_vocabulary()
        target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
        app.run(host='0.0.0.0', port=8000)
    else:
        print("Error: Model or vectorization layers failed to load.")
