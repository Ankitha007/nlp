import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from transformers import pipeline
import os
import subprocess
import h5py

# Define custom layers
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return keras.backend.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = keras.layers.Dense(embed_dim)
        self.key_dense = keras.layers.Dense(embed_dim)
        self.value_dense = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        inputs_norm = self.layernorm_1(inputs)
        attention_output = self.attention([inputs_norm, inputs_norm, inputs_norm])
        proj_input = self.layernorm_2(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return proj_input + proj_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = MultiHeadAttention(embed_dim, num_heads)
        self.attention_2 = MultiHeadAttention(embed_dim, num_heads)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()

    def call(self, inputs, encoder_outputs, training=False):
        inputs_norm = self.layernorm_1(inputs)
        attention_output_1 = self.attention_1([inputs_norm, inputs_norm, inputs_norm])
        out_1 = self.layernorm_2(inputs + attention_output_1)
        attention_output_2 = self.attention_2([out_1, encoder_outputs, encoder_outputs])
        out_2 = self.layernorm_3(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return out_2 + proj_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

# Download the model file if it does not exist
model_path = 'transformer_model.h5'
if not os.path.isfile(model_path):
    subprocess.run(['curl --output transformer_model.h5 "https://media.githubusercontent.com/media/Ankitha007/nlp/main/transformer_model.h5"'], shell=True)

# Verify file paths
source_vector_path = 'source_vectorization.pkl'
target_vector_path = 'target_vectorization.pkl'

assert os.path.exists(source_vector_path), f"Source vectorization file not found: {source_vector_path}"
assert os.path.exists(target_vector_path), f"Target vectorization file not found: {target_vector_path}"

# Function to check if the HDF5 file is valid
def is_hdf5(filepath):
    try:
        with h5py.File(filepath, 'r'):
            return True
    except Exception:
        return False

# Check if the model file is a valid HDF5 file
if not is_hdf5(model_path):
    raise ValueError(f"The model file at {model_path} is not a valid HDF5 file or is corrupted.")

# Load the trained Transformer model within a custom object scope
custom_objects = {
    "PositionalEmbedding": PositionalEmbedding,
    "MultiHeadAttention": MultiHeadAttention,
    "TransformerEncoder": TransformerEncoder,
    "TransformerDecoder": TransformerDecoder
}

with keras.utils.custom_object_scope(custom_objects):
    transformer = keras.models.load_model(model_path)

# Load the vectorization layers
with open(source_vector_path, 'rb') as f:
    source_vectorization = pickle.load(f)

with open(target_vector_path, 'rb') as f:
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
This app translates English
