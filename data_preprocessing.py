import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os

def get_tokenizer(corpus, vocab_size, max_len):
    """
    Creates and adapts a TextVectorization layer to the corpus.
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_len + 1)
    vectorize_layer.adapt(corpus)
    return vectorize_layer

def prepare_dataset(corpus, tokenizer, batch_size):
    """
    Takes a corpus and a tokenizer, and returns a TensorFlow dataset.
    """
    vectorized_corpus = tokenizer(corpus)
    
    def create_input_target_pair(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = tf.data.Dataset.from_tensor_slices(vectorized_corpus)
    sequences = dataset.batch(tokenizer.get_config()['output_sequence_length'], drop_remainder=True)
    dataset = sequences.map(create_input_target_pair)
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

if __name__ == '__main__':
    # This is a sample corpus. 
    # In a real-world scenario, you would load a large text file.
    sample_corpus = [
        "TensorFlow is an open source machine learning platform.",
        "It is a comprehensive, flexible ecosystem of tools, libraries, and community resources.",
        "Keras is TensorFlow's high-level API for building and training deep learning models.",
        "It's used for fast prototyping, state-of-the-art research, and production.",
        "A transformer is a deep learning model that adopts the mechanism of self-attention."
    ]

    VOCAB_SIZE = 20000
    MAX_LEN = 50
    BATCH_SIZE = 64

    # 1. Get tokenizer
    tokenizer = get_tokenizer(sample_corpus, VOCAB_SIZE, MAX_LEN)
    vocab = tokenizer.get_vocabulary()
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Vocabulary: {vocab[:10]}")

    # 2. Prepare dataset
    train_dataset = prepare_dataset(sample_corpus, tokenizer, BATCH_SIZE)
    
    # 3. Inspect a batch
    for input_batch, target_batch in train_dataset.take(1):
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Target batch shape: {target_batch.shape}")
        print(f"Input batch sample:\n{input_batch[0]}")
        print(f"Target batch sample:\n{target_batch[0]}")