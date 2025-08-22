import tensorflow as tf
import numpy as np
from model import Transformer
from data_preprocessing import get_tokenizer, prepare_dataset

class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = np.array(tokenizer.get_vocabulary())
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

    def __call__(self, seed_text, max_length=50, num_return_sequences=1):
        """
        Generates text from a seed text.
        """
        seed_tokens = self.tokenizer(tf.constant([seed_text]))
        
        # Start with the seed text
        input_tokens = seed_tokens[:, :-1] 

        for _ in range(max_length):
            # The context is the same as the input for a language model
            context = input_tokens 
            
            # Predict the next token
            predictions = self.model([context, input_tokens], training=False)
            
            # Select the last token from the seq_len dimension
            predictions = predictions[:, -1, :]  
            
            # Use top_k sampling to get some variety
            predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
            
            # Add the predicted token to the input
            input_tokens = tf.concat([input_tokens, [[predicted_id]]], axis=-1)

        # Convert the tokens back to text
        output_text = tf.strings.reduce_join(self.vocab[input_tokens], axis=-1)
        return output_text.numpy()[0].decode('utf-8')


if __name__ == '__main__':
    # Using the same sample corpus
    sample_corpus = [
        "TensorFlow is an open source machine learning platform.",
        "It is a comprehensive, flexible ecosystem of tools, libraries, and community resources.",
        "Keras is TensorFlow's high-level API for building and training deep learning models.",
        "It's used for fast prototyping, state-of-the-art research, and production.",
        "A transformer is a deep learning model that adopts the mechanism of self-attention."
    ]
    
    # These hyperparameters should be the same as in train.py
    NUM_LAYERS = 4
    D_MODEL = 128
    DFF = 512
    NUM_HEADS = 8
    DROPOUT_RATE = 0.1
    VOCAB_SIZE = 20000
    MAX_LEN = 50
    BATCH_SIZE = 64

    # 1. Data Preparation
    tokenizer = get_tokenizer(sample_corpus, VOCAB_SIZE, MAX_LEN)
    train_dataset = prepare_dataset(sample_corpus, tokenizer, BATCH_SIZE)

    # 2. Model Initialization (and training)
    # In a real-world scenario, you would save the trained model weights and load them here.
    # For this example, we re-train the model every time.
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT_RATE)
    
    # We are not compiling the model here because we are not training it,
    # but we will call fit to 'train' it for this example to have some weights.
    # In a real application, you would load pre-trained weights.
    
    # We call fit on a single batch to initialize the weights.
    for inputs, _ in train_dataset.take(1):
        transformer(inputs)
    
    # In a real use case, you should have a proper training loop an save the weights.
    # transformer.load_weights('path_to_your_saved_weights')

    # 3. Create a TextGenerator and generate text
    text_generator = TextGenerator(transformer, tokenizer)
    seed_text = "tensorflow is"
    generated_text = text_generator(seed_text, max_length=20)
    print(f"Seed: '{seed_text}'")
    print(f"Generated Text: '{generated_text}'")