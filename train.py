import tensorflow as tf
import time
from model import Transformer
from data_preprocessing import get_tokenizer, prepare_dataset

# A custom learning rate scheduler is often used for training Transformers.
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


if __name__ == '__main__':
    # Using the same sample corpus
    sample_corpus = [
        "TensorFlow is an open source machine learning platform.",
        "It is a comprehensive, flexible ecosystem of tools, libraries, and community resources.",
        "Keras is TensorFlow's high-level API for building and training deep learning models.",
        "It's used for fast prototyping, state-of-the-art research, and production.",
        "A transformer is a deep learning model that adopts the mechanism of self-attention."
    ]

    # 1. Hyperparameters
    NUM_LAYERS = 4
    D_MODEL = 128
    DFF = 512
    NUM_HEADS = 8
    DROPOUT_RATE = 0.1
    VOCAB_SIZE = 20000
    MAX_LEN = 50
    BATCH_SIZE = 64
    EPOCHS = 5

    # 2. Data Preparation
    tokenizer = get_tokenizer(sample_corpus, VOCAB_SIZE, MAX_LEN)
    train_dataset = prepare_dataset(sample_corpus, tokenizer, BATCH_SIZE)

    # 3. Model Initialization
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT_RATE)

    # 4. Optimizer and Loss
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    
    # 5. Compile and Train
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    # The model trains on (context, x) pairs, but for a language model, 
    # the context and x are the same. We will just pass the input for both.
    
    # This is a simplified training loop. In a real-world scenario, you would
    # use callbacks for saving checkpoints, etc.
    
    # Since our dataset is small, this will be very fast.
    # In a real scenario with a large dataset, this would take a long time.
    transformer.fit(train_dataset, epochs=EPOCHS)