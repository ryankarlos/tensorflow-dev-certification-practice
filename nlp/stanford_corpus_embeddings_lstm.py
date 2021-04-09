import os
from utils.io_preprocessing import (
    read_stanford_corpus,
    split_corpus_into_sentences_labels,
    tokenise_sentence_to_sequence,
    padded_sequences,
    create_glove_embedding_matrix,
    train_test_split_sentences_labels,
)
import numpy as np

# this suppresses the logs from tensorflow - needs to be set before tf is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

embedding_dim = 100
max_length = 16
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 106000
training_portion = 0.9
num_epochs = 14
num_sentences = 0
vocab_size = 1000
batch_size = 1000
steps_per_epoch = (training_size * 0.9) / batch_size


def build_lstm_model(embeddings_matrix, word_index):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                len(word_index) + 1,
                embedding_dim,
                input_length=max_length,
                weights=[embeddings_matrix],
                trainable=False,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 5, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


if __name__ == "__main__":

    corpus = read_stanford_corpus(num_sentences)
    sentences, labels = split_corpus_into_sentences_labels(corpus, training_size)
    (
        val_sentences,
        training_sentences,
        val_labels,
        training_labels,
    ) = train_test_split_sentences_labels(sentences, labels, training_portion)

    training_sequences, test_sequences, word_index = tokenise_sentence_to_sequence(
        training_sentences, vocab_size, oov_tok, val_sentences=val_sentences
    )

    train_padded, val_padded = padded_sequences(
        training_sequences,
        val_sequences=test_sequences,
        max_length=max_length,
        padding_type=padding_type,
        trunc_type=trunc_type,
    )

    embeddings_matrix = create_glove_embedding_matrix(word_index, embedding_dim)
    model = build_lstm_model(embeddings_matrix, word_index)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
        train_padded,
        np.array(training_labels, dtype=np.float),
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(val_padded, np.array(val_labels, dtype=np.float)),
        verbose=2,
    )
    """
    Epoch 1/14
    95/95 - 6s - loss: 0.6343 - accuracy: 0.6364 - val_loss: 0.5811 - val_accuracy: 0.6910
    Epoch 2/14
    95/95 - 4s - loss: 0.5813 - accuracy: 0.6875 - val_loss: 0.5529 - val_accuracy: 0.7106
    Epoch 3/14
    95/95 - 4s - loss: 0.5599 - accuracy: 0.7066 - val_loss: 0.5471 - val_accuracy: 0.7186
    Epoch 4/14
    95/95 - 4s - loss: 0.5487 - accuracy: 0.7156 - val_loss: 0.5348 - val_accuracy: 0.7263
    Epoch 5/14
    95/95 - 5s - loss: 0.5418 - accuracy: 0.7210 - val_loss: 0.5302 - val_accuracy: 0.7304
    Epoch 6/14
    95/95 - 5s - loss: 0.5317 - accuracy: 0.7285 - val_loss: 0.5265 - val_accuracy: 0.7327
    """
    print("Training Complete")
