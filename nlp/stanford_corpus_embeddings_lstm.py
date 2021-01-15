import tensorflow as tf
import numpy as np

from utils.io_preprocessing import (
    read_stanford_corpus,
    split_corpus_into_sentences_labels,
    tokenise_text_to_sequences,
    pad_sequencs,
    create_glove_embedding_matrix,
)

embedding_dim = 100
max_length = 16
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 160000
test_portion = 0.1
num_epochs = 50
num_sentences = 0


def train_test_split(padded, labels):
    split = int(test_portion * training_size)
    test_sequences = padded[0:split]
    training_sequences = padded[split:training_size]
    labels = np.asarray(labels)
    test_labels = labels[0:split]
    training_labels = labels[split:training_size]
    return test_sequences, training_sequences, test_labels, training_labels


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
    sequences, word_index = tokenise_text_to_sequences(sentences)
    padded = pad_sequencs(sequences, max_length, padding_type, trunc_type)
    test_sequences, training_sequences, test_labels, training_labels = train_test_split(
        padded, labels
    )
    embeddings_matrix = create_glove_embedding_matrix(word_index, embedding_dim)
    model = build_lstm_model(embeddings_matrix, word_index)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
        training_sequences,
        training_labels,
        epochs=num_epochs,
        validation_data=(test_sequences, test_labels),
        verbose=2,
    )
    print("Training Complete")
