import os
from utils.io_preprocessing import (
    read_stanford_corpus,
    split_corpus_into_sentences_labels,
    tokenise_text_to_sequences,
    padded_sequences,
    create_glove_embedding_matrix,
    train_test_split_sentences_labels,
)

# this suppresses the logs from tensorflow - needs to be set before tf is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

embedding_dim = 100
max_length = 16
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 160000
training_portion = 0.9
num_epochs = 50
num_sentences = 0


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
    padded = padded_sequences(sequences, max_length, padding_type, trunc_type)
    (
        test_sequences,
        training_sequences,
        test_labels,
        training_labels,
    ) = train_test_split_sentences_labels(padded, labels, training_portion)
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
    """
    4500/4500 - 18s - loss: 0.4242 - accuracy: 0.7974 - val_loss: 0.5130 - val_accuracy: 0.7529
    Epoch 49/50
    4500/4500 - 15s - loss: 0.4251 - accuracy: 0.7970 - val_loss: 0.5137 - val_accuracy: 0.7514
    Epoch 50/50
    4500/4500 - 18s - loss: 0.4239 - accuracy: 0.7993 - val_loss: 0.5087 - val_accuracy: 0.7553
    """
    print("Training Complete")
