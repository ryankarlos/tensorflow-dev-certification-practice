import tensorflow as tf
from utils.plotting import plot_acc_loss
from utils.io_preprocessing import (
    read_bbc_news_csv,
    train_test_split_sentences_labels,
    tokenise_sentence_to_sequence,
    tokenise_labels_to_sequences,
    padded_sequences,
)

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_portion = 0.8
num_epochs = 30
sentences = []
labels = []


def model_build(vocab_size, embedding_dim, input_length):
    """
    # Expected Output
    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # embedding (Embedding)        (None, 120, 16)           16000
    # _________________________________________________________________
    # global_average_pooling1d (Gl (None, 16)                0
    # _________________________________________________________________
    # dense (Dense)                (None, 24)                408
    # _________________________________________________________________
    # dense_1 (Dense)              (None, 6)                 150
    # =================================================================
    # Total params: 16,558
    # Trainable params: 16,558
    # Non-trainable params: 0
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, input_length=input_length
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )
    model.summary()
    return model


def model_compile(model, loss, optimizer, metrics):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


if __name__ == "__main__":
    sentences, labels = read_bbc_news_csv()
    (
        val_sentences,
        training_sentences,
        val_labels,
        training_labels,
    ) = train_test_split_sentences_labels(sentences, labels, training_portion)

    train_sequences, val_sequences, word_index = tokenise_sentence_to_sequence(
        training_sentences, vocab_size, oov_tok, val_sentences
    )
    train_label_seq, val_label_seq = tokenise_labels_to_sequences(
        training_labels, val_labels
    )

    train_padded, val_padded = padded_sequences(
        train_sequences,
        val_sequences=val_sequences,
        max_length=max_length,
        padding_type=padding_type,
        trunc_type=trunc_type,
    )

    model = model_build(vocab_size, embedding_dim, max_length)
    model_compile(
        model,
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_padded,
        train_label_seq,
        epochs=num_epochs,
        validation_data=(val_padded, val_label_seq),
        verbose=2,
    )
    """
    Epoch 28/30
    56/56 - 0s - loss: 0.0410 - accuracy: 0.9955 - val_loss: 0.2151 - val_accuracy: 0.9483
    Epoch 29/30
    56/56 - 0s - loss: 0.0371 - accuracy: 0.9966 - val_loss: 0.2149 - val_accuracy: 0.9483
    Epoch 30/30
    56/56 - 0s - loss: 0.0337 - accuracy: 0.9972 - val_loss: 0.2109 - val_accuracy: 0.9528
    """
    weights = model.layers[0].get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)
    plt = plot_acc_loss(history)
    plt.show()
