import tensorflow as tf
from utils.accuracy_loss import plot_acc_loss
from utils.io_preprocessing import (
    read_bbc_news_csv,
    train_test_split_sentences_labels,
    tokenise_text_to_sequences,
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
stopwords = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "nor",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "would",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
]


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
    sentences, labels = read_bbc_news_csv(stopwords)
    (
        val_sentences,
        training_sentences,
        val_labels,
        training_labels,
    ) = train_test_split_sentences_labels(sentences, labels, training_portion)
    (
        train_sequences,
        val_sequences,
        train_label_seq,
        val_label_seq,
    ) = tokenise_text_to_sequences(
        training_sentences,
        val_sentences,
        training_labels,
        val_labels,
        vocab_size,
        oov_tok,
    )
    train_padded, val_padded = padded_sequences(
        train_sequences,
        val_sequences,
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
    weights = model.layers[0].get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)
    plt = plot_acc_loss(history)
    plt.show()
