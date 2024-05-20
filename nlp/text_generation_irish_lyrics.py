import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def tokenise_dataset(corpus):
    # Initialize the Tokenizer class
    tokenizer = Tokenizer()
    # Generate the word index dictionary
    tokenizer.fit_on_texts(corpus)
    return tokenizer


def initialise_input_sequences_list(corpus, fitted_tokenizer):
    # Initialize the sequences list
    input_sequences = []

    # Loop over every line
    for line in corpus:

        # Tokenize the current line
        token_list = fitted_tokenizer.texts_to_sequences([line])[0]

        # Loop over the line several times to generate the subphrases
        for i in range(1, len(token_list)):
            # Generate the subphrase
            n_gram_sequence = token_list[: i + 1]

            # Append the subphrase to the sequences list
            input_sequences.append(n_gram_sequence)
    return input_sequences


def pad_sequences_and_split_labels(input_sequences, max_sequence_len, total_words):

    # Pad all sequences
    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )

    # Create inputs and label by splitting the last token in the subphrases
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

    # Convert the label into one-hot arrays
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    return xs, ys


def build_train_model(xs, ys, total_words, max_sequence_len):
    # Build the model
    model = Sequential([
        Embedding(total_words, 64, input_length=max_sequence_len-1),
        Bidirectional(LSTM(20)),
        Dense(total_words, activation='softmax')
    ])

    # Use categorical crossentropy because this is a multi-class problem
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit(xs, ys, epochs=500)
    return history, model


# Plot utility
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


def generate_text(model, seed_text, next_words):
    """
    With the model trained, you can now use it to make its own song! The process would look like:

        Feed a seed text to initiate the process.
        Model predicts the index of the most probable next word.
        Look up the index in the reverse word index dictionary
        Append the next word to the seed text.
        Feed the result to the model again.
    """

    # Loop until desired length is reached
    for _ in range(next_words):

        # Convert the seed text to a token sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        # Pad the sequence
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )

        # Feed to the model and get the probabilities for each index
        probabilities = model.predict(token_list)

        # Get the index with the highest probability
        predicted = np.argmax(probabilities, axis=-1)[0]

        # Ignore if index is 0 because that is just the padding.
        if predicted != 0:
            # Look up the word associated with the index.
            output_word = tokenizer.index_word[predicted]

            # Combine with the seed text
            seed_text += " " + output_word

    return seed_text


if __name__ == "__main__":
    # Define the lyrics of the song
    data = (
        "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis "
        "father died and made him a man again \n Left him a farm and ten acres of ground. \nHe "
        "gave a grand party for friends and relations \nWho didnt forget him when come to the "
        "wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions "
        "of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and "
        "boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round "
        "merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink "
        "for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans "
        "Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was "
        "bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing "
        "away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old "
        "hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey "
        "were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we "
        "banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls "
        "got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks "
        "Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long "
        "weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans "
        "Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped "
        "out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls "
        "they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance "
        "McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled "
        "for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans "
        "Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was "
        "painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from "
        "under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim "
        "McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, "
        "bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
    )
    corpus = data.lower().split("\n")
    tokenizer = tokenise_dataset(corpus)
    # Define the total words. You add 1 for the index `0` which is just the padding token.
    total_words = len(tokenizer.word_index) + 1
    print(f"word index dictionary: {tokenizer.word_index}")
    print(f"total words: {total_words}")

    input_sequences = initialise_input_sequences_list(corpus, tokenizer)
    # Get the length of the longest line
    max_sequence_len = max([len(x) for x in input_sequences])
    print(max_sequence_len)
    xs, ys = pad_sequences_and_split_labels(input_sequences, max_sequence_len, total_words)
    history, model = build_train_model(xs, ys, total_words, max_sequence_len)
    # Visualize the accuracy
    plot_graphs(history, "accuracy")

    # Define seed text
    seed_text = "Laurence went to Dublin"
    # Define total words to predict
    next_words = 100
    generate_text(model, seed_text, next_words)
