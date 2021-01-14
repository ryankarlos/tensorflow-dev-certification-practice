import csv
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

PATH_STANFORD = "data/tmp/stanford_nlp/training_cleaned.csv"
PATH_GLOVE = "data/glove.6B.100d.txt"
SUNSPOT_PATH = "data/tmp/sunspots/daily-min-temperatures.csv"


def get_data_from_csv(filename):
    with open(filename) as training_file:
        labels = []
        images = []
        training_file.readline()
        lines = training_file.readlines()
        for row in lines:
            label = np.array(row.strip().split(",")[0])
            data = np.array(row.strip().split(",")[1:785]).reshape((28, 28))
            labels.append(label)
            images.append(data)

        labels = np.array(labels).astype(float)
        images = np.array(images).astype(float)

    return images, labels


def extract_from_zip(local_zip, base_dir):
    """
    Extracts train and validation dirs from zip files and places in temp dir
    """
    zip_ref = zipfile.ZipFile(local_zip, "r")
    zip_ref.extractall(base_dir)
    zip_ref.close()


def paths_to_train_val_dirs(base_dir):
    """
    defines path variables to train and validation folders in tmp
    labels is list like containing label names as string
    """

    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")

    return train_dir, validation_dir


def read_sunsplot_data_series():
    time_step = []
    temps = []
    with open(SUNSPOT_PATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        step = 0
        for row in reader:
            temps.append(float(row[1]))
            time_step.append(step)
            step = step + 1

    series = np.array(temps)
    time = np.array(time_step)
    return series, time


def train_test_split(series, time, split_time):
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return time_train, x_train, time_valid, x_valid


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def read_stanford_corpus(num_sentences):
    corpus = []
    with open(PATH_STANFORD) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            list_item = []
            list_item.append(row[5])
            this_label = row[0]
            if this_label == "0":
                list_item.append(0)
            else:
                list_item.append(1)
            num_sentences = num_sentences + 1
            corpus.append(list_item)
    return corpus


def split_corpus_into_sentences_labels(corpus, training_size):
    sentences = []
    labels = []
    random.shuffle(corpus)
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])
    return sentences, labels


def tokenise_text_to_sequences(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences, word_index


def pad_sequencs(sequences, max_length, padding_type, trunc_type):
    padded = pad_sequences(
        sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    return padded


def create_glove_embedding_matrix(word_index, embedding_dim):
    # Note this is the 100 dimension version of GloVe from Stanford
    embeddings_index = {}
    with open(PATH_GLOVE) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    vocab_size = len(word_index)
    embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    return embeddings_matrix
