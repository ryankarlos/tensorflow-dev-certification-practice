import tensorflow as tf
from utils.time_series_components import create_time_series_with_noise
from utils.io_preprocessing import windowed_dataset
from utils.plotting import plot_series, plot_lr_schedule
import numpy as np
from utils.callback import callback_lrschedule


SPLIT_TIME = 3000
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
learning_rate = 1e-5
momentum = 0.9
epochs = 3


def train_test_split(time, series):
    time_train = time[:SPLIT_TIME]
    x_train = series[:SPLIT_TIME]
    time_test = time[SPLIT_TIME:]
    x_test = series[SPLIT_TIME:]
    return time_train, x_train, time_test, x_test


def LSTM_model_ts():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 100.0),
        ]
    )

    return model


def optimiser(learning_rate, momentum):
    return tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)


def model_compile(model, optimizer: tf.keras.optimizers, *metrics):
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=metrics)
    return model


def model_fit(model, epochs: int, callbacks=None):
    if callbacks is not None:
        return model.fit(dataset, epochs=epochs, callbacks=[callbacks])
    else:
        return model.fit(dataset, epochs=epochs)


def forecast_results(series, window_size, model):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time : time + window_size][np.newaxis]))
    forecast = forecast[SPLIT_TIME - window_size :]
    return np.array(forecast)[:, 0, 0]


if __name__ == "__main__":
    time = np.arange(10 * 365 + 1, dtype="float32")
    series = create_time_series_with_noise(time)
    # plot_series(time, series)
    time_train, x_train, time_test, x_test = train_test_split(time, series)
    dataset = windowed_dataset(
        x_train,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer_size,
    )
    metrics = ["mae"]
    model = model_compile(
        LSTM_model_ts(),
        optimiser(learning_rate=learning_rate, momentum=momentum),
        *metrics
    )
    history = model_fit(model, epochs=epochs, callbacks=callback_lrschedule())
    plot_lr_schedule(history)
    results = forecast_results(series, window_size, model)
    plot_series(time_test, results)
