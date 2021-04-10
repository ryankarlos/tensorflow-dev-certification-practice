import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import plot_series
from utils.callback import callback_lrschedule, callback_earlystopping
from utils.io_preprocessing import (
    read_sunsplot_data_series,
    train_test_split_series_time,
    windowed_dataset,
)

window_size = 60
batch_size = 100
shuffle_buffer_size = 1000
split_time = 2500
epochs = 150


def build_rnn_timeseries_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=60,
                kernel_size=5,
                strides=1,
                padding="causal",
                activation="relu",
                input_shape=[None, 1],
            ),
            tf.keras.layers.LSTM(60, return_sequences=True),
            tf.keras.layers.LSTM(60, return_sequences=True),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 400),
        ]
    )


def compile_model(model):
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


if __name__ == "__main__":
    series, time = read_sunsplot_data_series()
    time_train, x_train, time_valid, x_valid = train_test_split_series_time(
        series, time, split_time
    )
    plt.figure(figsize=(10, 6))
    plot_series(time, series)

    lr_schedule = callback_lrschedule()

    train_set = windowed_dataset(
        x_train,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer_size,
    )
    print(train_set)
    print(x_train.shape)

    model = build_rnn_timeseries_model()
    compile_model(model)
    history = model.fit(train_set, epochs=epochs, callbacks=[lr_schedule])

    """
        Epoch 145/150
        25/25 [==============================] - 2s 77ms/step - loss: 1.5136 - mae: 1.9549
        Epoch 146/150
        25/25 [==============================] - 2s 77ms/step - loss: 1.5255 - mae: 1.9675
        Epoch 147/150
        25/25 [==============================] - 2s 77ms/step - loss: 1.5217 - mae: 1.9632
        Epoch 148/150
        25/25 [==============================] - 2s 77ms/step - loss: 1.5175 - mae: 1.9590
        Epoch 149/150
        25/25 [==============================] - 2s 78ms/step - loss: 1.5114 - mae: 1.9526
        Epoch 150/150
        25/25 [==============================] - 2s 77ms/step - loss: 1.5149 - mae: 1.9560
    """

    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size : -1, -1, 0]
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, rnn_forecast)

    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    print(f"Mean Absolute Error: {mae}")
