import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    """
    Define a Callback class that stops training once accuracy reaches a specified threshold
    """

    def on_epoch_end(self, epoch, logs={}):
        if logs.get("acc") > 0.97:
            print(f"\nReached 97% accuracy so cancelling training!")
            self.model.stop_training = True


def callback_lrschedule():
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20)
    )
    return lr_schedule
