from matplotlib import pyplot as plt
from utils.accuracy_loss import compute_loss


def plot_acc_loss(history):
    """
    Plots the chart for accuracy and loss on both training and validation
    """

    acc, loss, val_acc, val_loss = compute_loss(history)
    epochs = range(len(acc))
    plt.plot(epochs, acc, "r", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training and validation loss")
    plt.legend()

    return plt


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def plot_lr_schedule(history):
    plt.figure(figsize=(10, 6))
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0, 30])
    plt.show()
