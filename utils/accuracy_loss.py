from matplotlib import pyplot as plt


def evaluate_model(model, images, labels):
    """
    Evaluates the trained model on the test images and labels
    """
    model.evaluate(images, labels, verbose=0)


def compute_loss(history):
    """
    Computes training and val loss and accuracy
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    return acc, loss, val_acc, val_loss


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
