def evaluate_model(model, images, labels):
    """
    Evaluates the trained model on the test images and labels
    """
    model.evaluate(images, labels, verbose=0)


def compute_loss(history):
    """
    Computes training and val loss and accuracy
    """
    acc = history.history.get("acc", history.history["accuracy"])
    val_acc = history.history.get("val_acc", history.history["val_accuracy"])
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    return acc, loss, val_acc, val_loss
