from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import RMSprop
from utils.callback import myCallback
from utils.augmentation import (
    train_val_generator_flow_from_dir,
    data_generator_with_augmentation,
)
from utils.io_preprocessing import paths_to_train_val_dirs, extract_from_zip
from utils.plotting import plot_acc_loss

PATH_INCEPTION = f"data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


def load_pretrained_model_inception():
    """
    Create an instance of the inception model from the local pre-trained weights
    """

    local_weights_file = PATH_INCEPTION

    model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)

    model.load_weights(local_weights_file)

    # Make all the layers in the pre-trained model non-trainable
    for layer in model.layers:
        layer.trainable = False

    return model


def build_model_with_transfer_learning(pre_trained_model):
    """
    # Expected output will be large. Last few lines should be:

    # mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_248[0][0]
    #                                                                  activation_251[0][0]
    #                                                                  activation_256[0][0]
    #                                                                  activation_257[0][0]
    # __________________________________________________________________________________________________
    # flatten_4 (Flatten)             (None, 37632)        0           mixed7[0][0]
    # __________________________________________________________________________________________________
    # dense_8 (Dense)                 (None, 1024)         38536192    flatten_4[0][0]
    # __________________________________________________________________________________________________
    # dropout_4 (Dropout)             (None, 1024)         0           dense_8[0][0]
    # __________________________________________________________________________________________________
    # dense_9 (Dense)                 (None, 1)            1025        dropout_4[0][0]
    # ==================================================================================================
    # Total params: 47,512,481
    # Trainable params: 38,537,217
    # Non-trainable params: 8,975,264
    """

    last_layer = pre_trained_model.get_layer("mixed6")
    print("last layer output shape: ", last_layer.output.shape)
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation="sigmoid")(x)
    model = Model(pre_trained_model.inputs, x)
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    return model


if __name__ == "__main__":
    local_zip = "data/cats_and_dogs_filtered.zip"
    extract_from_zip(local_zip, base_dir="data/tmp")
    pre_trained_model = load_pretrained_model_inception()
    model = build_model_with_transfer_learning(pre_trained_model)
    train_dir, validation_dir = paths_to_train_val_dirs(
        base_dir="data/tmp/cats_and_dogs_filtered"
    )
    train_generator, validation_generator = train_val_generator_flow_from_dir(
        train_dir,
        validation_dir,
        data_generator_with_augmentation(),
        data_generator_with_augmentation(),
    )
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=10,
        epochs=5,
        validation_steps=5,
        verbose=2,
    )
    plt = plot_acc_loss(history)
    plt.show()
