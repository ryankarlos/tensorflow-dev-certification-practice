from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def data_generator_with_augmentation():
    """
    Create an ImageDataGenerator and do Image Augmentation
    """

    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def data_gen_no_augmentation():
    """
    For use with validation data which does not need to be augmented
    """

    return ImageDataGenerator(rescale=1 / 0.255)


def train_val_generator_with_flow(
    training_images,
    training_labels,
    testing_images,
    testing_labels,
    train_datagen,
    test_datagen,
):
    """
    uses the flow method of ImageDataGenerator to generate batches of tensor image data
    with real-time data augmentation. requires input images to be in np array format
    """
    train_generator = train_datagen.flow(
        training_images, training_labels, batch_size=10
    )
    validation_generator = test_datagen.flow(
        testing_images, testing_labels, batch_size=10
    )

    return train_generator, validation_generator


def train_val_generator_flow_from_dir(
    train_dir,
    validation_dir,
    train_datagen,
    test_datagen,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary",
):
    """
    uses the flow _from_directory method to generate batches of tensor image data
    with real-time data augmentation from training and validation directories containing raw
    images.

    """
    # Flow training and val images in batches from dirs using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode
    )

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
    )

    return train_generator, validation_generator
