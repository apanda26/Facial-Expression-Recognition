import tensorflow as tf

def load_dataset(train_dir, test_dir, batch_size=64, img_size=(48, 48)):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )

    return train_generator, test_generator
