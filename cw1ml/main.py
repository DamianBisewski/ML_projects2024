import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import pickle

def train_and_save_model():
    # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # Save model architecture as JSON
    model_json = model.to_json()
    with open('saved_model_architecture.json', 'w') as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights('saved_model.weights.h5')

if __name__ == "__main__":
    train_and_save_model()