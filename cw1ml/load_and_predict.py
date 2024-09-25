import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

def load_and_predict_model():
    # Load model architecture from JSON
    with open('saved_model_architecture.json', 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)

    # Load model weights
    model.load_weights('saved_model.weights.h5')

    # Define function for perturbing image
    def perturb_image(image, noise_factor):
        noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        return noisy_image

    # Load test data
    (_, _), (test_images, test_labels) = datasets.cifar10.load_data()
    test_images = test_images / 255.0

    # Choose an image
    print("Enter a value from 0 to 9999")
    index = int(input())
    original_image = test_images[index]
    original_label = test_labels[index]

    # Perturb image
    perturbed_image = perturb_image(original_image, 0.1)

    # Predict labels for original and perturbed images
    predictions_original = model.predict(np.expand_dims(original_image, axis=0))
    predictions_perturbed = model.predict(np.expand_dims(perturbed_image, axis=0))

    # Display original and perturbed images along with predicted labels
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title(f'Original Image (Label: {original_label})')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(perturbed_image)
    plt.title('Perturbed Image')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.bar(range(10), tf.nn.softmax(predictions_original[0]))
    plt.title('Original Predictions')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(range(10))
    plt.tight_layout()

    plt.subplot(1, 4, 4)
    plt.bar(range(10), tf.nn.softmax(predictions_perturbed[0]))
    plt.title('Perturbed Predictions')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(range(10))
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    load_and_predict_model()