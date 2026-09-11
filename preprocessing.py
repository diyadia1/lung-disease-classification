import cv2
import numpy as np


def preprocess_image(image_path, image_size=(224, 224)):
    """
    Load and preprocess a chest X-ray image.

    Parameters:
        image_path: Path to the X-ray image.
        image_size: Target image dimensions.

    Returns:
        Preprocessed image as a NumPy array.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    image = cv2.resize(image, image_size)

    # Normalize pixel values to the range 0–1
    image = image.astype(np.float32) / 255.0

    # Add channel dimension for model input
    image = np.expand_dims(image, axis=-1)

    return image


if __name__ == "__main__":
    print("Chest X-ray preprocessing module ready.")
