import os 
import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path and convert it to a numpy array.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        np.ndarray: The image as a numpy array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    with Image.open(image_path) as img:
        img_array = np.array(img)
    
    return img_array

def convert_rgb_to_custom_grayscale_green_red_blue(image_array):
    """
    Convert an RGB image to grayscale where:
    - Green maps to 0.0
    - Red maps to 0.5
    - Blue maps to 1.0
    
    Args:
        image_array (np.ndarray): RGB image (H, W, 3) in uint8 format.
        
    Returns:
        np.ndarray: Grayscale image (H, W) in uint8 format.
    """
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3)")
    
    # Define anchor points
    green = np.array([0, 255, 0], dtype=np.float32)
    red   = np.array([255, 0, 0], dtype=np.float32)
    blue  = np.array([0, 0, 255], dtype=np.float32)

    img = image_array.astype(np.float32)
    h, w, _ = img.shape
    flat_img = img.reshape(-1, 3)

    # Vectors for each segment
    gr_vec = red - green  # green → red
    rb_vec = blue - red   # red → blue

    # Empty array for grayscale
    grayscale = np.zeros(flat_img.shape[0], dtype=np.float32)

    for i, color in enumerate(flat_img):
        if np.allclose(color, green):
            grayscale[i] = 0.0
            continue
        elif np.allclose(color, blue):
            grayscale[i] = 1.0
            continue

        # Vector from green to current pixel
        if np.linalg.norm(color - green) < np.linalg.norm(blue - color):
            # In green → red segment
            t = np.dot(color - green, gr_vec) / (np.linalg.norm(gr_vec) ** 2)
            t = np.clip(t, 0, 1)  # Clamp between 0 and 1
            grayscale[i] = t * 0.5  # Scale to [0, 0.5]
        else:
            # In red → blue segment
            t = np.dot(color - red, rb_vec) / (np.linalg.norm(rb_vec) ** 2)
            t = np.clip(t, 0, 1)
            grayscale[i] = 0.5 + t * 0.5  # Scale to [0.5, 1.0]

    # Reshape and scale to [0, 255]
    grayscale = (grayscale * 255).astype(np.uint8).reshape(h, w)
    return grayscale

def compare_predicted_and_target_images(predicted_image_path, target_image_path):
    """
    Compare a predicted image with a target image by converting both to grayscale.
    
    Args:
        predicted_image_path (str): The path to the predicted image file.
        target_image_path (str): The path to the target image file.
        
    Returns:
        float: The Mean Squared Error (MSE) loss between the predicted and target images.
    """
    predicted_image = load_image(predicted_image_path)
    target_image = load_image(target_image_path)
    
    predicted_gray = convert_rgb_to_custom_grayscale_green_red_blue(predicted_image)
    
    # for each pixel in the predicted image, do an MSE comparison with the target image (which is greyscale already)
    mse_loss = np.zeros(predicted_gray.shape)

    for i in range(predicted_gray.shape[0]):
        for j in range(predicted_gray.shape[1]):
            mse_loss[i, j] = (predicted_gray[i, j] - target_image[i, j]) ** 2

    mse_loss = np.mean(mse_loss)
    
    print(f"MSE Loss between predicted and target images: {mse_loss}")
    return mse_loss

