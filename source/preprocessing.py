import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
import glob

image_size = 64
batch_size = 128

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Get original image dimensions
    shape = tf.shape(image)
    original_height = tf.cast(shape[0], tf.float32)
    original_width = tf.cast(shape[1], tf.float32)
    
    # Calculate crop dimensions for 178x178 center crop
    crop_size = 178
    offset_height = tf.cast((original_height - crop_size) / 2, tf.int32)
    offset_width = tf.cast((original_width - crop_size) / 2, tf.int32)
    
    # Center crop to 178x178
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)

    # Resize to IMG_HEIGHT x IMG_WIDTH (64x64)
    image = tf.image.resize(image, [image_size, image_size])
    image = image / 255.0  # Normalize to [0, 1]
    return image

def split_images(data_root):
    # Get a list of all image file paths
    image_paths = [os.path.join(data_root, fname) for fname in os.listdir(data_root) if fname.endswith('.jpg')]

    # Shuffle the image paths for a random split
    random.seed(42) # for reproducibility
    random.shuffle(image_paths)

    # Split data into training and testing sets (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(image_paths))
    train_image_paths = image_paths[:train_size]
    test_image_paths = image_paths[train_size:]

    # Create tf.data.Dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create tf.data.Dataset for testing
    test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset
