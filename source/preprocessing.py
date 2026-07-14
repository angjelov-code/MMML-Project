import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

image_size = 64

def preprocess_image(image_path):
    """
    Crops the image to 178px, centers it a crops again to image_size
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    shape = tf.shape(image)
    original_height = tf.cast(shape[0], tf.float32)
    original_width = tf.cast(shape[1], tf.float32)
    
    crop_size = 178
    offset_height = tf.cast((original_height - crop_size) / 2, tf.int32)
    offset_width = tf.cast((original_width - crop_size) / 2, tf.int32)
    
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)

    image = tf.image.resize(image, [image_size, image_size])
    image = image / 255.0  # Normalize to [0, 1]
    return image

def get_datasets(data_root, train_size=0.8, batch_size=128):
    """
    Splits the data into a train and test set, preprocesses them and returns them into batches
    """
    image_paths = [os.path.join(data_root, fname) for fname in os.listdir(data_root) if fname.endswith(".jpg")]

    random.seed(42) # for reproducibility
    random.shuffle(image_paths)

    train_size = int(train_size * len(image_paths))
    train_image_paths = image_paths[:train_size]
    test_image_paths = image_paths[train_size:]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset
