import os
import tensorflow as tf
import glob
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

import model

checkpoint_dir = '../parameters'

# Define the VAE loss function
def vae_loss(x, x_recon, mean, log_variance):
    # Reconstruction loss (Binary Cross-Entropy for image data normalized to [0,1])
    # The error indicates binary_crossentropy is already reducing the channel dimension,
    # so we only sum over height and width to get per-sample loss.
    reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_recon), axis=[1, 2])
    
    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1)
    
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    return total_loss

@tf.function
def train_step(images, vae, optimizer):
    with tf.GradientTape() as tape:
        reconstructed_images, mean, log_variance = vae(images)
        loss = vae_loss(images, reconstructed_images, mean, log_variance)

    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

def train_VAE(epochs, vae, learning_rate, train_dataset, test_dataset):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_train_loss = 0
        num_train_batches = 0
        for batch_num, image_batch in enumerate(train_dataset): # Use train_dataset for training
            loss = train_step(image_batch, vae, optimizer)
            total_train_loss += loss
            num_train_batches += 1
            if batch_num % 100 == 0: # Print loss every 100 batches
                print(f"Epoch {epoch+1}, Batch {batch_num}, Train Loss: {loss:.4f}")

        avg_train_loss = total_train_loss / num_train_batches

        # Calculate validation loss on the test set
        total_test_loss = 0
        num_test_batches = 0
        for image_batch in test_dataset:
            reconstructed_images, mean, log_variance = vae(image_batch)
            test_loss = vae_loss(image_batch, reconstructed_images, mean, log_variance)
            total_test_loss += test_loss
            num_test_batches += 1
        avg_test_loss = total_test_loss / num_test_batches

        print(f"\nEpoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}, Average Test Loss: {avg_test_loss:.4f}\n")

        # Save best model parameters
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'vae_weights_loss_{best_loss:.4f}.weights.h5')
            vae.save_weights(checkpoint_path)
            print(f"Saved best model weights to {checkpoint_path} with test loss: {best_loss:.4f}")

    print("VAE Training Complete.")

def load_parameters(vae, test_dataset):
    saved_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'vae_weights_loss_*.h5'))

    if saved_checkpoints:
        # Get the latest (or 'best' if sorted by loss in filename) checkpoint
        latest_checkpoint = sorted(saved_checkpoints)[-1] # Assumes last in sorted list is 'best' or latest
        print(f"Loading weights from: {latest_checkpoint}")
        for sample_batch in test_dataset.take(1):
            _ = vae(sample_batch) # Pass a sample batch through the model to build its layers
            break
        vae.load_weights(latest_checkpoint)
        print("VAE weights loaded successfully.")
    else:
        print("No saved VAE weights found in the checkpoint directory.")
