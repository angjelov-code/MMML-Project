import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

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

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Compile the VAE model (though we'll use a custom training loop)
# For simplicity, we can define a custom train step if not compiling with fit()
@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        reconstructed_images, mean, log_variance = vae(images)
        loss = vae_loss(images, reconstructed_images, mean, log_variance)

    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

print("VAE loss function and optimizer defined.")

EPOCHS = 35
BEST_LOSS = float('inf')
CHECKPOINT_DIR = './vae_checkpoints'

# Create directory if it doesn't exist
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if should_train:
    for epoch in range(EPOCHS):
        total_train_loss = 0
        num_train_batches = 0
        for batch_num, image_batch in enumerate(train_dataset): # Use train_dataset for training
            loss = train_step(image_batch)
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
        if avg_test_loss < BEST_LOSS:
            BEST_LOSS = avg_test_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'vae_weights_best_loss_{BEST_LOSS:.4f}.weights.h5')
            vae.save_weights(checkpoint_path)
            print(f"Saved best model weights to {checkpoint_path} with test loss: {BEST_LOSS:.4f}")

    print("VAE Training Complete.")
else:
    saved_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, 'vae_weights_best_loss_*.h5'))

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

num_generations = 16
random_latent_vectors = tf.random.normal(shape=(num_generations, latent_dim))

generated_images = vae.decoder(random_latent_vectors)