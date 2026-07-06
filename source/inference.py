num_generations = 16
random_latent_vectors = tf.random.normal(shape=(num_generations, latent_dim))

generated_images = vae.decoder(random_latent_vectors)

plt.figure(figsize=(10, 10))
for i in range(num_generations):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i].numpy())
    plt.axis('off')
plt.suptitle('Generated Images from VAE')
plt.savefig("Generated.png")
plt.close()
print("Generated images")

for test_images in test_dataset.take(1):
    # Get reconstructions from the VAE
    reconstructed_images, _, _ = vae(test_images)

    # Display original and reconstructed images
    num_display = min(8, test_images.shape[0]) # Display up to 8 images
    
    plt.figure(figsize=(15, 5))
    for i in range(num_display):
        # Original Image
        plt.subplot(2, num_display, i + 1)
        plt.imshow(test_images[i].numpy())
        plt.title('Original')
        plt.axis('off')

        # Reconstructed Image
        plt.subplot(2, num_display, i + 1 + num_display)
        plt.imshow(reconstructed_images[i].numpy())
        plt.title('Reconstructed')
        plt.axis('off')
    plt.suptitle('Original vs. Reconstructed Images from Test Set', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig("Reconstructed.png")
    plt.close()
    print("Reconstructed images")
    break

image_path_1 = os.path.join(data_root, '136115.jpg')
image_path_2 = os.path.join(data_root, '200575.jpg')

# Preprocess the images
img1 = preprocess_image(image_path_1)
img2 = preprocess_image(image_path_2)

# Add a batch dimension to the preprocessed images
img1_batch = tf.expand_dims(img1, axis=0)
img2_batch = tf.expand_dims(img2, axis=0)

# Encode the images to get their latent means and log-variances
mean1, log_var1 = vae.encoder(img1_batch)
mean2, log_var2 = vae.encoder(img2_batch)

# Sample one latent vector for each image (using the reparameterization trick)
z1 = vae.sample(mean1, log_var1)
z2 = vae.sample(mean2, log_var2)

# Number of interpolation steps
num_steps = 5

plt.figure(figsize=(num_steps * 2, 4))

# Display the first original image
plt.subplot(2, num_steps + 2, 1)
plt.imshow(img1.numpy())
plt.title('Original 1')
plt.axis('off')

# Perform interpolation in the latent space
for i in range(num_steps):
    alpha = i / (num_steps - 1)
    interpolated_z = z1 * (1 - alpha) + z2 * alpha
    
    # Decode the interpolated latent vector
    generated_image = vae.decoder(interpolated_z)
    
    plt.subplot(2, num_steps + 2, i + 2)
    plt.imshow(generated_image[0].numpy())
    plt.title(f'Step {i+1}')
    plt.axis('off')

# Display the second original image
plt.subplot(2, num_steps + 2, num_steps + 2)
plt.imshow(img2.numpy())
plt.title('Original 2')
plt.axis('off')

plt.suptitle(f'Interpolation between {os.path.basename(image_path_1)} and {os.path.basename(image_path_2)}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Interpolation.png")
plt.close()
print("Interpolation")