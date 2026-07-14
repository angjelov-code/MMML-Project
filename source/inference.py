import tensorflow as tf
import matplotlib.pyplot as plt
import os

import preprocessing
import model
import training

def generate_images(vae, num_generations):
    random_latent_vectors = tf.random.normal(shape=(num_generations, vae.latent_dim))

    generated_images = vae.decoder(random_latent_vectors)

    plt.figure(figsize=(10, 10))
    for i in range(num_generations):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].numpy())
        plt.axis('off')
    plt.suptitle('Generated Images from VAE')
    plt.savefig("../images/Generated.png")
    plt.close()

def reconstruct_images(vae, test_datasets):
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
        plt.savefig("../images/Reconstructed.png")
        plt.close()
        break


def interpolate_images(vae, num_steps, image_path_1, image_path_2):
    # Preprocess the images
    img1 = preprocessing.preprocess_image(image_path_1)
    img2 = preprocessing.preprocess_image(image_path_2)

    # Add a batch dimension to the preprocessed images
    img1_batch = tf.expand_dims(img1, axis=0)
    img2_batch = tf.expand_dims(img2, axis=0)

    # Encode the images to get their latent means and log-variances
    mean1, log_var1 = vae.encoder(img1_batch)
    mean2, log_var2 = vae.encoder(img2_batch)

    # Sample one latent vector for each image (using the reparameterization trick)
    z1 = vae.sample(mean1, log_var1)
    z2 = vae.sample(mean2, log_var2)

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
    plt.savefig("../images/Interpolation.png")
    plt.close()

if __name__ == "__main__":
    data_root = "../../../VAE old/img_align_celeba"
    train_dataset, test_dataset = preprocessing.split_images(data_root)
    vae = model.VAE(128, preprocessing.image_size) #define a model with 128 dim latent space
    
    #training.train_VAE(35, vae, 1e-3, train_dataset, test_dataset) #train for 35 epochs with learning rate 1e-3
    training.load_parameters(vae, test_dataset)
    
    generate_images(vae, 16)
    reconstruct_images(vae, test_dataset)
    interpolate_images(vae, 5, f"{data_root}/012882.jpg", f"{data_root}/013482.jpg")
