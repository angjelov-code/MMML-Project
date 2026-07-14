import tensorflow as tf
import matplotlib.pyplot as plt

import preprocessing
import model
import training

def generate_images(vae, num_generations, filename="Generation.png"):
    """
    Generates num_generations number of images from random latent vectors and saves them to a file in the images folder
    """
    random_latent_vectors = tf.random.normal(shape=(num_generations, vae.latent_dim))

    generated_images = vae.decoder(random_latent_vectors)

    plt.figure(figsize=(10, 10))
    for i in range(num_generations):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].numpy())
        plt.axis("off")
    plt.suptitle("Generated Images from VAE")
    plt.savefig(f"../images/{filename}")
    plt.close()

def reconstruct_images(vae, test_datasets, num_display, filename="Reconstruction.png"):
    """
    Reconstructs images from the test dataset and saves num_display of them to a file in the images folder
    """
    for test_images in test_dataset.take(1):
        reconstructed_images, _, _ = vae(test_images)
        
        plt.figure(figsize=(15, 5))
        for i in range(num_display):
            # Original Image
            plt.subplot(2, num_display, i + 1)
            plt.imshow(test_images[i].numpy())
            plt.title("Original")
            plt.axis("off")

            # Reconstructed Image
            plt.subplot(2, num_display, i + 1 + num_display)
            plt.imshow(reconstructed_images[i].numpy())
            plt.title("Reconstructed")
            plt.axis("off")
        plt.suptitle("Original vs. Reconstructed Images from Test Set", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"../images/{filename}")
        plt.close()
        break


def interpolate_images(vae, num_steps, data_root, image_name_1, image_name_2, filename="Interpolation.png"):
    """
    Linear Interpolation two images from the dataset with num_steps steps and saves the result to a file in the images folder
    """
    img1 = preprocessing.preprocess_image(f"{data_root}/{image_name_1}")
    img2 = preprocessing.preprocess_image(f"{data_root}/{image_name_2}")

    
    img1_batch = tf.expand_dims(img1, axis=0)
    img2_batch = tf.expand_dims(img2, axis=0)

    
    mean1, log_var1 = vae.encoder(img1_batch)
    mean2, log_var2 = vae.encoder(img2_batch)

    
    z1 = vae.sample(mean1, log_var1)
    z2 = vae.sample(mean2, log_var2)

    plt.figure(figsize=(num_steps * 2, 4))

    
    plt.subplot(2, num_steps + 2, 1)
    plt.imshow(img1.numpy())
    plt.title("Original 1")
    plt.axis("off")

    
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        interpolated_z = z1 * (1 - alpha) + z2 * alpha
        
        generated_image = vae.decoder(interpolated_z)
        
        plt.subplot(2, num_steps + 2, i + 2)
        plt.imshow(generated_image[0].numpy())
        plt.title(f"Step {i+1}")
        plt.axis("off")

    
    plt.subplot(2, num_steps + 2, num_steps + 2)
    plt.imshow(img2.numpy())
    plt.title("Original 2")
    plt.axis("off")

    plt.suptitle(f"Interpolation between {image_name_1} and {image_name_2}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"../images/{filename}")
    plt.close()

if __name__ == "__main__":
    data_root = "../../../datasets/img_align_celeba"
    train_dataset, test_dataset = preprocessing.get_datasets(data_root)
    vae = model.VAE(128, preprocessing.image_size) #define a model with 128 dim latent space
    
    #training.train_VAE(35, vae, 1e-3, train_dataset, test_dataset) #train for 35 epochs with learning rate 1e-3
    vae.load_best()
    
    generate_images(vae, 16)
    reconstruct_images(vae, test_dataset, 8)
    interpolate_images(vae, 5, data_root, "012882.jpg", "013482.jpg")
