class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_net = tf.keras.Sequential([
            layers.Conv2D(32, 5, padding='same', input_shape=(image_size, image_size, 3)), # Changed strides from 2 to 1 (default)
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D((2, 2), padding='same'), # Added MaxPooling
            layers.Conv2D(64, 4, padding='same'), # Changed strides from 2 to 1 (default)
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D((2, 2), padding='same'), # Added MaxPooling
            layers.Conv2D(128, 3, padding='same'), # Changed strides from 2 to 1 (default)
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D((2, 2), padding='same'), # Added MaxPooling, results in 8x8x128 feature map
            layers.Flatten(), # 8x8x128 -> 8192
            layers.Dense(latent_dim * 4), # New intermediate Dense layer
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(2 * latent_dim) # For mean and log_variance
        ])

    def call(self, inputs):
        z_params = self.encoder_net(inputs)
        mean, log_variance = tf.split(z_params, num_or_size_splits=2, axis=1)
        return mean, log_variance

class Decoder(Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)), # This InputLayer is fine as its followed by a Dense layer
            layers.Dense(latent_dim * 4), # First Dense layer in decoder
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(8 * 8 * 128), # Second Dense layer, matches flattened output of encoder
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((8, 8, 128)), # Reshape to 8x8x128
            layers.Conv2DTranspose(64, 3, strides=2, padding='same'), # 8x8x128 -> 16x16x64
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(32, 4, strides=2, padding='same'), # 16x16x64 -> 32x32x32
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(3, 5, activation='sigmoid', strides=2, padding='same') # 32x32x32 -> 64x64x3
        ])

    def call(self, inputs):
        return self.decoder_net(inputs)

class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def sample(self, mean, log_variance):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_variance * 0.5) + mean

    def call(self, inputs):
        mean, log_variance = self.encoder(inputs)
        z = self.sample(mean, log_variance)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mean, log_variance