# Generating Digits using an AutoEncoder

## Creating A Variational AutoEncoder

### Import Necessary Libraries


```python
import tensorflow as tf
import tensorflow.keras.backend as K
```

### Define Paramaters And A Sampling Layer


```python
num_units_hidden_1 = 500
num_units_hidden_2 = 500
num_units_hidden_3 = 20
num_units_hidden_4 = num_units_hidden_2
num_units_hidden_5 = num_units_hidden_1

initializer = tf.keras.initializers.VarianceScaling()
params = {
    'activation': 'elu',
    'kernel_initializer': initializer
}

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, gamma = inputs
        return K.random_normal(tf.shape(gamma)) * K.exp(gamma / 2) + mean
```

### Define Encoder Model


```python
input_layer = tf.keras.layers.Input(shape = (28 * 28, ))

hidden_1 = tf.keras.layers.Dense(num_units_hidden_1, **params)(input_layer)
hidden_2 = tf.keras.layers.Dense(num_units_hidden_2, **params)(hidden_1)

mean  = tf.keras.layers.Dense(num_units_hidden_3, activation = None)(hidden_2)
gamma = tf.keras.layers.Dense(num_units_hidden_3, activation = None)(hidden_2)
codings = Sampling()([mean, gamma])

encoder = tf.keras.models.Model(inputs = [input_layer], outputs = [mean, gamma, codings])
```

### Define Decoder Model


```python
decoder_input_layer = tf.keras.layers.Input(shape = (20, ))

hidden_3 = tf.keras.layers.Dense(num_units_hidden_4, **params)(decoder_input_layer)
hidden_4 = tf.keras.layers.Dense(num_units_hidden_5, **params)(hidden_3)
output_layer = tf.keras.layers.Dense(28 * 28, activation = 'sigmoid')(hidden_4)

decoder = tf.keras.models.Model(inputs = [decoder_input_layer], outputs = [output_layer])
```

### Define Variational Autoencoder From Encoders And Decoders


```python
_, _, encodings = encoder(input_layer)
reconstructions = decoder(encodings)
vae_model = tf.keras.models.Model(inputs = [input_layer], outputs = [reconstructions])
```

### Define The Loss Function, And Compile The Model With An Adam Optimizer


```python
latent_loss = -0.5 * K.sum(1 + gamma - K.exp(gamma) - K.square(mean), axis = -1)
vae_model.add_loss(K.mean(latent_loss) / 784.)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
vae_model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
```

## MNIST dataset


```python
import matplotlib.pyplot as plt
import numpy as np

mnist_dataset = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
train_images, test_images = train_images / 255, test_images / 255

fig, axes = plt.subplots(6, 6, figsize = (15, 15))

samples = np.array([
    [0,  1,  2,  3,   4,  5],
    [6,  7,  8,  9,  10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35]
])

for row in range(0, 6):
    for col in range(0, 6):
        current_image = train_images[samples[row, col]]
        axes[row, col].set_title(train_labels[samples[row, col]])
        axes[row, col].axis('off')
        axes[row, col].imshow(current_image)
        
plt.show()

train_images = train_images.reshape((-1, 28 * 28))
test_images  = test_images.reshape((-1, 28 * 28))
images = np.vstack((train_images, test_images))
```


![png](output_15_0.png)


## Training


```python
fit_params = {
    'epochs': 10,
    'batch_size': 32
}
vae_model.fit(images, images, **fit_params)
```

    Train on 70000 samples
    Epoch 1/10
    70000/70000 [==============================] - 42s 604us/sample - loss: 0.1597
    Epoch 2/10
    70000/70000 [==============================] - 45s 641us/sample - loss: 0.1432
    Epoch 3/10
    70000/70000 [==============================] - 44s 629us/sample - loss: 0.1400
    Epoch 4/10
    70000/70000 [==============================] - 44s 624us/sample - loss: 0.1382
    Epoch 5/10
    70000/70000 [==============================] - 45s 641us/sample - loss: 0.1369
    Epoch 6/10
    70000/70000 [==============================] - 44s 627us/sample - loss: 0.1360
    Epoch 7/10
    70000/70000 [==============================] - 40s 570us/sample - loss: 0.1351
    Epoch 8/10
    70000/70000 [==============================] - 41s 591us/sample - loss: 0.1344
    Epoch 9/10
    70000/70000 [==============================] - 44s 629us/sample - loss: 0.1339
    Epoch 10/10
    70000/70000 [==============================] - 42s 606us/sample - loss: 0.1333





    <tensorflow.python.keras.callbacks.History at 0x6549bb090>



## Results


```python
input_codings = np.random.normal(size = [60, num_units_hidden_3])
generated_images = decoder.predict(input_codings)

fig, axes = plt.subplots(10, 6, figsize = (20, 20))
i = 0
for row in range(0, 10):
    for col in range(0, 6):
        current_image = np.array(generated_images[i]).reshape(28, 28)
        axes[row, col].axis('off')
        axes[row, col].imshow(current_image)
        i += 1
```


![png](output_19_0.png)

