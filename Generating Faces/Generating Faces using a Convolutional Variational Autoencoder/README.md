# Generating Faces using a Convolutional Variational Autoencoder

## Creating A Convolutional Variational Autoencoder

### Import Necessary Libraries


```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Define Parameters And A Sampling Layer


```python
input_dim = (128, 128, 3)

encoder_conv_filters   = [32, 64, 64, 64]
decoder_conv_t_filters = [64, 32, 32, 3]

z_dim = 200

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, gamma = inputs
        return K.random_normal(tf.shape(gamma)) * K.exp(gamma / 2) + mean
```

### Define Encoder Model


```python
encoder_input = Input(shape = input_dim, name = 'encoder_input')

x = encoder_input
for i in range(4):
    encoder_params = {
        'filters': encoder_conv_filters[i],
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same',
        'name': 'encoder_conv_' + str(i)
    }
    x = Conv2D(**encoder_params)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate = 0.25)(x)
    
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)

mean  = Dense(z_dim, name = 'mean')(x)
gamma = Dense(z_dim, name = 'gamma')(x)
encoder_output = Sampling()([mean, gamma])

encoder = Model(encoder_input, encoder_output, name = 'encoder')    
```

### Define Decoder Model


```python
decoder_input = Input(shape = (z_dim, ), name = 'decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)

for i in range(4):
    decoder_params = {
        'filters': decoder_conv_t_filters[i],
        'kernel_size': 3,
        'strides': 2,
        'padding': 'same',
        'name': 'decoder_conv_' + str(i)
    }
    x = Conv2DTranspose(**decoder_params)(x)
    if i < 3: 
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(rate = 0.25)(x)
    else: 
        x = Activation('sigmoid')(x)
    
decoder_output = x
decoder = Model(decoder_input, decoder_output, name = 'decoder')

```

### Define Convolutional Variational Autoencoder


```python
model_input  = encoder(encoder_input)
model_output = decoder(model_input)
model        = Model(encoder_input, model_output)
```

### Define Loss Function, And Compile The Model With An Adam Optimizer


```python
latent_loss = -0.0005 * K.sum(1 + gamma - K.exp(gamma) - K.square(mean), axis = -1)
model.add_loss(latent_loss)

optimizer = Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = 'mse')
```

## Faces dataset

### Data Preperation


```python
generator = ImageDataGenerator(rescale = 1. / 255)

#images folder has a sub folder that contains actual images
params = {
    'batch_size': 2, 
    'directory': './images/', 
    'shuffle': True, 
    'target_size': (128, 128),
    'class_mode': 'input'
}

image_data_generator = generator.flow_from_directory(**params)
```

    Found 5186 images belonging to 1 classes.


### Display Images


```python
def plot_images(images_arr):
    fig, axes = plt.subplots(5, 5, figsize = (20, 20))
    axes = axes.flatten()
    
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_images([image_data_generator[i][0][0] for i in range(25)])
```


![png](output_18_0.png)


## Training


```python
fit_params = {
    'epochs': 10,
    'steps_per_epoch': 5186 // 2,
    'verbose': 1
}

history = model.fit(image_data_generator, **fit_params)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 2593 steps
    Epoch 1/10
    2593/2593 [==============================] - 331s 128ms/step - loss: 0.0643
    Epoch 2/10
    2593/2593 [==============================] - 325s 125ms/step - loss: 0.0380
    Epoch 3/10
    2593/2593 [==============================] - 349s 135ms/step - loss: 0.0339
    Epoch 4/10
    2593/2593 [==============================] - 337s 130ms/step - loss: 0.0321
    Epoch 5/10
    2593/2593 [==============================] - 358s 138ms/step - loss: 0.0313
    Epoch 6/10
    2593/2593 [==============================] - 400s 154ms/step - loss: 0.0304
    Epoch 7/10
    2593/2593 [==============================] - 330s 127ms/step - loss: 0.0298
    Epoch 8/10
    2593/2593 [==============================] - 308s 119ms/step - loss: 0.0289
    Epoch 9/10
    2593/2593 [==============================] - 324s 125ms/step - loss: 0.0282
    Epoch 10/10
    2593/2593 [==============================] - 294s 114ms/step - loss: 0.0276


## Results


```python
z_input = np.random.normal(size = [25, 200])
generated_image = decoder.predict(z_input)

fig, axes = plt.subplots(5, 5, figsize = (20, 20))
i = 0
for row in range(0, 5):
    for col in range(0, 5):
        current_image = generated_image[i]
        axes[row, col].axis('off')
        axes[row, col].imshow(current_image)
        i += 1

plt.tight_layout()
plt.show()

```


![png](output_22_0.png)

