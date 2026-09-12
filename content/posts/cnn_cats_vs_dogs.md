---
date: '2024-12-07T00:00:00-05:00'
draft: false
title: 'CNN Cats vs Dogs Classification'
author: 'Jorgen Bergstrom'
tags: ['CNN', 'Image Classification', 'Python']
categories: ['CNN']
---

## Introduction
In this example I will use the [Kaggle cats and dogs dataset](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip) to train a CNN to classify if an image is a cat or a dog.

The code below reads all images from a local `PetImages` folder and splits them into a training and a validation set. Before training, the images are preprocessed with simple data augmentation: random horizontal flips and small rotations are applied to make the model more robust, and the pixel values are rescaled to the range [0, 1].

The network itself is a small convolutional neural network built with Keras. It consists of three convolutional blocks, each pairing a 2D convolution with batch normalization and max pooling to extract increasingly abstract features, followed by a densely connected layer with dropout and a single output neuron that gives the cat/dog score.

The model is compiled with the Adam optimizer and binary cross-entropy loss, since this is a two-class problem, and is then trained for 25 epochs. Finally, the trained model is applied to an image it has not seen before, and the prediction is reported as a percentage for each class.

<br>

```python
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

"""
# Read in the images
"""
print("\n\n\n***\n*** Read in the data\n***\n")
print("> calling image_dataset_from_diectory")
image_size = (180, 180)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    directory="PetImages",
    validation_split=0.25,
    seed=123,
    subset="both",
    image_size=image_size,
    batch_size=99,
)

"""
# Plot some images?
        xx = list(train_ds.take(2)) # create a list with the first 2 entries in train_ds
        type(xx)          = <class 'list'>
        len(xx)           = 2
        type(xx[0])       = <class 'tuple'>   # a tuple of tensors
        len(xx[0])        = 2
        xx[0][0].shape    = (batch_size, 180, 180, 3)   # image matrix
        xx[0][1].shape    = (batch_size,)               # label
        xx[0][0][0].shape = (180, 180, 3)
"""
val = input("\n> do you want to plot some of the images? [y/n]: ")
if (val == 'y'):
    plt.figure(figsize=(10, 10))
    for elem in train_ds.take(1):
        for i in range(9): # plot 9 images
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(np.array(elem[0][i]).astype("uint8"))
            plt.title(int(elem[1][i]))
            plt.axis("off")
    plt.savefig("example_image.png")
    plt.show()

"""
# Preprocess the images
"""
print("\n> Preprocess (augment) the image data")
# JB: somehow these have to be global
augment1 = layers.RandomFlip("horizontal")
augment2 = layers.RandomRotation(0.1)  # rotate angle [-0.1*2*pi, +0.1*2*pi]
augment3 = layers.Rescaling(1.0/255)
def preprocess(images, labels):
    images = augment1(images)
    images = augment2(images)
    images = augment3(images)
    return images, labels
train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)

"""
# Build the model
"""
def jb_make_simple_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    # First Convolutional Block
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Second Convolutional Block
    x = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Third Convolutional Block
    x = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x= layers.MaxPooling2D((2,2))(x)
    # Flatten the 2D feature maps for dense layers
    x = layers.Flatten()(x)
    # Fully Connected Layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # Output Layer
    outputs = layers.Dense(1, activation=None)(x)
    return keras.Model(inputs, outputs)

print("\n> Build the model")
model = jb_make_simple_model(input_shape=image_size + (3,))
keras.utils.plot_model(model, show_shapes=True)

"""
# Train the model
"""
print("\n> Train the model")
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(train_ds, epochs=25, callbacks=callbacks, validation_data=val_ds)
model.save("cat_dog_model.keras")

"""
# Apply the model
"""
print("\n> Apply the model")
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
plt.imshow(img)
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```
