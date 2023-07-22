import pathlib
import numpy as np
import tensorflow as tf

dataset = pathlib.Path("datasets/PetImages")
model_name = 'pets_model'
model_path = pathlib.Path(f"models/{model_name}")
test_image = 'test_image.jpg'

count = len(list(dataset.glob("*/*.jpg")))
batch_size = 16
img_number = 3
epochs = 30
img_width = 180
img_height = 180

print(f"\n\n== SOME STATS ==\n\n")


train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"class names: {class_names}")

# cache
autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = autotune)
valid_ds = valid_ds.cache().prefetch(buffer_size = autotune)

# create model
num_classes = len(class_names)
layers = tf.keras.layers
model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    
    # augmentation
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape = (img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.2),


    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    #regularization
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)    
])

#compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# load the model
model.load_weights(model_path).expect_partial()

# evaluate the model
loss, acc = model.evaluate(train_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}% \n".format(100 * acc))
print(f"\n\n== SOME STATS ==\n\n")

# load the test image
img = tf.keras.utils.load_img(test_image, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# print results
print("Detected: {} ({:.2f}% probability)".format(
    class_names[np.argmax(score)],
    100 * np.max(score)
))
    
print("\n\nDONE!")
