import pathlib
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

dataset = pathlib.Path("datasets/PetImages")
model_name = 'pets_model'
model_path = pathlib.Path("models")

count = len(list(dataset.glob("*/*.jpg")))
batch_size = 8
epochs = 40
img_width = 180
img_height = 180

print(f"datasetis moculoba sheadgens: {count} fails\n")


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

#print the model
for images, labels in train_ds.take(1):
    model(images)
    
model.summary()


# TRAINING
try:
    history = model.fit(
    train_ds,
    validation_data=valid_ds,
    batch_size=batch_size,
    epochs=epochs
)
except Exception as e:
    print("An error occurred during training:")
    print(e)
    
print("done")
sys.stdout.flush()

#visualize
accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label = 'Training Accuracy') 
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('training and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'training loss')
plt.plot(epochs_range, val_loss, label = 'validation loss')
plt.legend(loc='upper left')
plt.title('training and validation loss')
plt.show()

# export the model
model_weights_filename = model_path / model_name
model.save_weights(model_weights_filename)
print('model saved.')