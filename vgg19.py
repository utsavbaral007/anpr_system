# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from tensorflow.keras import Model, layers
import matplotlib.pyplot as plt

# Load the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification head
additional_layer = base_model.output
additional_layer = layers.Flatten()(additional_layer)
additional_layer = layers.Dense(1024, activation='relu')(additional_layer)
additional_layer = layers.Dropout(0.5)(additional_layer)
additional_layer = layers.Dense(29, activation='softmax')(additional_layer)

# Define the model
model = Model(inputs=base_model.input, outputs=additional_layer)

print(model.summary())

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the data generators
train_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory('digits_and_numbers_dataset/train/', batch_size = 32, class_mode = 'categorical', target_size=(32, 32))
valid_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory('digits_and_numbers_dataset/valid/', batch_size = 32, class_mode = 'categorical', target_size=(32, 32))
test_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory('digits_and_numbers_dataset/test/', batch_size = 32, class_mode = 'categorical', target_size=(32, 32))

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=valid_dataset)

# Save the model
model.save('vgg19.keras')

# Evaluate the model
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.tight_layout()
plt.show()

