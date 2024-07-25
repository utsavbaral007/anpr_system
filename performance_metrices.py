import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt

test_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory('digits_and_numbers_dataset/test/', batch_size = 32, class_mode = None, shuffle=False, target_size=(32, 32))

model = load_model('resnet50.keras')

number_of_classes = len(test_dataset.class_indices)

y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = test_dataset.classes

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes) 
plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

for i in range(number_of_classes):
    for j in range(number_of_classes):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment='center', color='black')

plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(number_of_classes)
plt.xticks(tick_marks, test_dataset.class_indices, rotation=45)
plt.yticks(tick_marks, test_dataset.class_indices)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

print("Classification Report:")
class_names = list(test_dataset.class_indices.keys())
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print(report)