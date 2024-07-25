from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os
from keras.preprocessing import image

data_dir = 'digits_and_numbers_dataset/train'

image_class = os.listdir(data_dir)
images = os.listdir(os.path.join(data_dir, image_class[image_class.index('0')]))
class_index = image_class.index('0')

for pic in images:
  img = image.load_img(f'digits_and_numbers_dataset/train/{image_class[class_index]}/{pic}', target_size=(32, 32))

  datagen = ImageDataGenerator(
  rotation_range = 5,
  horizontal_flip = True,
  width_shift_range = 2.0,
  height_shift_range = 2.0,
  zca_whitening=True
)
  img = image.img_to_array(img)
  input_batch = img.reshape(1 ,32, 32, 3)
  i = 0
  for output in datagen.flow(input_batch, batch_size = 1, save_to_dir = 'digits_and_numbers_dataset'):
    i = i + 1

    #generates a total of 5 augmented images including the original one
    if i == 4:
      break




