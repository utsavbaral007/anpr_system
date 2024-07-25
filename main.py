import cv2
import math
from ultralytics import YOLO
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.stats import mode
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Path of the image
img_path = 'vehicle-images/test1.jpg' 

# load the train dataset inorder to extract the classes
train_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory('digits_and_numbers_dataset/train/', batch_size = 32, class_mode = None, shuffle=False, target_size=(32, 32))

frame = cv2.imread(img_path)
frame = cv2.resize(frame, (1440, 960))

model = YOLO('anpr_nepal_yolov8.pt')

classnames = ['license-plate']

# Check if the frame is not None (indicating successful reading of the image)
if frame is not None:
    frame = cv2.resize(frame, (1440, 960))
    results = model(frame)

    # Array to store output image
    output_images = [] 

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            print(conf)
            if conf > 50 and class_detect == 'license-plate':
                # Crop the region containing the license plate
                license_plate = frame[y1:y2, x1:x2]
                # Convert the cropped license plate region to RGB format
                license_plate_rgb = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)
                output_images.append(license_plate_rgb) 
                # Draw a bounding box around the license plate region on the original image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bounding box in blue color

    image = rgb2gray(output_images[0])
    cv2.imshow("image", frame)
    cv2.waitKey(0)

    def skew_angle_hough_transform(image):
        # convert to edges
        edges = canny(image)
   
        # Classic straight-line Hough transform between 0.1 - 180 degrees.
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        
        # find line peaks and angles
        accum, angles, dists = hough_line_peaks(h, theta, d)
        
        # round the angles to 2 decimal places and find the most common angle.
        most_common_angle = mode(np.around(angles, decimals=2))[0]
        
        # convert the angle to degree for rotation.
        skew_angle = np.rad2deg(most_common_angle - np.pi/2)
        print(skew_angle)
        return skew_angle

    # Calculate the skew angle using the provided function
    skew_angle = skew_angle_hough_transform(image)

    # Get the image dimensions
    height, width = image.shape[:2]

    # Define the center of the image
    center = (width // 2, height // 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)

    # Convert the single-channel image to a 32-bit floating-point image
    rotated_image = cv2.convertScaleAbs(rotated_image)

    # Convert the single-channel image to a 3-channel image
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)

    # Normalize the pixel values to the range [0, 255]
    rotated_image = cv2.normalize(rotated_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    gray_img = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray image", gray_img)
    cv2.waitKey(0)


    gray = cv2.resize(gray_img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
 
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""

    segmented_characters = []

    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 0.5: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 15: continue
        
        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        segmented_characters.append(roi)
    
    cv2.imshow('Segmented Character Regions', im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    inverted_characters = []  # Array to store inverted characters
    for char in segmented_characters:
        inverted_char = cv2.bitwise_not(char)  # Invert the character
        inverted_characters.append(inverted_char)  # Store the inverted character

    # output_folder = 'digits_and_numbers_dataset'
    predicted_number = []
    model = load_model('vgg19.keras')
    for idx, inverted_char in enumerate(inverted_characters):
        # cv2.imshow(f"Inverted Character {idx+1}", inverted_char)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        resized_char = cv2.resize(inverted_char, (32, 32)) 
        # reshape into a single sample with 1 channel
        img = resized_char.reshape((1, resized_char.shape[0], resized_char.shape[1], 1))
        # # prepare pixel data
        img = img / 255.0
        # # convert grayscale to RGB
        img = np.concatenate((img, img, img), axis=-1)

        # predict the class
        predict_value = model.predict(img)
        predicted_class = np.argmax(predict_value, axis=1)
        class_indices = train_dataset.class_indices
        class_labels = {v: k for k, v in class_indices.items()}
        predicted_labels = [class_labels[class_idx] for class_idx in predicted_class]
        predicted_number.append(predicted_labels[0])
    
    print(predicted_number)

        # Generate characters Dataset
        # filename = os.path.join(output_folder, f'I3-{idx}.png')  # Define the file path
        # cv2.imwrite(filename, resized_char)





