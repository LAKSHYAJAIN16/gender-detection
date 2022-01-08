# Import Libraries
import sys
import cv2
import numpy as np

# Gender Model Paths
GENDER_MODEL = (
    r"D:\Projects\machine-learning\gender-detection\models\deploy_gender.prototxt"
)
GENDER_PROTO = (
    r"D:\Projects\machine-learning\gender-detection\models\gender_net.caffemodel"
)

# Face Model Paths
FACE_PROTO = r"D:\Projects\machine-learning\gender-detection\models\deploy.prototxt"
FACE_MODEL = r"D:\Projects\machine-learning\gender-detection\models\res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Mean values (got by trial and error)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Gender List we're predicting
GENDER_LIST = ["Male", "Female"]

# Frame Width & height
frame_width = 600
frame_height = 800

# Load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Utility function to display image
def display_img(title, img):
    # Display Image on screen
    cv2.imshow(title, img)

    # Mantain output until user presses a key
    cv2.waitKey(0)

    # Destroy windows when user presses a key
    cv2.destroyAllWindows()


# Get Font Scale
def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1
        )
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


# Code From : https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


# Function to get faces
def get_faces(frame, confidence_threshold=0.5):

    # Convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))

    # Set the image as input to the NN
    face_net.setInput(blob)

    # Perform inference and get predictions
    output = np.squeeze(face_net.forward())

    # Initialize the result list
    faces = []

    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)

            # widen the box a little
            start_x, start_y, end_x, end_y = (
                start_x - 10,
                start_y - 10,
                end_x + 10,
                end_y + 10,
            )
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


# Main function to predict gender
def predict_gender(input_path: str, test_num: int):
    # Read Input Image
    img = cv2.imread(input_path)

    # Take a copy of the initial image and resize it
    frame = img.copy()
    if frame.shape[1] > frame_width:
        frame = image_resize(frame, width=frame_width)

    # Predict the faces
    faces = get_faces(frame)

    # Loop over the faces detected
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y:end_y, start_x:end_x]

        # Create Blov
        blob = cv2.dnn.blobFromImage(
            image=face_img,
            scalefactor=1.0,
            size=(227, 227),
            mean=MODEL_MEAN_VALUES,
            swapRB=False,
            crop=False,
        )

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]

        # Draw the box
        label = "{}-{:.2f}%".format(gender, gender_confidence_score * 100)
        print(label)
        yPos = start_y - 15
        while yPos < 15:
            yPos += 15

        # Get the font scale for this image size
        optimal_font_scale = get_optimal_font_scale(label, ((end_x - start_x) + 25))
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)

        # Label processed image
        cv2.putText(
            frame,
            label,
            (start_x, yPos),
            cv2.FONT_HERSHEY_SIMPLEX,
            optimal_font_scale,
            box_color,
            2,
        )

        # Display processed image
        display_img("Gender Estimator", frame)

        # Save Image
        path = (
            r"D:\Projects\machine-learning\gender-detection\tests\output"
            + "-"
            + test_num.__str__()
            + ".jpg"
        )
        cv2.imwrite(path, frame)

        # Cleanup
        cv2.destroyAllWindows()


# Top Level
PATH_TO_IMAGE = r"D:\Projects\machine-learning\gender-detection\tests\16.jpg"
TEST_NUMBER = 16
predict_gender(PATH_TO_IMAGE, test_num=TEST_NUMBER)
