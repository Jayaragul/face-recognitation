import cv2 # OpenCV
import os # Library os
from retinaface import RetinaFace # RetinaFace library for face detection

# Set paths for dataset
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'  
sub_data = 'jayaragul'  # Name of the folder to store the data

# Create the directory if it doesn't exist
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)  # Resize dimensions for the face

# Initialize webcam
webcam = cv2.VideoCapture(0)

count = 1
while count < 31:
    print(count)
    (_, im) = webcam.read()  # Read frame from webcam
    
    # RetinaFace detection (this returns face landmarks and bounding boxes)
    faces = RetinaFace.detect_faces(im)
    
    if faces:  # If faces are detected
        for key in faces:
            face_data = faces[key]  # Get the data of detected face
            facial_area = face_data['facial_area']  # The bounding box of the face
            x, y, x_w, y_h = facial_area  # Unpack the bounding box coordinates
            
            # Draw rectangle around the face
            cv2.rectangle(im, (x, y), (x_w, y_h), (255, 0, 0), 2)
            
            # Crop the face part from the image
            face = im[y:y_h, x:x_w]
            face_resize = cv2.resize(face, (width, height))  # Resize the face
            
            # Save the cropped face
            cv2.imwrite('%s/%s.png' % (path, count), face_resize)
            count += 1

    # Display the image with rectangles drawn
    cv2.imshow('RetinaFace - Face Detection', im)
    
    # Break the loop if 'q' is pressed
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
