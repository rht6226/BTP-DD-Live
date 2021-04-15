import cv2
import dlib
import numpy as np
import pandas as pd
import time
from imutils import face_utils
from features import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye
from load_model import model

def live(mean, std):

    # Set up predictor and detector from dlib
    p = "./models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Enable webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', (1280,720))

    data = []
    result = []
    while True:
        # Getting out image by webcam 
        _, image = cap.read()
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(image, 0)

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            Result_String, features = model(shape, mean, std)

            
            
            cv2.putText(image,Result_String, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 136),2)
            cv2.putText(image,"Press ESC to end program...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,99,71), 2)

            data.append (features)
            result.append(Result_String)

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Show the image
        cv2.imshow("image", image)

        k = cv2.waitKey(3000) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    
    return data,result

