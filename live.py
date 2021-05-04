import cv2
import dlib
import numpy as np
import pandas as pd
import time
from imutils import face_utils
from features import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye
from helpers import get_average_prediction
from load_model import model

def live(mean, std):

    # Set up predictor and detector from dlib
    p = "./models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Enable webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('DrowsinessDetector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('DrowsinessDetector', (1280,720))

    data_list = []
    result_list = []
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

            results = []
            feature_list = []

            for _ in range(3):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                result, features = model(shape, mean, std)
                results.append(result)
                feature_list.append(features)

            Result_String = get_average_prediction(results)

            
            if Result_String == "Alert":
                cv2.putText(image,Result_String, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,0),2)
            else:
                cv2.putText(image,Result_String, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150),2)

            cv2.putText(image,"Press q to quit program...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,99,71), 2)

            data_list.append (feature_list)
            result_list.append(Result_String)

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (139, 0, 0), -1)

        # Show the image
        cv2.imshow("DrowsinessDetector", image)

        if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    cap.release()
    
    return data_list,result_list

