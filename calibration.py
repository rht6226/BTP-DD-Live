import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
from features import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye



def calibration():

    # Set up predictor and detector from dlib
    p = "./models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Enable webcam

    data = []
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', (1280,720))

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
            data.append(shape)
            cv2.putText(image,"Calibrating...", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,139), 2)
            cv2.putText(image,"Press ESC to end callibration...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,99,71), 2)

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Show the image
        cv2.imshow("image", image)
        

        k = cv2.waitKey(500) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

    features_test = []
    for d in data:
        eye = d[36:68]
        ear = eye_aspect_ratio(eye)
        mar = mouth_aspect_ratio(eye)
        cir = circularity(eye)
        mouth_eye = mouth_over_eye(eye)
        features_test.append([ear, mar, cir, mouth_eye])
    
    features_test = np.array(features_test)
    x = features_test
    y = pd.DataFrame(x,columns=["EAR","MAR","Circularity","MOE"])
    df_means = y.mean(axis=0)
    df_std = y.std(axis=0)
    
    # return values
    return df_means,df_std



if __name__ == '__main__':
    x, y = calibration()
    print(x)
    print(y)