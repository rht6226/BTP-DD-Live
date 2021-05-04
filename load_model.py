import numpy as np
import pandas as pd
from features import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye
import joblib


def model(landmarks, mean, std, classifier='random_forest'):

    model = joblib.load('./models/{}.pkl'.format(classifier) , mmap_mode ='r')

    features = pd.DataFrame(columns=["EAR","MAR","Circularity","MOE"])

    eye = landmarks[36:68]
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    cir = circularity(eye)
    mouth_eye = mouth_over_eye(eye)

    df = features.append({"EAR":ear,"MAR": mar,"Circularity": cir,"MOE": mouth_eye},ignore_index=True)

    df["EAR_N"] = (df["EAR"]-mean["EAR"])/ std["EAR"]
    df["MAR_N"] = (df["MAR"]-mean["MAR"])/ std["MAR"]
    df["Circularity_N"] = (df["Circularity"]-mean["Circularity"])/ std["Circularity"]
    df["MOE_N"] = (df["MOE"]-mean["MOE"])/ std["MOE"]
    
    Result = model.predict(df)
    # if Result == 1:
    #     Result_String = "Drowsy"
    # else:
    #     Result_String = "Alert"
    

    return Result, df