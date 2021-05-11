import numpy as np
import pandas as pd
from features import eye_aspect_ratio, mouth_aspect_ratio, circularity, mouth_over_eye
import joblib


def ensemble_model(landmarks, mean, std):

    modelNames = ['decision_tree', 'random_forest', 'xgb']
    models = [joblib.load('./models/{}.pkl'.format(modelName) , mmap_mode ='r') for modelName in modelNames]

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
    
    
    
    Result0 = models[0].predict(df)
    Result1 = models[1].predict(df)
    Result2 = models[2].predict(df.values)

    Result_stack = np.stack((Result0, Result1, Result2), axis=1)

    lr_model = joblib.load('./models/ensemble_model.pkl', mmap_mode='r')

    Result = lr_model.predict(Result_stack)
    

    # if Result == 1:
    #     Result_String = "Drowsy"
    # else:
    #     Result_String = "Alert"
    

    return Result, df