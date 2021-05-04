import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import playsound



def get_dataset():
    DIR = './processed_data/'
    train_df = pd.read_csv(DIR + 'train.csv')
    test_df = pd.read_csv(DIR + 'test.csv')

    y_train = train_df['Label'].values
    y_test = test_df['Label'].values

    X_train = train_df.drop(['Label'], axis=1).to_numpy()
    X_test = test_df.drop(['Label'], axis=1).to_numpy()

    print("X_train - {} , y_train - {}".format(X_train.shape, y_train.shape))
    print("X_test - {} , y_test - {}".format(X_test.shape, y_test.shape))

    return (X_train, y_train), (X_test, y_test)

def print_analysis(y_test, y_pred, y_score):
    print("Accuracy:", accuracy_score(y_test,y_pred))
    print("Precison:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC Score: ", roc_auc_score(y_test, y_score))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def average(y_pred):
  for i in range(len(y_pred) - 1):
    if i % 240 == 0 or (i+1) % 240 == 0:
      pass
    else: 
      average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
      if average >= 0.5:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
  return y_pred


def get_average_prediction(results):
  sum = 0.0
  for r in results:
        sum = sum + float(r)
  average = sum / len(results)
  if average >= 0.5:
        return "Drowsy"
  else:
        return "Alert"

