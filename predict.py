import pickle
from sklearn.metrics import *
from model import darkchannel, sobel,mkdf
import pandas as pd
import joblib
import numpy as np


#이미지 주소

def predict(X_test):
    svclassifier_from_pickle = joblib.load('fog.pkl')
    
    Y_pred = svclassifier_from_pickle.predict(X_test)
    if Y_pred == 0:
        print("fog")
    else:
        print("normal")
    
    #print("정확도 : ", accuracy_score(Y_test,Y_pred))
    #print(confusion_matrix(Y_test,Y_pred))
    #print(classification_report(Y_test,Y_pred))

img = "C:/takensoft/workspace/project1/FCBed/85CF_HD_20211216_031051.jpg"

alpha_map,result=darkchannel(img)
edge = sobel(img)
value=np.array([[result,edge]])
predict(value)

