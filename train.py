import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import Bunch
import torch
import torch.autograd
import pandas as pd
from sklearn.svm import SVC
from model import darkchannel,sobel,mkdf
import pickle
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"


    
li_category=['Hazy', 'Normal']
target=[]
average=[]
li_edge=[]
    
        
for q,category in enumerate(li_category):
    filelist= list(glob.glob(f"C:/takensoft/workspace/project1/085.다양한 기상 상황 주행 데이터/01.데이터/1.Training/원천데이터/TS_{category}/{category}/Day/Front/*"))
    random.shuffle(filelist)
    for file1 in filelist[:8000]:            
            alpha_map,result=darkchannel(file1)
            edge = sobel(file1)
            li_edge.append(edge)
            average.append(result)
            target.append(q)


X_train, X_test, Y_train, Y_test = mkdf(average,target,li_edge)

print(X_train, X_test, Y_train, Y_test)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, Y_train)
Y_pred = svclassifier.predict(X_test)
saved_model = pickle.dumps(svclassifier)
joblib.dump(svclassifier, 'C:/takensoft/workspace/project1/fog.pkl')
svclassifier_from_pickle = pickle.loads(saved_model)

Y_pred = svclassifier_from_pickle.predict(X_test)


print("정확도 : ", accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))