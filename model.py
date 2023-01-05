
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

def darkchannel(file1):
    img = imread(file1)
    img = resize(img, (512,512))
    
    b,g,r = cv2.split(img)
    jx = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(64,64))
    dark=cv2.erode(jx,kernel)

    size = img.shape[:2]
    k = int(0.001*np.prod(size))
    idx = np.argpartition(-dark.ravel(),k)[:k]
    x, y = np.hsplit(np.column_stack(np.unravel_index(idx, size)), 2)
    A = np.array([img[x,y,0].max(), img[x,y,1].max(), img[x,y,2].max()])

    zz = np.column_stack(np.unravel_index(idx, dark.shape))
    x, y = np.hsplit(zz,2)

    norm_img = img / A
    b2,g2,r2= cv2.split(norm_img)
    jx2 = cv2.min(cv2.min(r2,g2),b2)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(64,64))
    dark2=cv2.erode(jx2,kernel2)
    alpha_map=(1-0.95*dark2)*255
    alpha_map = np.array(alpha_map)
    result=np.average(alpha_map)

    return alpha_map,result
        
def sobel(img):
    img_color = imread(img)
    #img_color= resize(img_color,(512,512))
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    
    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    
    img_sobel=np.array(img_sobel)
    result=np.var(img_sobel)
    return result

def mkdf(average,target,li_edge):
    
    df=pd.DataFrame({"average":average,"target":target, "edge":li_edge})

    Y_data = df['target']
    X_data = df.drop('target', axis = 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)
    
    return X_train, X_test, Y_train, Y_test
