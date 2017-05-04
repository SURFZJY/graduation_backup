import os 
import numpy as np
import cv2
from keras.preprocessing.image import load_img , img_to_array
from keras.models import Sequential, load_model
from keras.utils import np_utils

path = r'H:\surfzjy\casiap_cut\20100723'
files = os.listdir(path)

des_path = r'C:\Users\liu\Documents\Python Scripts\keras_learn\20100723_cams_classified'
des_files = os.listdir(des_path)
nb_files = len(des_files)

for i in files[nb_files:]:
    img_path = path + '\\' + i
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    cv2.imwrite('tmp.jpg', resized_img)
    new_img = load_img('tmp.jpg')
    x = img_to_array(new_img)
    x = np.array(x)
    x /= 255
    x = x[np.newaxis, ...]
#    model = load_model('kiel_model.h5')
    model = load_model('CAMS_model.h5')
    res = model.predict(x)
    pred_label = np.argmax(res)
#    cloud_name = ['1_cummulus', '2_cirrus', '3_altocumulus', '4_clearsky', '5_stratocumulus', '6_stratus', '7_cumulonimbus']
    cloud_name = ['1 cirrus', '2 clear', '3 cumulus', '4 hybrid', '5 stratus']
    new_name = des_path + '\\' + i.split('.')[0] + '_' + cloud_name[pred_label] + '.jpg'
    cv2.imwrite(new_name, img)