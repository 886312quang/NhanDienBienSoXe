import cv2
import numpy as np
import os
import glob

w = 30
h = 60

path = "./data/"

def get_digit_data(path):
    digitList = []
    labelList = []

    # load so
    for number in range(10):
        i = 0
        for imgPath in glob.iglob(path + str(number) + '/*.jpg'):
            #print(imgPath)
            img = cv2.imread(imgPath, 0)
            img = np.array(img)
            #print(img.shape)
            img = img.reshape(-1, h * w)
            #print(img.shape)

            # luu cac so da load vao digitList va gan nhan so do trong labelList
            digitList.append(img)
            labelList.append([int(number)])
            #print([int(number)])

    #load ki tu
    for number in range(65, 91):
        i = 0
        for imgPath in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(imgPath, 0)
            img = np.array(img)
            img = img.reshape(-1, h * w)

            # luu cac ki tu da load vao digitList va gan nhan so do trong labelList
            digitList.append(img)
            labelList.append([int(number)])

    return digitList, labelList

digitList, labelList = get_digit_data(path)

digitList = np.array(digitList, dtype=np.float32)
digitList = digitList.reshape(-1, h * w)

labelList = np.array(labelList)
labelList = labelList.reshape(-1, 1)

svmModel = cv2.ml.SVM_create()
svmModel.setType(cv2.ml.SVM_C_SVC)
svmModel.setKernel(cv2.ml.SVM_INTER)
svmModel.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svmModel.train(digitList, cv2.ml.ROW_SAMPLE, labelList)

svmModel.save("trainSVM.txt")
print("Train success!")