
import cv2
import imutils as imutils
import numpy as np
from imutils import contours
from datetime import datetime
from lib_detection import load_model, detect_lp, im2single


#Ham sap xep contour tu trai sang phai
def sort_contours(cnts, method="left-to-right"):

    reverse = False
    i = 0

    # if method == "top-to-bottom":
    #     i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                         key=lambda b: b[1][i], reverse=reverse))
    return cnts



# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)
# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('trainSVM.txt')

# Đường dẫn ảnh để test
img_path = "test/test15.jpg"
# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# open file
f = open("FrameDetected.txt", "r+")

# Load video
vidCap = cv2.VideoCapture("test/vidTest2.mp4")
success, Ivehicle = vidCap.read()
count = 0
success = True
while success:
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    try:
        _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
        if (len(LpImg)):

            # Chuyen doi anh bien so
            LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

            roi = LpImg[0]
            label = lp_type
            # Chuyen anh bien so ve gray
            gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)


            # Ap dung threshold de phan tach so va nen
            # Nhị phân hóa ảnh bằng cách đặt ngưỡng
            binary = cv2.threshold(gray, 127, 255,
                                 cv2.THRESH_BINARY_INV)[1]

            cv2.imshow("Anh bien so sau threshold", binary)
            #cv2.waitKey()

            # Segment: cắt kí tự
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            cont, _= cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Tìm contours

            plate_info = ""

            cnt = 0
            plate_info_tmp = ""
            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                #ratio = h/w
                if label == 1:
                    if 1.5<=h/w<=3.5: # Chon cac contour dam bao ve ratio w/h
                        if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                            # Ve khung chu nhat quanh so
                            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # Tach so va predict
                            curr_num = thre_mor[y:y+h,x:x+w]
                            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                            curr_num = np.array(curr_num,dtype=np.float32)
                            curr_num = curr_num.reshape(-1, digit_w * digit_h)

                            # Dua vao model SVM
                            result = model_svm.predict(curr_num)[1]
                            result = int(result[0, 0])

                            if result<=9: # Neu la so thi hien thi luon
                                result = str(result)
                            else: #Neu la chu thi chuyen bang ASCII
                                result = chr(result)

                            plate_info +=result
                elif label == 2:
                    if w / h <= 1:
                        if h / roi.shape[0] >= 0.3:

                            cv2.rectangle(roi, (x, y), (x + (w), y + (h)), (0, 255, 0), 2)

                            curr_num = thre_mor[y:y + (h), x:x + (w)]
                            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                            curr_num = np.array(curr_num, dtype=np.float32)
                            curr_num = curr_num.reshape(-1, digit_w * digit_h)

                            result = model_svm.predict(curr_num)[1]
                            result = int(result[0, 0])

                            if result <= 9:
                                result = str(result)
                            else:
                                result = chr(result)
                            if cnt % 2: # Luu ki tu dong tren bien so
                                plate_info += result
                            else: # Luu ki tu dong duoi bien so
                                plate_info_tmp +=result
                            cnt += 1
                            #plate_info += result

            cv2.imshow("Cac contour tim duoc", roi)
            # cv2.waitKey()

            # Ghep 2 dong bien so vuong lai
            plate_info = plate_info + plate_info_tmp

            # Viet bien so len anh
            cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

            # Lay ngay gio
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")
            print(dt_string)
            # Luu frame nhan dien duoc
            cv2.imwrite("./Frame/frame %s.jpg" % dt_string, roi)
            f.write("Detected in frame")

            # Hien thi anh
            # print("Bien so=", plate_info)
            # cv2.imshow("Hinh anh output",Ivehicle)
            #c v2.waitKey()
    except:
        print("No plate detected!")
    cv2.imshow("Video", Ivehicle)
    if cv2.waitKey(40) == 27:
        break
    success, Ivehicle = vidCap.read()
    # print("Success save a new frame: ", success)
    count += 1


cv2.destroyAllWindows()
