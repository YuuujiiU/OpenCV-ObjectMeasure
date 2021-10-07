import cv2
import numpy as np
import utlis

webcam = True
path = '1.jpg'

cap = cv2.VideoCapture(1)

#图像亮度
cap.set(10,1060)
#图像宽度
cap.set(3,1920)
#图像高度
cap.set(4,1080)

# P4纸张显示大小
scale = 3
wP = 210 * scale
hP = 297 * scale




while True:

    # 测试摄像头
    if webcam:success, img = cap.read()
    # 测试图片
    else: img = cv2.imread(path)


    img, conts= utlis.getContours(img, minArea=50000,filter=4)

    # A4内物体
    if len(conts)!= 0:
        # 外围拟合approx数据
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utlis.warpImg(img,biggest,wP,hP)
        imgContours2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4,cTHr=[50,50],draw=False)

        # 绘制A4物体外框拟合
        if len(conts2)!=0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = utlis.reorder(obj[2])
                nW = round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((utlis.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)

                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)


        cv2.imshow('A4', imgContours2)


    #改变图片大小
    img = cv2.resize(img,(0,0),None,0.5,0.5)

    cv2.imshow('original',img)
    cv2.waitKey(1)