import cv2
import numpy as np

def getContours(img,cTHr=[100,100],showCanny=False, minArea = 1000, filter = 0, draw = False):

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5),1)
    imgCanny = cv2.Canny(imgBlur,cTHr[0],cTHr[1])

    #dilation & diffusion
    Kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,Kernel,iterations=3)
    imgThre = cv2.erode(imgDial,Kernel,iterations=2)

    if showCanny:cv2.imshow('Canny',imgCanny)

    #找四边形数据
    contours, hierarchy= cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #记录数据
    finalCountours = []

    for i in contours:
        #计算轮廓面积
        area = cv2.contourArea(i)
        if area > minArea:
            #求周长
            peri = cv2.arcLength(i,True)
            #外围拟合
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            #四边形拟合
            bbox = cv2.boundingRect(approx)

            #指定角确定形状
            if filter >0:
                if len(approx) == filter:
                    finalCountours.append([len(approx),area,approx,bbox,i])
            else:
                finalCountours.append([len(approx),area,approx,bbox,i])

    #面积降序
    finalCountours = sorted(finalCountours,key=lambda x:x[1],reverse=True)

    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)

    return img,finalCountours



#点顺序
def reorder(myPoints):
    print(myPoints.shape)

    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))

    add = myPoints.sum(1) #从0开始计数
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)#为后两项做准备
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew



# 截取A4纸平面
def warpImg(img,points,w,h,pad=20):
    # print(points)
    # print(reorder(points))
    points = reorder(points)

    #img
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])

    #映射变换
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    #透视变换
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    #缩进纸张
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]

    return imgWarp

# 计算长度
def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
