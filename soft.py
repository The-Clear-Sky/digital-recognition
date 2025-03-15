import cv2
import numpy as np
from cv2 import minMaxLoc


def cv_show(win,img):
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 读取模板
tem = cv2.imread('template.png')
tem_gray = cv2.cvtColor(tem,cv2.COLOR_BGR2GRAY)
tem_bin = cv2.threshold(tem_gray,128,255,
                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
tem_contours = cv2.findContours(tem_bin, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
# print(len(tem_contours))
# tem_c = tem.copy()
# tem_c = cv2.drawContours(tem_c,tem_contours,-1,(255,0,0),3)
# cv_show('contours',tem_c)

# print('tem.ndim=',tem.ndim,'tem.shape=',tem.shape)
num_tem = len(tem_contours)
tem_array = []
x_array=[]
roi = []
if num_tem==10:
    for tem_contour in tem_contours:
        (x,y,w,h) = cv2.boundingRect(tem_contour)
        # print(x,y,w,h)
        roi = tem[y:y+h,x:x+w]
        # print(roi.size)
        # print(type(roi))
        x_array.append(x)
        tem_array.append(roi)
        # print('roi.shape=',roi.shape)

        # cv2.rectangle(tem_c,(x,y),(x+w,y+h),128,2)
        # cv_show('rectangle',tem_c)
(tem_sort, x_new) = zip(*sorted(zip(tem_array,x_array), key=lambda k: k[1]))
# tem_sort  是最终排序完成的单个数字

# print('#'*30)
# print('array.len=',len(tem_array))
# for index,item in enumerate(tem_sort):
#     print(index)
#     cv_show(str(index),item)


#下面进行数字识别
img = cv2.imread('test.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_bin = cv2.threshold(img_gray,128,256,cv2.THRESH_BINARY)[1]
cons = cv2.findContours(img_bin,cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[0]
conArray = []
xArray = []
if cons:
    for con in cons:
        (x,y,w,h) = cv2.boundingRect(con)
        Roi = img[y:y+h,x:x+w]
        conArray.append(Roi)
        xArray.append(x)
(conSorted,xArray) = zip(*sorted(zip(conArray,xArray),key=lambda c:c[1]))
# for index,con in enumerate(conSorted):
#     cv_show('sample'+str(index),con)
result = []

for index,con in enumerate(conSorted):
    scoreArray = []
    h,w=con.shape[0:2]
    resArray = []
    for temp in tem_sort :
        # print(temp.size)
        temp = cv2.resize(temp,(int(0.9*w),int(0.9*h)))
        res = cv2.matchTemplate(con,temp,cv2.TM_CCOEFF)
        (_,score,_,_) = minMaxLoc(res)
        # print(score)
        scoreArray.append(score)
    # print('_'*20)
    matchRes = np.argmax(scoreArray)
    result.append(matchRes)
print(result)



