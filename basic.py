import cv2
from cv2 import putText
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.resize(img, (400, 400))
#img_e = cv2.imread("whit.jpg")
img = cv2.imread("imidterm_mage\medicine_with_noise.jpg")
img = cv2.resize(img, (600, 600))
img_BG = np.ones((600, 600, 3), dtype=np.uint8)*255
blur = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imwrite("asd.jpg",blur)
# ช่วงสี B G R l
#สีดำ
lower_D = np.array([0, 0, 0])
upper_D = np.array([104, 110, 105])
mask_D = cv2.inRange(blur, lower_D, upper_D)
result_D = cv2.bitwise_and(blur, blur, mask=mask_D)
#สีส้ม
lower_O = np.array([119, 127, 139])
upper_O = np.array([168, 188, 216])
mask_O = cv2.inRange(blur, lower_O, upper_O)
result_O = cv2.bitwise_and(blur, blur, mask=mask_O)
#สีเหลือง
lower_y = np.array([33, 85, 85])
upper_y = np.array([98, 222, 255])
mask_y = cv2.inRange(blur, lower_y, upper_y)
result_y = cv2.bitwise_and(blur, blur, mask=mask_y)
#สีชมพู
lower_P = np.array([106, 111, 125])
upper_P = np.array([186, 170, 224])
mask_P = cv2.inRange(blur, lower_P, upper_P)
result_P = cv2.bitwise_and(blur, blur, mask=mask_P)
#สีน้ำเงิน
lower = np.array([106, 111, 125])#([65, 55,  65])
upper = np.array([186, 170, 224])#([189, 158, 121])
mask_B = cv2.inRange(blur, lower, upper)
result_B = cv2.bitwise_and(blur, blur, mask=mask_B)
# kernel = np.ones((2,2),dtype = np.uint8) 
# mask_B = cv2.morphologyEx(mask_B,cv2.MORPH_CLOSE,kernel) 
# mask_B = cv2.morphologyEx(mask_B,cv2.MORPH_OPEN,kernel)

# ขนาด Array ภาพ
print(blur.shape)
maskB_0 = np.zeros((600, 600), dtype=np.uint8) #เก็บวัตถุเเต่ละอันเพื่อไปหาตำเเหน่งเเต่ล่ะอัน
maskB_1 = np.zeros((600, 600), dtype=np.uint8)
maskB_2 = np.zeros((600, 600), dtype=np.uint8)
maskB_3 = np.zeros((600,600),dtype = np.uint8)

request, arrange = cv2.findContours(mask_B, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(request)):
    area = cv2.contourArea(request[i])
    if area == 679.0:#387.0
        cv2.drawContours(maskB_0,request,i,(255,255,255),-1)
    if area == 721.0:#359.0
        cv2.drawContours(maskB_1,request,i,(255,255,255),-1)
    if area == 673.5 :#420.0
        cv2.drawContours(maskB_2,request,i,(255,255,255),-1)
    if area == 525.0 :#273.5
        cv2.drawContours(maskB_3,request,i,(255,255,255),-1)

    
    # cv2.imshow("img",maskB_3)
    mask_indices_0 = np.where(maskB_0==255)
    img_BG[mask_indices_0[0]-330,mask_indices_0[1]-200] = blur[mask_indices_0]

    mask_indices_1 = np.where(maskB_1==255)
    img_BG[mask_indices_1[0]-239,mask_indices_1[1]-120] = blur[mask_indices_1]
    # cv2.imshow("img",maskB_1)

    mask_indices_2 = np.where(maskB_2==255)
    img_BG[mask_indices_2[0]-148,mask_indices_2[1]-32] = blur[mask_indices_2]

    mask_indices_3 = np.where(maskB_3==255)
    img_BG[mask_indices_3[0]-72,mask_indices_3[1]-44] = blur[mask_indices_3]



# #สีเหลือง
maskY_0 = np.zeros((600, 600), dtype=np.uint8) #เก็บวัตถุเเต่ละอันเพื่อไปหาตำเเหน่งเเต่ล่ะอัน
maskY_1 = np.zeros((600, 600), dtype=np.uint8)

request_y, arrange_y = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(request_y)):
    area_y = cv2.contourArea(request_y[i])
    # print(area_y)
    if area_y == 2423.0:
        cv2.drawContours(maskY_0,request_y,i,(255,255,255),-1)
    if area_y == 2407.0:
        cv2.drawContours(maskY_1,request_y,i,(255,255,255),-1)

    maskY_space = np.where(maskY_0==255)
    img_BG[maskY_space[0]-145,maskY_space[1]] = blur[maskY_space]
    # cv2.imshow("img_1",maskY_0)

    maskY_space1 = np.where(maskY_1==255)
    img_BG[maskY_space1[0]-177,maskY_space1[1]] = blur[maskY_space1]
    # cv2.imshow("img_25",maskY_1)
#สีชมพู
maskP_0 = np.zeros((600, 600), dtype=np.uint8) #เก็บวัตถุเเต่ละอันเพื่อไปหาตำเเหน่งเเต่ล่ะอัน
maskP_1 = np.zeros((600, 600), dtype=np.uint8)
maskP_2 = np.zeros((600, 600), dtype=np.uint8)
maskP_3 = np.zeros((600, 600), dtype=np.uint8)
maskP_4 = np.zeros((600, 600), dtype=np.uint8)

#สีดำ
maskD_1 = np.zeros((600, 600), dtype=np.uint8)
maskD_2 = np.zeros((600,600),dtype = np.uint8)
request_P, arrange_P = cv2.findContours(mask_P, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(request_P)):
    area_P = cv2.contourArea(request_P[i])
    # print(area_P)
    if area_P == 928.5:
        cv2.drawContours(maskP_0,request_P,i,(255,255,255),-1)
    if area_P == 668.0:
        cv2.drawContours(maskP_1,request_P,i,(255,255,255),-1)
    if area_P == 594.5:
        cv2.drawContours(maskP_2,request_P,i,(255,255,255),-1)
    if area_P == 640.5:
        cv2.drawContours(maskP_3,request_P,i,(255,255,255),-1)
    if area_P == 686.0:#686.0
        cv2.drawContours(maskP_4,request_P,i,(255,255,255),-1)

    #สีดำ
    if area_P == 1915.5:
        cv2.drawContours(maskD_1,request_P,i,(255,255,255),-1)
    if area_P == 1573.5:
        cv2.drawContours(maskD_2,request_P,i,(255,255,255),-1)

    
    maskP_space0 = np.where(maskP_0==255)
    img_BG[maskP_space0[0]-295,maskP_space0[1]-55] = blur[maskP_space0]

    maskP_space1 = np.where(maskP_1==255)
    img_BG[maskP_space1[0]-250,maskP_space1[1]-245] = blur[maskP_space1]

    maskP_space2 = np.where(maskP_2==255)
    img_BG[maskP_space2[0]-190,maskP_space2[1]-210] = blur[maskP_space2]

    maskP_space3 = np.where(maskP_3==255)
    img_BG[maskP_space3[0]-50,maskP_space3[1]-25] = blur[maskP_space3]

    maskP_space4 = np.where(maskP_4==255)
    img_BG[maskP_space4[0],maskP_space4[1]-40] = blur[maskP_space4]

    
    #สีดำ
    maskD_space1 = np.where(maskD_1==255)
    img_BG[maskD_space1[0]-50,maskD_space1[1]-73] = blur[maskD_space1]

    maskD_space2 = np.where(maskD_2==255)
    img_BG[maskD_space2[0],maskD_space2[1]-140] = blur[maskD_space2]
    # cv2.imshow("img_5",maskD_2)
    
maskD_3 = np.zeros((600, 600), dtype=np.uint8)
request_D, arrange_D = cv2.findContours(mask_D, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(request_D)):
    area_D = cv2.contourArea(request_D[i])
    print(area_D)
    if area_D == 1180.0:
        cv2.drawContours(maskD_3,request_D,i,(255,255,255),-1)
        maskD_space3 = np.where(maskD_3==255)
        img_BG[maskD_space3[0]+120,maskD_space3[1]-185] = blur[maskD_space3]

#สีส้ม
maskO_1 = np.zeros((600, 600), dtype=np.uint8)
maskO_2 = np.zeros((600, 600), dtype=np.uint8)
mzskO_3 = np.zeros((600, 600), dtype=np.uint8)
request_O, arrange_O = cv2.findContours(mask_O, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(request_O)):
    area_O = cv2.contourArea(request_O[i])
    print(area_O)
    if area_O == 1174.5:
        cv2.drawContours(maskO_1,request_O,i,(255,255,255),-1)
    if area_O == 1237.5:
        cv2.drawContours(maskO_2,request_O,i,(255,255,255),-1)
    if area_O == 909.0:
        cv2.drawContours(mzskO_3,request_O,i,(255,255,255),-1)
       
maskO_space1 = np.where(maskO_1==255)
img_BG[maskO_space1[0]+27,maskO_space1[1]+270] = blur[maskO_space1]
maskO_space2 = np.where(maskO_2==255)
img_BG[maskO_space2[0]-90,maskO_space2[1]+60] = blur[maskO_space2]
maskO_space3 = np.where(mzskO_3==255)
img_BG[maskO_space3[0]-160,maskO_space3[1]+10] = blur[maskO_space3]

cv2.rectangle(img_BG,(10,30),(180,100),(0,0,0),1)
cv2.putText(img_BG,"Med1: 4 tablet",(20,24),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))

cv2.rectangle(img_BG,(10,160),(210,235),(0,0,0),1)
cv2.putText(img_BG,"Med2: 5 tablet",(20,155),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))

cv2.rectangle(img_BG,(10,270),(210,350),(0,0,0),1)
cv2.putText(img_BG,"Med2: 3 capsule",(20,263),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))
#XY.XY
cv2.rectangle(img_BG,(360,30),(540,100),(0,0,0),1)
cv2.putText(img_BG,"Med2: 2 tablet",(370,25),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))

cv2.rectangle(img_BG,(360,160),(540,235),(0,0,0),1)
cv2.putText(img_BG,"Med2: 3 capsule",(370,155),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))
# cv2.imshow("blur55", mzskO_3)
# cv2.imshow("blur", blur)

# cv2.imshow("mack", mask_y)
# cv2.imshow("result", result_O)

# cv2.imshow("mack_B", mask_B)
cv2.imshow("mack_BG", img_BG)

# cv2.imshow("mack_B", mask_P)

# cv2.imshow("result_B", result_B)
cv2.waitKey()
