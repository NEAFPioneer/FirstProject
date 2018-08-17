# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:45:14 2018

@author: Chih Kai Cheng
"""

import cv2
import numpy as np
import time
import os
import glob
from FIT_pass import passwordsgenerator
import getpass

psw = getpass.getpass('Passwords : ')
psw = passwordsgenerator(psw)
if psw=='right':
    print ("Successfully Login")
    print ("Start the program............")
    print ("Collecting the face images...")
    cascPath = 'C:\\FIT\\haarcascade_frontalface_default.xml'
    src1 = 'C:\\follow_test\\data\\me\\'
    src2 = 'C:\\follow_test\\data\\test\\'
    faceCascade = cv2.CascadeClassifier(cascPath)
    path1 = glob.glob(src1+'*.jpg')
    path2 = glob.glob(src1+'*.txt')
    path3 = glob.glob(src2+'*.jpg')
    
    for file in path1 : 
            os.remove(file)
    for file in path2 : 
            os.remove(file)
    for file in path3 : 
            os.remove(file)
    video_capture = cv2.VideoCapture(0)
    total_num=0
    src_num=0
    error =0
    t1=time.time()
    t2=0
    while True:
        
    # Capture frame-by-frame
        ret, frame = video_capture.read()
        total_num+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #face detection 
        faces = faceCascade.detectMultiScale(gray,1.1,5)
        # calculate the area
        if type(faces) is not tuple:
            faces_cand_num = faces.shape[0]
            Area = np.zeros((faces_cand_num,1), dtype=np.int64)
            for x in range(faces_cand_num):
                Area[x,0] = faces[x,2]*faces[x,3] 
            faces = np.hstack((faces,Area))
            max_index = np.argmax(faces[:,4])
            x=faces[max_index,0]
            y=faces[max_index,1]
            w=faces[max_index,2]
            h=faces[max_index,3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # choose the correct face image
            if faces[max_index,4]>30000 and faces[max_index,4]<100000: #avoid the smaller or larger error image
                if  (t2-t1)>10 and (t2-t1)<=30 :
                    output = gray[faces[max_index,1]:faces[max_index,1]+faces[max_index,3],faces[max_index,0]:faces[max_index,0]+faces[max_index,2]]
                    cv2.imwrite(src1+str(src_num)+'.jpg',output)
                if (t2-t1)>30 and (t2-t1)<=45:
                    output = gray[faces[max_index,1]:faces[max_index,1]+faces[max_index,3],faces[max_index,0]:faces[max_index,0]+faces[max_index,2]]
                    cv2.imwrite(src2+str(src_num)+'.jpg',output)
                src_num+=1
        else:
            error+=1
        #create the textbox
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (60,30)
        fontScale              = 0.8
        fontColor              = (255,255,255)
        lineType               = 2
        cv2.rectangle(frame, (80,70), (80+490,70+330), (0,0,255), 3)
        cv2.putText(frame,'Please keep your head in the red box', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        if (t2-t1)>6 and (t2-t1)<10 : 
            bottomLeftCornerOfText = (40,60)
            cv2.putText(frame,'The collection is about to begin in 3 sec!', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        if (t2-t1)>10 and (t2-t1)<45 :
            bottomLeftCornerOfText = (60,60)
            cv2.putText(frame,'Face images collecting......', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        if (t2-t1)>45 and (t2-t1)<48 :
            bottomLeftCornerOfText = (40,60)
            cv2.putText(frame,'The collection is about to end in 3 sec!', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)    
        cv2.imshow('Video', frame)
        if (t2-t1)>48:
            break
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break 
        t2 = time.time()
    
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print ("----------------End of the collection of the face images----------------")
    print ("----------------The condition of image collections----------------")
    print ("The collection time : " , (t2-t1), "sec")
    print ("Total images : ", total_num, "frames")
    print ("Successful images : ", (total_num-error), "frames")
    print ("Error images : ",error, "frames")
elif psw=='wrong':
    print ("Invalid passwords!")