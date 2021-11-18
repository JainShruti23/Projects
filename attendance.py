import cv2 as cv 
import face_recognition as fr
import numpy as np
import os
import datetime

path = 'Research/photos'

images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findencodings(images):
    encodeList = []
    for img in images:
       # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open("Research/NN/attendance sheet.csv",'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeList = findencodings(images)

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    facesCurframe = fr.face_locations(frame)
    encode = fr.face_encodings(frame,facesCurframe)

    for encodeFace,faceLoc in zip(encode , facesCurframe):
        matches = fr.compare_faces(encodeList, encodeFace)
        faceDis = fr.face_distance(encodeList,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv.rectangle(frame,(x1+8,y1+8),(x2+8,y2+8),(0,255,0),2)
            cv.rectangle(frame,(x1+8,y2+35),(x2+8,y2),(0,255,0),cv.FILLED)
            cv.putText(frame,name,(x1+12,y2+25),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv.imshow("Display", frame)
    if cv.waitKey(1) == 27:
        break