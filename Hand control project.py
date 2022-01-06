import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
class HandDetector():
    def __init__(self, mode=False, maxHands =2 , modelComplexity=1, detectionCon= 0.6, trackCon= 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findHands(self, img, draw= True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img, self.results

    def findPosition(self, img, handNo = 0, draw = True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx ,cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id , cx ,cy])
                if draw:        
                    cv2.circle(img, (cx, cy), 5, (255,0,2))
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin , ymin, xmax ,ymax
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax+20 , ymax + 20), (0,255 , 0) ,2)
        
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]- 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2]< self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #Total Fingers 
        return fingers

    def findDistance(self, p1,p2,img, draw=True, r=3, t=3):
        x1,y1 = self.lmList[p1][1:]
        x2, y2 =self.lmList[p2][1:]
        cx ,cy = (x1 + x2) //2 , (y1 +y2) // 2

        if draw :
            cv2.line(img , (x1, y1), (x2 ,y2), (255 , 0 ,255) , t)
            cv2.circle(img ,(x1,y1) , r, (255, 0 , 255), cv2.FILLED) 
            cv2.circle(img ,(x2,y2) , r, (255, 0 , 255), cv2.FILLED) 
            cv2.circle(img ,(cx,cy) , r, (0, 0 , 255), cv2.FILLED) 
        length = math.hypot(x2 - x1, y2 -y1) 
        return length , img, [x1, y1, x2, y2, cx, cy]

    
    def countFingers(self ,frame, results, draw = True):
        '''
        This function will count the number of fingers up for each hand in the image.
        Args:
            image:   The image of the hands on which the fingers counting is required to be performed.
            results: The output of the hands landmarks detection performed on the image of the hands.
            draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                    output image.
            display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
        Returns:
            output_image:     A copy of the input image with the fingers count written, if it was specified.
            fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
            count:            A dictionary containing the count of the fingers that are up, of both hands.
        '''
        height ,width, _ = frame.shape
        # Create a copy of the input image to write the count of fingers on
        output_image = frame.copy()
        # Initialize a dictionary to store the count of fingers of both hands.
        count = {'RIGHT':0 , 'LEFT':0}
        # Store the indexes of the tips landmarks of each finger of a hand in a list.
        fingers_tips_ids = [self.mpHands.HandLandmark.INDEX_FINGER_TIP, self.mpHands.HandLandmark.MIDDLE_FINGER_TIP,
                            self.mpHands.HandLandmark.RING_FINGER_TIP, self.mpHands.HandLandmark.PINKY_TIP]

        # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
        fingers_statuses = {'RIGHT_THUMB':False , 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING':False, 'RIGHT_PINKY':False ,
        'LEFT_THUMB':False , 'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING':False, 'LEFT_PINKY':False }

        # Iterate over the found hands in the image.
        for hand_index,hand_info in enumerate(results.multi_handedness):
            # Retrieve the label of the found hand.
            hand_label = hand_info.classification[0].label
            # Retrieve the landmarks of the found hand.
            hand_landmarks = results.multi_hand_landmarks[hand_index]
            # Iterate over the indexes of the tips landmarks of each finger of the hand.
            for tip_index in fingers_tips_ids:
                finger_name = tip_index.name.split("_")[0]
                print("Tip: ",hand_landmarks.landmark[tip_index].y)
                print(hand_landmarks.landmark[tip_index -2].y)
                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index -2].y):
                    # Update the status of the finger in the dictionary to true.
                    fingers_statuses[hand_label.upper()+"_"+finger_name] = True

                    count[hand_label.upper()] += 1
            # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
            thumb_tip_x = hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_TIP-2].x
            # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
            if (hand_label=='Right' and (thumb_tip_x< thumb_mcp_x)) or (hand_label=="Left" and (thumb_tip_x> thumb_mcp_x)):
                fingers_statuses[hand_label.upper()+"_THUMB"] = True

                count[hand_label.upper()] += 1
        # Check if the total count of the fingers of both hands are specified to be written on the output image.
        if draw:
            cv2.putText(output_image, "Total Fingers: ",(10, 25), cv2.FONT_HERSHEY_COMPLEX , 1, (20, 255, 155), 2)
            cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                        8.9, (20,255,155), 10, 10)

        return output_image, fingers_statuses, count


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    wCam, hCam= 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = HandDetector()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    while True:
        ch = int(input("""\t1. Counter
        2. Volume Control
        3. Virtual Mouse
        4. Quit
        ***Press Esc key to exit your current choice***
        Enter your choice:"""))
        q = 0
        while True:
            success , img = cap.read()
            img = cv2.flip(img,1)
            img.flags.writeable = False
            img, results = detector.findHands(img)
            img.flags.writeable = True
            lmList, bbox = detector.findPosition(img, draw=False)
            # if len(lmList)!=0:
            #     print(lmList[4])
            cTime = time.time()
            fps = 1 / (cTime-pTime)
            pTime = cTime

            cv2.putText(img, f"FPS: {int(fps)}", (10,70), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,2),1)
            if ch==1:  
                    output_image = img.copy()
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            detector.mpDraw.draw_landmarks(output_image, hand_landmarks, detector.mpHands.HAND_CONNECTIONS, 
                            landmark_drawing_spec= detector.mpDraw.DrawingSpec(color=(255,255,255),
                                                    thickness=2, circle_radius=2),
                                                    connection_drawing_spec= detector.mpDraw.DrawingSpec(color=(0,255,0),
                                                    thickness=2, circle_radius=2))
                        img , fingers_statuses, count = detector.countFingers(output_image, results)
            elif ch==2:
                volume.SetMasterVolumeLevel(-65, None)
                minVol = volRange[0]
                maxVol = volRange[1]
                if len(lmList)!=0:
                    # print(lmList[4],lmList[8])
                    x1 ,y1 = lmList[4][1],lmList[4][2]
                    x2 ,y2 = lmList[8][1],lmList[8][2]
                    cx , cy = (x1 + x2)//2 , (y1+y2)//2
                    cv2.circle(img, (x1,y1), 15, (255,255,0), cv2.FILLED)
                    cv2.circle(img, (x2,y2), 15,  (255,255,0), cv2.FILLED)
                    cv2.circle(img, (cx,cy), 15, (255,255,0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2,y2),(255,0,255),3)
                    length = math.hypot(x2 -x1, y2 -y1)
                    # print(length)
                    # Hand range  : 15 - 150
                    # Volume Range : -65.25 - 0
                    vol = np.interp(length, [15,150],[minVol,maxVol])
                    print(int(length), vol)
                    volume.SetMasterVolumeLevel(vol, None)
                    if length<50:
                        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            elif ch==3:
                if len(lmList)!=0:
                    x1, y1 = lmList[8][1:]      #Index finger
                    x2 , y2 = lmList[12][1:]    #Middle Finger
                    # print(x1, y1, x2, y2)
                    plocX, plocY = 0,0
                    clocX, clocY = 0,0
                    frameR = 100
                    wScr , hScr = autopy.screen.size()
                    smoothening = 6
                    # 3. Check which fingers are up
                    fingers = detector.fingersUp()
                    # print(fingers)
                    cv2.rectangle(img, (frameR,frameR), (wCam-frameR, hCam-frameR), (255,0,255), 2)
                    # 4. Only Index Finger: Moving Mode
                    if fingers[1]==1 and fingers[2]==0:
                    # 5. Convert Coordinates
                        x3 = np.interp(x1 ,(frameR,wCam-frameR), (0, wScr))
                        y3 = np.interp(y1 ,(frameR,hCam-frameR ), (0, hScr))
                        # 6. Smoothen Values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening

                        plocX ,plocY = clocX ,clocY
                         # 7. Move Mouse
                        autopy.mouse.move(x3,y3)
                        cv2.circle(img , (x1, y1), 15, (255,0,255), 5)
                    
                    # 8. Both Index and Middle fingers are up: Clicking Mode
                    if fingers[1]==1 and fingers[2]==1:
                        # 9. Find distance between fingers
                        length, img, lineInfo = detector.findDistance(8, 12, img)
                        print(length)
                        # 10. Click mouse if distance is short
                        if length<40: 
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                            autopy.mouse.click()
            elif ch==4:
                q = 1
                break
            cv2.imshow("Live", img)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
                
        if q==1:
            break
        
if __name__== "__main__":
    main()