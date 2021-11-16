import cv2 as cv
import mediapipe as mp

def countFingers(frame, results, draw = True):
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
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

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
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP-2].x
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x< thumb_mcp_x)) or (hand_label=="Left" and (thumb_tip_x> thumb_mcp_x)):
            fingers_statuses[hand_label.upper()+"_THUMB"] = True

            count[hand_label.upper()] += 1
# Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:
        cv.putText(output_image, "Total Fingers: ",(10, 25), cv.FONT_HERSHEY_COMPLEX , 1, (20, 255, 155), 2)
        cv.putText(output_image, str(sum(count.values())), (width//2-150,240), cv.FONT_HERSHEY_SIMPLEX,
                    8.9, (20,255,155), 10, 10)

    return output_image, fingers_statuses, count

cap = cv.VideoCapture(0)        # get a video capture object for the camera
cap.set(3,1200)

mp_hands = mp.solutions.hands   # Initialize the mediapipe hands class.
mp_drawing = mp.solutions.drawing_utils     # Initialize the mediapipe drawing class.

with mp_hands.Hands(min_detection_confidence = 0.5, max_num_hands = 2, min_tracking_confidence = 0.5) as hands:  
    while True:
        _ , frame = cap.read()
        frame = cv.flip(frame,1)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        output_image = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                        thickness=2, circle_radius=2),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                        thickness=2, circle_radius=2))
            frame , fingers_statuses, count = countFingers(output_image, results)
        cv.imshow("Live" , frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv.destroyAllWindows()