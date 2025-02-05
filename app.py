import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            try:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

               # Checks if fingers are open:
                thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # Distance between thumb and index finger
                distance_open = ((thumb.x - index_finger.x)**2 + (thumb.y - index_finger.y)**2)**0.5
                # Distance between thumb and index finger for "OK" symbol
                distance_ok = ((thumb.x - index_finger.x)**2 + (thumb.y - index_finger.y)**2)**0.05  
                
                if distance_ok < 0.05:  # Threshold for "OK" symbol gesture
                    
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW)  
                elif distance_open > 0.1:  
                
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)  
                else:
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            except Exception as e:
                print(f"Error while processing hand landmarks: {e}")
                continue  

    
    frame_resized = cv2.resize(frame, (1000, 800))  # Resize to 640x480

    
    cv2.imshow('Hand Gesture Effect', frame_resized)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    if cv2.getWindowProperty('Hand Gesture Effect', cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
