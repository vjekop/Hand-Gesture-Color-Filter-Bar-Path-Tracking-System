import cv2
import numpy as np
import time
import math

video_path = r"C:\Users\sydne\OneDrive\Desktop\Olympic weightlifting tracker\liftvideo.mp4"


cap = cv2.VideoCapture("liftvideo.mp4") #you can put any type of video here like birds moving, cars moving.. and track them

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

window_width = 800
window_height = 800


previous_time = 0

ret, first_frame = cap.read()


first_frame_resized = cv2.resize(first_frame, (window_width, window_height))

prev_gray = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2GRAY)

# Select a point manually for tracking
point = cv2.selectROI("Select Barbell Point", first_frame_resized, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Barbell Point")


initial_point = (int(point[0] + point[2] / 2), int(point[1] + point[3] / 2))


corners = np.array([[initial_point]], dtype=np.float32)


mask = np.zeros_like(first_frame_resized)


paused = False

path_points = []

start_time = time.time()

while True:
    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached. Pausing...")
            paused = True  # Set to pause when video ends
            continue  

        resized_frame = cv2.resize(frame, (window_width, window_height))

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)

        good_new = next_points[status == 1]

        if good_new.shape[0] > 0:
            # coordinates of the first point
            new = good_new[0].ravel()  # coordinates of the new point
            new_point = (int(new[0]), int(new[1]))  

            path_points.append(new_point)

            for i in range(1, len(path_points)):
                cv2.line(resized_frame, path_points[i-1], path_points[i], (0, 255, 0), 2)

            resized_frame = cv2.circle(resized_frame, new_point, 5, (0, 0, 255), -1)

       
        cv2.imshow('Barbell Path Tracking', resized_frame)

        elapsed_time = time.time() - start_time
        delay = max(1, int(1000 / fps - elapsed_time * 1000))  # Delay based on FPS
        start_time = time.time()

        if cv2.waitKey(delay) & 0xFF == ord('q'):  
            print("Exiting...")
            break


        prev_gray = gray.copy()
        corners = good_new.reshape(-1, 1, 2)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  
        paused = not paused
        print(f"Video {'Paused' if paused else 'Resumed'}")

    
    if cv2.getWindowProperty('Barbell Path Tracking', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed. Exiting...")
        break


cap.release()
cv2.destroyAllWindows()
