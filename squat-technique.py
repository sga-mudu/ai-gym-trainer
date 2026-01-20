import cv2
import numpy as np
import time
import squatModule as pm

cap = cv2.VideoCapture("./photos/squat-biology.mp4") # API this link

detector = pm.poseDetector()
count = 0
direction = 0  # 0: going down, 1: going up
prev_time = 0
squat_threshold = 90  # Angle threshold for counting a squat
knee_angles = []
hip_angles = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(img, (1280, 580))
    img = detector.findPose(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        # Calculate knee angles (hip-knee-ankle)
        # CORRECTED: For right knee, it should be 24-26-28 (hip-knee-ankle)
        # CORRECTED: For left knee, it should be 23-25-27 (hip-knee-ankle)

        # Left hip angle (shoulder-hip-knee)
        left_hip_angle = detector.findAngle(img, 11, 23, 25)

        # Right hip angle (shoulder-hip-knee)
        right_hip_angle = detector.findAngle(img, 12, 24, 26)
        
        # Left knee angle (hip-knee-ankle)
        left_knee_angle = detector.findAngle(img, 23, 25, 27)
        
        # Right knee angle (hip-knee-ankle)
        right_knee_angle = detector.findAngle(img, 24, 26, 28)
        
        # Calculate average knee angle
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Store recent angles for smoothing
        knee_angles.append(avg_knee_angle)
        hip_angles.append(avg_hip_angle)
        if len(knee_angles) > 5:  # Keep last 5 frames
            knee_angles.pop(0)

        if len(hip_angles) > 5:  # Keep last 5 frames
            hip_angles.pop(0)

        # Calculate smoothed angle
        smoothed_angle = np.mean(knee_angles) if knee_angles else avg_knee_angle
        smoothed_hip_angle = np.mean(hip_angles) if hip_angles else avg_hip_angle

        # Squat counter logic
        if smoothed_angle > 160:
            direction = 0  # Standing/going down
        elif smoothed_angle < 100 and direction == 0:
            direction = 1  # Reached bottom, start going up
            count += 1
            print(f"Squat #{count} completed!")
        

        # Display angles
        cv2.putText(img, f"avg hip angle: {int(smoothed_hip_angle)}", (50, 100), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, f"avg knee angle: {int(smoothed_angle)}", (50, 150), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        # Display squat count
        cv2.putText(img, f"Squats: {count}", (50, 200), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        
        print(smoothed_angle)
        # Display squat phase
        if smoothed_angle > 160:
            phase_text = "STANDING"
            color = (0, 255, 0) 
        elif smoothed_angle > 100:
            phase_text = "GOING UP/DOWN"
            color = (0, 255, 255)  
        elif smoothed_angle <= 100:
            phase_text = "BOTTOM - GO UP!"
            color = (0, 165, 255)  
        else:
            phase_text = "DEEP SQUAT"
            color = (0, 0, 255) 
        
        cv2.putText(img, phase_text, (img.shape[1] - 300, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        
        
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1] - 150, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
    
    cv2.imshow("AI Squat Trainer", img)
    
    # Press 'q' to quit, 'r' to reset count
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        count = 0
        print("Counter reset!")

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import time
# import poseModule as pm

# cap = cv2.VideoCapture("./photos/correct-squat.mp4")

# detector = pm.poseDetector()
# count = 0
# dir = 0
# prevTime = 0 


# while True:
#     success, img = cap.read()
#     # img = cv2.imread("./photos/sideplank.jpg")
#     img = cv2.resize(img, (1280, 580))
#     img = detector.findPose(img)
#     lmList = detector.findPosition(img, draw=False)
#     # print(lmList)

#     if len(lmList) != 0:
#         #left shoulder angle
#         detector.findAngle(img, 14, 12, 24)

#         #right shoulder angle
#         detector.findAngle(img, 13, 11, 23)

#         #left hip angle
#         detector.findAngle(img, 12, 24, 26)
#         #right hip angle
#         detector.findAngle(img, 25, 23, 11)

#         #left knee angle
#         detector.findAngle(img, 28, 26, 24)
#         #right knee angle
#         detector.findAngle(img, 27, 25, 23)

#         #right shoulder angle
#         # angle = detector.findAngle(img, 11, 13, 15)
#         # per = np.interp(angle, (200, 278), (0, 100))
#         # print(angle, per)

#         #check for the seated row
#         # if per == 100:
#         #     if dir == 0:
#         #         count += 0.5
#         #         dir = 1

#         # if per == 0:
#         #     if dir == 1:
#         #         count += 0.5
#         #         dir = 0

#         # print(count)
#         # cv2.putText(img, f'{int(count)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

#         cTime = time.time()
#         fps = 1 / (cTime - prevTime)
#         prevTime = cTime
#         cv2.putText(img, f'{int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

#     cv2.imshow("image", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break