import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('./photos/mandancing1.mp4')
prevTime = 0

while True:
    success, img = cap.read()

    # If frame is empty: stop
    if not success:
        print("Failed to read frame. Check video path or codec.")
        break

    # Resize frame safely
    img = cv2.resize(img, (960, 540))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
    # Highlight specific landmarks (e.g., nose)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Example: Highlight the nose (landmark 0)
            if id == 12:
                cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Pose Estimation", img)


    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
