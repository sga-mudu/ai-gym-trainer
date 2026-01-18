import cv2
import poseModule as pm
import time   

cap = cv2.VideoCapture('./photos/mandancing1.mp4')
prevTime = 0

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    if not success or img is None:
        break

    img = cv2.resize(img, (960, 540))

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) > 10:
        print(lmList[10])
        cv2.circle(img, (lmList[10][1], lmList[10][2]), 10,
                    (0, 255, 0), cv2.FILLED)

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