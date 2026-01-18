import cv2
import mediapipe as mp
import time
import math
from math import atan2, degrees
import numpy as np

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.upBody,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.results = None

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList
    
    # Update the findAngle method in poseModule.py
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        lmList = self.findPosition(img, draw=False)
        
        if len(lmList) <= max(p1, p2, p3):
            return 0
        
        # Get the coordinates
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        x3, y3 = lmList[p3][1:]
        
        # Calculate the angle using the law of cosines (more accurate)
        # Calculate vectors
        a = np.array([x1 - x2, y1 - y2])
        b = np.array([x3 - x2, y3 - y2])
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            angle = 0
        else:
            # Calculate cosine of angle
            cos_angle = dot_product / (norm_a * norm_b)
            # Handle floating point errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            # Calculate angle in degrees
            angle = np.degrees(np.arccos(cos_angle))
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)
            
            cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, (0, 0, 255), cv2.FILLED)
            
            # Draw angle arc
            cv2.putText(img, f"{int(angle)}°", (x2 - 20, y2 + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        
        return angle
    
    # def findAngle(self, img, p1, p2, p3, draw=True):
    #     # Get the landmarks
    #     lmList = self.findPosition(img, draw=False)


    #     # Get the coordinates
    #     x1, y1 = lmList[p1][1:]
    #     x2, y2 = lmList[p2][1:]
    #     x3, y3 = lmList[p3][1:]

    #     # Calculate the angle
    #     angle = degrees(
    #         (atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2))
    #     )
        
    #     if angle < 0:
    #         angle += 360

    #     if draw:
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #         cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)

    #         cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
    #         # cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
    #         cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
    #         # cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    #         cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
    #         # cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

    #         cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 30),
    #                     cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    #     return angle



# def main():
#     cap = cv2.VideoCapture('./photos/mandancing1.mp4')
#     prevTime = 0

#     detector = poseDetector()

#     while True:
#         success, img = cap.read()
#         if not success or img is None:
#             print("Bad frame, skipping...")
#             continue

#         img = cv2.resize(img, (960, 540))

#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) > 10:
#             print(lmList[10])
#             cv2.circle(img, (lmList[10][1], lmList[10][2]), 10,
#                        (0, 255, 0), cv2.FILLED)

#         curTime = time.time()
#         fps = 1 / (curTime - prevTime)
#         prevTime = curTime

#         cv2.putText(img, str(int(fps)), (10, 70),
#                     cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

#         cv2.imshow("Pose Estimation", img)

#         if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
#             break

#     cap.release()
#     cv2.destroyAllWindows()





# if __name__ == "__main__":
#     main()



