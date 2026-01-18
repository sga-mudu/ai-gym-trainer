# squat_coach_windows.py - FIXED FOR WINDOWS!
import cv2
import mediapipe as mp
import numpy as np
import os
import sys

print("="*60)
print("🤸 AI SQUAT COACH - Windows Version")
print("="*60)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================== FIXED PATH HANDLING ==================
def fix_windows_path(path):
    """Fix path for Windows"""
    # Remove ./ at start if present
    if path.startswith('./'):
        path = path[2:]
    
    # Replace forward slashes with backslashes for Windows
    path = path.replace('/', '\\')
    
    # If it's a relative path, make it absolute
    if not os.path.isabs(path):
        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    
    return path

# ================== ANALYSIS FUNCTIONS ==================
def calculate_angle(a, b, c):
    """Calculate angle between three points a, b, c (b is middle)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

class SquatAnalyzer:
    def __init__(self):
        self.squat_phase = "START"
        self.rep_count = 0
        self.depth_history = []
        self.feedback_log = []
        self.in_squat = False
        self.max_depth = 180  # Start with standing position
        
    def analyze_pose(self, landmarks):
        """Analyze current pose for squat form"""
        feedback = []
        
        try:
            # Get key points (using RIGHT side for consistency)
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            # Calculate angles
            knee_angle = calculate_angle(hip, knee, ankle)
            torso_angle = calculate_angle(shoulder, hip, knee)
            
            # Determine squat phase
            if knee_angle > 160:
                phase = "STANDING"
                if self.in_squat:
                    self.in_squat = False
                    if self.max_depth < 100:  # Completed a rep
                        self.rep_count += 1
                        feedback.append(f"REP {self.rep_count} COMPLETE!")
            elif knee_angle < 100:
                phase = "BOTTOM"
                self.in_squat = True
                self.max_depth = min(self.max_depth, knee_angle)
            else:
                phase = knee_angle > 130 and "DESCENDING" or "ASCENDING"
                self.in_squat = True
            
            self.squat_phase = phase
            
            # Form checks
            knee_over_toe = knee[0] - ankle[0]
            if knee_over_toe > 0.1 and phase in ["DESCENDING", "BOTTOM"]:
                feedback.append("KNEES TOO FAR FORWARD")
            
            if torso_angle < 60 and phase != "STANDING":
                feedback.append("TOO MUCH FORWARD LEAN")
            elif torso_angle > 100:
                feedback.append("CHEST UP! DON'T LEAN BACK")
            
            if phase == "BOTTOM":
                if knee_angle > 90:
                    feedback.append("GO DEEPER! (Aim < 90°)")
                elif knee_angle < 70:
                    feedback.append("GREAT DEPTH!")
                else:
                    feedback.append("GOOD DEPTH!")
            
            return {
                'knee_angle': knee_angle,
                'torso_angle': torso_angle,
                'phase': phase,
                'feedback': feedback[:3]
            }
            
        except:
            return {
                'knee_angle': 0,
                'torso_angle': 0,
                'phase': 'ERROR',
                'feedback': ['No person detected']
            }

# ================== MAIN FUNCTION ==================
def main():
    print("📁 Checking for video files...")
    print(f"Current directory: {os.getcwd()}")
    print("\nFiles in this folder:")
    
    # List all files
    files = os.listdir('.')
    for i, file in enumerate(files):
        if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"  🎬 {file}")
        else:
            print(f"  📄 {file}")
    
    # Ask for video path
    print("\n" + "="*40)
    video_path = input("./photos/squat-biology.mp4").strip()
    
    if not video_path:
        print("❌ No video file entered. Exiting.")
        return
    
    # Fix the path for Windows
    video_path = fix_windows_path(video_path)
    print(f"Looking for: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ ERROR: File not found!")
        print(f"Tried to find: {video_path}")
        print("\n🔧 Solutions:")
        print("1. Put video in SAME folder as this script")
        print("2. Use just filename (not path)")
        print("3. Check spelling")
        return
    
    print(f"✅ Found video! Size: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB")
    
    # Process the video
    process_squat_video(video_path)

def process_squat_video(video_path):
    """Process the video with analysis"""
    print(f"\n🎬 Processing: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    print("🎮 Press 'q' to quit, 'p' to pause, SPACE for screenshot")
    
    # Create output
    output_path = "squat_analysis_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    analyzer = SquatAnalyzer()
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            analysis = {'knee_angle': 0, 'torso_angle': 0, 'phase': 'NO PERSON', 'feedback': []}
            
            if results.pose_landmarks:
                # Draw pose
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Analyze
                analysis = analyzer.analyze_pose(results.pose_landmarks.landmark)
                
                # Draw angles
                if analysis['knee_angle'] > 0:
                    # Draw knee angle
                    knee_point = (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * width),
                                 int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * height))
                    cv2.putText(image, f"{analysis['knee_angle']:.0f}°", 
                               (knee_point[0] + 20, knee_point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # ========== DISPLAY ANALYSIS ==========
            # Info box
            cv2.rectangle(image, (10, 10), (400, 200), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (400, 200), (255, 255, 255), 2)
            
            y = 40
            cv2.putText(image, "AI SQUAT COACH", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(image, f"Phase: {analysis['phase']}", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25
            cv2.putText(image, f"Knee: {analysis['knee_angle']:.1f}°", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25
            cv2.putText(image, f"Torso: {analysis['torso_angle']:.1f}°", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 25
            cv2.putText(image, f"Reps: {analyzer.rep_count}", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
            
            # Feedback
            for i, fb in enumerate(analysis['feedback']):
                cv2.putText(image, f"• {fb}", (20, y + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # Progress
            cv2.putText(image, f"Frame: {frame_count}/{total_frames}", (width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(image)
        
        # Show
        cv2.imshow(f'🤸 Squat Analysis: {os.path.basename(video_path)}', image)
        
        # Controls
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            cv2.imwrite(f'screenshot_{frame_count}.jpg', image)
            print(f"📸 Saved screenshot_{frame_count}.jpg")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*60)
    print("📊 FINAL REPORT")
    print("="*60)
    print(f"Frames analyzed: {frame_count}")
    print(f"Squat reps: {analyzer.rep_count}")
    print(f"Best depth: {180 - analyzer.max_depth:.1f}°")
    print(f"✅ Analysis saved: {output_path}")

# ================== RUN IT ==================
if __name__ == "__main__":
    main()