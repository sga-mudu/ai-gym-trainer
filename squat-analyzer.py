import cv2
import mediapipe as mp
import numpy as np

class SquatAnalyzer:
    def __init__(self):
        self.squat_phase = "standing"  # standing, descending, bottom, ascending
        self.feedback = []
        self.scores = []
        
        # Perfect squat standards (in degrees)
        self.ideal_knee_angle_at_bottom = 90  # 90° bend is perfect
        self.max_back_lean = 30  # Back shouldn't lean more than 30°
    
    def analyze_frame(self, landmarks):
        """Analyze one frame of squat video"""
        frame_feedback = []
        
        # 1. Get key points
        left_hip = get_landmark_coordinates(landmarks, 'left_hip')
        left_knee = get_landmark_coordinates(landmarks, 'left_knee')
        left_ankle = get_landmark_coordinates(landmarks, 'left_ankle')
        left_shoulder = get_landmark_coordinates(landmarks, 'left_shoulder')
        
        # 2. Calculate knee angle (hip-knee-ankle)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        # 3. Calculate back angle (shoulder-hip-knee)
        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        
        # 4. Check knee position relative to toes
        knee_over_toe = self.check_knee_over_toe(left_knee, left_ankle)
        
        # 5. Determine squat phase
        self.detect_squat_phase(knee_angle)
        
        # 6. Give feedback based on phase
        if self.squat_phase == "descending":
            if knee_angle < 100:  # Getting low
                if abs(180 - back_angle) > self.max_back_lean:
                    frame_feedback.append("⚠️ Keep chest up! Don't lean forward too much")
                
                if knee_over_toe > 0.3:  # Knee too far past toes
                    frame_feedback.append("👣 Keep knees behind toes!")
        
        elif self.squat_phase == "bottom":
            if knee_angle > 100:  # Not deep enough
                frame_feedback.append("⬇️ Go deeper! Aim for thighs parallel to ground")
            elif knee_angle < 80:  # Too deep
                frame_feedback.append("⬆️ That's very deep! Good if comfortable")
        
        # Store for overall analysis
        self.scores.append({
            'knee_angle': knee_angle,
            'back_angle': back_angle,
            'phase': self.squat_phase
        })
        
        return frame_feedback, knee_angle, back_angle
    
    def check_knee_over_toe(self, knee, ankle):
        """Check if knee is too far past toes"""
        # Simple check: if knee x-position is much bigger than ankle x-position
        return max(0, knee[0] - ankle[0])  # Positive = knee past toes
    
    def detect_squat_phase(self, knee_angle):
        """Detect what part of squat user is in"""
        if knee_angle > 160:
            self.squat_phase = "standing"
        elif knee_angle > 100:
            self.squat_phase = "descending"
        elif knee_angle < 100 and knee_angle > 80:
            self.squat_phase = "bottom"
        else:
            self.squat_phase = "ascending"
    
    def get_overall_feedback(self):
        """Give final feedback after full video"""
        if not self.scores:
            return ["No squats detected"]
        
        avg_knee_angle = np.mean([s['knee_angle'] for s in self.scores if s['phase'] == 'bottom'])
        
        feedback = []
        feedback.append(f"📊 **SQUAT ANALYSIS REPORT**")
        feedback.append(f"Average depth: {180 - avg_knee_angle:.1f}° bend")
        
        # Depth feedback
        if avg_knee_angle > 100:
            feedback.append("🎯 **Goal**: Go deeper! Aim for 90° knee bend")
        elif 80 < avg_knee_angle < 100:
            feedback.append("✅ **Great depth!** You're hitting parallel")
        else:
            feedback.append("🔥 **Excellent depth!** You're going below parallel")
        
        # Count reps
        rep_count = self.count_squat_reps()
        feedback.append(f"🔢 **Rep Count**: {rep_count} squats detected")
        
        return feedback
    
    def count_squat_reps(self):
        """Simple rep counter - counts bottoms"""
        bottoms = [s for s in self.scores if s['phase'] == 'bottom']
        
        # Group consecutive bottom frames as one rep
        rep_count = 0
        was_in_bottom = False
        
        for score in self.scores:
            if score['phase'] == 'bottom' and not was_in_bottom:
                rep_count += 1
                was_in_bottom = True
            elif score['phase'] != 'bottom':
                was_in_bottom = False
        
        return rep_count