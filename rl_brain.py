import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntelligentQNet(nn.Module):
    def __init__(self, sensor_dim=7, action_dim=4):
        super(IntelligentQNet, self).__init__()
        
        # 1. Vision Branch (The Eyes)
        # Input: (1, 160, 120) -> Output: Flattened Features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 2. Sensor Branch (The Inner Ear/GPS)
        # We process sensors separately now to give them more weight
        self.sensor_fc = nn.Linear(sensor_dim, 64)
        
        # 3. Combined Branch
        # Calculation: 
        # Conv1: (160-5)/2 + 1 = 78, (120-5)/2 + 1 = 58
        # Conv2: (78-5)/2 + 1 = 37, (58-5)/2 + 1 = 27
        # Flatten: 32 channels * 37 * 27 = 31968
        # Combined Input: 31968 (Vision) + 64 (Sensors)
        self.fc_input = nn.Linear(31968 + 64, 256)
        
        # 4. Dueling Heads (Better stability for obstacle avoidance)
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, action_dim)

    def forward(self, img, sensors):
        x_img = self.conv(img)
        x_sens = F.relu(self.sensor_fc(sensors)) # Boost sensor importance
        
        combined = torch.cat([x_img, x_sens], dim=1)
        x = F.relu(self.fc_input(combined))
        
        val = self.value_head(x)
        adv = self.advantage_head(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))

class EKFSLAM:
    def __init__(self, start_x=400, start_y=400):
        # Robot State: [x, y, theta]
        self.mu = np.array([float(start_x), float(start_y), np.pi / 2])
        
        # Uncertainty Matrix
        self.P = np.diag([1.0, 1.0, 0.1]) 
        self.landmark_map = {} 
        
        # Tuned Noise (LOWERED for Slow/Precise Movement)
        self.Q = np.diag([2.0, 2.0, 0.01])   # Motion noise (Lower = More confident in physics)
        self.R = np.diag([10.0, 0.05])       # Measurement noise

    def predict(self, action):
        """Updates robot pose based on dead reckoning."""
        # --- TUNED PHYSICS FOR SLOW & ACCURATE MOVEMENT ---
        v = 2.0       # Reduced from 6.0 (Slow crawl)
        omega = 0.2   # Reduced from 0.5 (Precise turns)

        theta = self.mu[2]
        
        if action == 'forward':
            self.mu[0] += v * np.cos(theta)
            self.mu[1] += v * np.sin(theta)
        elif action == 'backward':
            self.mu[0] -= v * np.cos(theta)
            self.mu[1] -= v * np.sin(theta)
        elif action == 'left': 
            self.mu[2] += omega
        elif action == 'right': 
            self.mu[2] -= omega

        # Normalize Angle to -pi to pi
        self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi
        
        # Increase uncertainty after moving
        self.P[:3, :3] += self.Q

    def update(self, label, r_meas, phi_meas, frame, box):
        """EKF Update for landmarks."""
        rx, ry, rt = self.mu[:3]
        
        global_angle = rt + phi_meas
        lx_est = rx + r_meas * np.cos(global_angle)
        ly_est = ry + r_meas * np.sin(global_angle)

        found_idx = -1
        min_dist = 80.0 

        # Data Association
        for name, idx in self.landmark_map.items():
            if name.startswith(label):
                dist = np.hypot(self.mu[idx] - lx_est, self.mu[idx+1] - ly_est)
                if dist < min_dist:
                    found_idx = idx
                    min_dist = dist 

        landmark_name = ""

        if found_idx == -1:
            # --- NEW LANDMARK ---
            idx = len(self.mu)
            count = sum(1 for k in self.landmark_map if k.startswith(label))
            landmark_name = f"{label}_{count+1}"
            
            self.landmark_map[landmark_name] = idx
            self.mu = np.append(self.mu, [lx_est, ly_est])
            
            # Expand P matrix
            size = len(self.P)
            new_P = np.zeros((size + 2, size + 2))
            new_P[:size, :size] = self.P
            new_P[size:, size:] = np.diag([100.0, 100.0]) 
            self.P = new_P

            # Crop Thumbnail (Optional utility)
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            thumbnail = frame[y1:y2, x1:x2]
            return True, thumbnail, landmark_name
        
        else:
            # --- UPDATE EXISTING ---
            landmark_name = [name for name, i in self.landmark_map.items() if i == found_idx][0]
            lx, ly = self.mu[found_idx], self.mu[found_idx+1]
            
            delta_x = lx - rx
            delta_y = ly - ry
            q = delta_x**2 + delta_y**2
            r_expected = np.sqrt(q)
            phi_expected = np.arctan2(delta_y, delta_x) - rt
            phi_expected = (phi_expected + np.pi) % (2 * np.pi) - np.pi

            H = np.zeros((2, len(self.mu)))
            H[0, 0] = -delta_x / r_expected
            H[0, 1] = -delta_y / r_expected
            H[0, 2] = 0
            H[1, 0] = delta_y / q
            H[1, 1] = -delta_x / q
            H[1, 2] = -1
            H[0, found_idx] = delta_x / r_expected
            H[0, found_idx+1] = delta_y / r_expected
            H[1, found_idx] = -delta_y / q
            H[1, found_idx+1] = delta_x / q

            z = np.array([r_meas, phi_meas])
            h_x = np.array([r_expected, phi_expected])
            y_res = z - h_x
            y_res[1] = (y_res[1] + np.pi) % (2 * np.pi) - np.pi

            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.mu = self.mu + K @ y_res
            I = np.eye(len(self.mu))
            self.P = (I - K @ H) @ self.P
            
            return False, None, landmark_name