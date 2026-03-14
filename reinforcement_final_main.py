import cv2, socket, struct, threading, requests, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse, Wedge
from ultralytics import YOLO

# --- CRITICAL IMPORTS ---
from rl_brain import EKFSLAM, IntelligentQNet

# --- CONFIGURATION ---
ROBOT_IP = "10.137.181.220"  # <--- VERIFY YOUR ROBOT IP
DATA_PORT, VIDEO_PORT = 9998, 9999
MAP_SIZE_PIXELS = 800
CAMERA_FOV_DEG = 60.0
CAMERA_FOV_RAD = np.radians(CAMERA_FOV_DEG)

# --- LOAD AI BRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntelligentQNet(sensor_dim=7, action_dim=4).to(device)

try:
    model.load_state_dict(torch.load("intelligent_robot.pth", map_location=device))
    print("✅ AI Brain Loaded (7-Input Mode)")
except Exception as e:
    print(f"❌ Brain Load Error: {e}")
    exit()

model.eval()

# --- COCO CLASS FILTER (Indoor Objects Only) ---
TARGET_CLASSES = [39, 41, 56, 57, 58, 60, 62, 63, 67, 73, 74, 75]

# --- STATE TRACKING (For AI Inputs) ---
class RobotState:
    def __init__(self):
        self.coverage = np.zeros((20, 20)) 
        self.steps_since_new_tile = 0
        self.grid_size = MAP_SIZE_PIXELS / 20.0 

    def update_and_get_sensors(self, rx, ry, rt, total_steps):
        gx = int(np.clip(rx // self.grid_size, 0, 19))
        gy = int(np.clip(ry // self.grid_size, 0, 19))

        if self.coverage[gx, gy] == 0:
            self.coverage[gx, gy] = 1; self.steps_since_new_tile = 0
        else: self.steps_since_new_tile += 1

        local_visited = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= gx+i < 20 and 0 <= gy+j < 20:
                    local_visited += self.coverage[gx+i, gy+j]
        
        return torch.FloatTensor([[
            rx / MAP_SIZE_PIXELS, ry / MAP_SIZE_PIXELS, np.cos(rt), np.sin(rt),
            (total_steps % 1000) / 800.0, local_visited / 25.0, min(1.0, self.steps_since_new_tile / 100.0)
        ]]).to(device)

# --- INITIALIZATION ---
slam = EKFSLAM(start_x=MAP_SIZE_PIXELS/2, start_y=MAP_SIZE_PIXELS/2)
robot_state = RobotState()
yolo = YOLO("yolov8s.pt") # Small model for better accuracy

lock = threading.Lock()
latest_frame = None
current_dist = -1.0
step_counter = 0 
path_history = []
wall_points = []
landmark_thumbs = {}
vis_mu = slam.mu[:3].copy() # For smooth animation
SMOOTH_FACTOR = 0.15

# --- NETWORKING THREADS ---
def video_receiver():
    global latest_frame
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try: s.bind(("0.0.0.0", VIDEO_PORT)); s.listen(1)
    except: return
    print("✅ Video Thread Bound")
    conn, _ = s.accept()
    data = b""; payload_size = struct.calcsize(">L")
    while True:
        try:
            while len(data) < payload_size: data += conn.recv(8192)
            msg_size = struct.unpack(">L", data[:payload_size])[0]
            data = data[payload_size:]
            while len(data) < msg_size: data += conn.recv(8192)
            frame = cv2.imdecode(np.frombuffer(data[:msg_size], np.uint8), 1)
            data = data[msg_size:]
            if frame is not None: 
                with lock: latest_frame = frame
        except: break

def data_receiver():
    global current_dist
    dsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: dsock.bind(("0.0.0.0", DATA_PORT))
    except: return
    print("✅ Data Thread Bound")
    while True:
        try:
            data, _ = dsock.recvfrom(1024)
            if len(data) >= 4:
                d = struct.unpack("f", data[:4])[0]
                if d > 0: 
                    with lock: current_dist = d
        except: pass

# --- VISUALIZATION HELPERS ---
def draw_ellipse(ax, mu, P, color):
    vals, vecs = np.linalg.eigh(P)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 4 * np.sqrt(np.maximum(vals, 1e-9))
    ell = Ellipse(xy=mu, width=w, height=h, angle=theta, edgecolor=color, fc='none', lw=2, zorder=2)
    ax.add_patch(ell)

def get_action_from_brain(frame, slam_obj, steps):
    resized = cv2.resize(frame, (160, 120))
    gray = np.mean(resized, axis=2) / 255.0
    img_t = torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0).to(device)
    rx, ry, rt = slam_obj.mu[:3]
    sensor_t = robot_state.update_and_get_sensors(rx, ry, rt, steps)
    with torch.no_grad(): action_idx = torch.argmax(model(img_t, sensor_t)).item()
    return ["forward", "left", "right", "backward"][action_idx]

# --- MAIN LOOP ---
def main():
    global step_counter, vis_mu
    threading.Thread(target=video_receiver, daemon=True).start()
    threading.Thread(target=data_receiver, daemon=True).start()
    
    plt.ion(); fig, ax_map = plt.subplots(figsize=(10,10))
    path_history.append((slam.mu[0], slam.mu[1]))

    print("--- AUTONOMOUS MAPPING STARTED ---")

    while True:
        with lock:
            frame, d_val = latest_frame, current_dist
            # Animation Smoothing
            diff_theta = (slam.mu[2] - vis_mu[2] + np.pi) % (2*np.pi) - np.pi
            vis_mu[0] += (slam.mu[0] - vis_mu[0]) * SMOOTH_FACTOR
            vis_mu[1] += (slam.mu[1] - vis_mu[1]) * SMOOTH_FACTOR
            vis_mu[2] += diff_theta * SMOOTH_FACTOR
            rx, ry, rt = vis_mu
        
        if frame is not None:
            display_frame = frame.copy()
            step_counter += 1
            h, w = frame.shape[:2]

            # 1. AI DECISION
            action = get_action_from_brain(frame, slam, step_counter)
            print(f"🤖 {action.upper()} | Dist: {d_val:.0f} | Stag: {robot_state.steps_since_new_tile}")

            try:
                duration = 0.25 if action in ['left', 'right'] else 0.4
                requests.post(f"http://{ROBOT_IP}:5000/{action}?time={duration}", timeout=0.1)
                slam.predict(action) 
                path_history.append((slam.mu[0], slam.mu[1]))
            except: pass

            # 2. MAPPING (YOLO + Wall Detection)
            if step_counter % 3 == 0: 
                results = yolo(frame, verbose=False, conf=0.65, classes=TARGET_CLASSES)
                target_box, best_label, min_center_dist = None, None, float('inf')
                found_obj = False

                for r in results:
                    for i, box in enumerate(r.boxes.xyxy):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        cx = (x1 + x2) / 2
                        dist_from_center = abs(cx - (w / 2))
                        if dist_from_center < 120:
                            found_obj = True
                            if dist_from_center < min_center_dist:
                                min_center_dist, target_box = dist_from_center, (x1, y1, x2, y2)
                                best_label = r.names[int(r.boxes.cls[i])]

                if target_box is not None and (10 < d_val < 350):
                    pixel_offset = (w / 2) - ((target_box[0] + target_box[2]) / 2)
                    phi = (pixel_offset / w) * CAMERA_FOV_RAD
                    is_new, thumb, name = slam.update(best_label, d_val, phi, frame, target_box)
                    
                    if is_new and thumb is not None:
                        landmark_thumbs[name] = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                    
                    x1, y1, x2, y2 = map(int, target_box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                elif not found_obj and (10 < d_val < 200): # Wall Detection
                    wx = rx + d_val * np.cos(rt)
                    wy = ry + d_val * np.sin(rt)
                    is_duplicate = False
                    for wx_old, wy_old in wall_points:
                        if np.hypot(wx - wx_old, wy - wy_old) < 15: is_duplicate = True; break
                    if not is_duplicate: wall_points.append((wx, wy))

            # 3. PROFESSIONAL RENDERING (Matches your image)
            ax_map.clear()
            ax_map.set_facecolor('#1e1e1e') # Dark background
            
            # Auto-Scale Map
            all_x = [p[0] for p in path_history] + [p[0] for p in wall_points] + [rx]
            all_y = [p[1] for p in path_history] + [p[1] for p in wall_points] + [ry]
            if all_x:
                margin = 150
                ax_map.set_xlim(min(all_x)-margin, max(all_x)+margin)
                ax_map.set_ylim(min(all_y)-margin, max(all_y)+margin)

            # Layer 1: Path (Cyan Line)
            if len(path_history) > 1:
                pts = np.array(path_history)
                ax_map.plot(pts[:,0], pts[:,1], color='cyan', lw=1, alpha=0.5, zorder=1)

            # Layer 2: Walls (Gray Squares)
            if wall_points:
                wx_arr, wy_arr = zip(*wall_points)
                ax_map.scatter(wx_arr, wy_arr, c='gray', marker='s', s=15, alpha=0.7, zorder=2)

            # Layer 3: Landmarks (Floating Images + Green Dots)
            drawn_positions = [] 
            for name, idx in slam.landmark_map.items():
                lx, ly = slam.mu[idx], slam.mu[idx+1]
                
                # Check for visual overlap to prevent clutter
                overlap = False
                for (dx, dy) in drawn_positions:
                    if np.hypot(lx-dx, ly-dy) < 50: overlap = True; break
                
                if not overlap:
                    draw_ellipse(ax_map, (lx, ly), slam.P[idx:idx+2, idx:idx+2], 'lime')
                    
                    if name in landmark_thumbs:
                        img_box = OffsetImage(landmark_thumbs[name], zoom=0.2)
                        ab = AnnotationBbox(img_box, (lx, ly), xybox=(0, 40), xycoords='data',
                                            boxcoords="offset points", frameon=True,
                                            bboxprops=dict(edgecolor='lime', lw=1),
                                            arrowprops=dict(arrowstyle="->", color="lime"))
                        ab.set_zorder(3) 
                        ax_map.add_artist(ab)
                    
                    # Label below object
                    ax_map.text(lx, ly - 20, name.split('_')[0].upper(), color='white', fontsize=8, 
                               ha='center', fontweight='bold', zorder=4,
                               bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))
                    drawn_positions.append((lx, ly))

            # Layer 4: Robot (Top Layer)
            wedge_start = np.degrees(rt) - (CAMERA_FOV_DEG / 2)
            wedge_end = np.degrees(rt) + (CAMERA_FOV_DEG / 2)
            fov_wedge = Wedge((rx, ry), 80, wedge_start, wedge_end, color='white', alpha=0.1, zorder=9)
            ax_map.add_patch(fov_wedge)

            robot_circle = plt.Circle((rx, ry), 12, color='white', fill=True, zorder=10)
            ax_map.add_patch(robot_circle)
            ax_map.plot([rx, rx + 25 * np.cos(rt)], [ry, ry + 25 * np.sin(rt)], color='red', lw=2, zorder=10)

            plt.pause(0.001)
            cv2.imshow("Robot AI View", display_frame)
        
        if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    main()