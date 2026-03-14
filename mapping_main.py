import cv2, socket, struct, threading, requests, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse
from ultralytics import YOLO
from rl_brain import EKFSLAM # Ensure your EKFSLAM class is in rl_brain.py

# --- CONFIG ---
ROBOT_IP = "172.26.159.220"
DATA_PORT, VIDEO_PORT, CMD_SYNC_PORT = 9998, 9999, 9997
GRID_SIZE = 40

# --- INITIALIZATION ---
slam = EKFSLAM(start_x=400, start_y=400)
yolo = YOLO("yolov8n.pt")
landmark_thumbs = {}
path_history = [] # For the crystal clear trail line

lock = threading.Lock()
latest_frame = None
current_servo = 90
current_dist = 600.0
trigger_render = False # High-priority flag for movement animation

# --- HIGH PRIORITY ACTION LISTENER ---
def action_sync_listener():
    global trigger_render
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", CMD_SYNC_PORT))
    print(f"🔥 High-Priority Action Sync Active on Port {CMD_SYNC_PORT}")
    while True:
        try:
            data, _ = s.recvfrom(1024)
            action = struct.unpack("10s", data)[0].decode().strip()
            with lock:
                # One command = one distinct movement step on map
                slam.predict(action)
                # Store position for the path line
                path_history.append((slam.mu[0], slam.mu[1]))
                trigger_render = True # Tell main loop to redraw NOW
        except: pass

# --- DATA RECEIVERS ---
def video_receiver():
    global latest_frame
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", VIDEO_PORT))
    s.listen(1)
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
    global current_servo, current_dist
    dsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dsock.bind(("0.0.0.0", DATA_PORT))
    while True:
        try:
            data, _ = dsock.recvfrom(1024)
            d, s = struct.unpack("ff", data)
            with lock: 
                current_dist = d if d > 0 else 600.0
                current_servo = s
        except: pass

def draw_ellipse(ax, mu, P, color):
    vals, vecs = np.linalg.eigh(P)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 4 * np.sqrt(np.abs(vals))
    ell = Ellipse(xy=mu, width=w, height=h, angle=theta, edgecolor=color, fc='none', lw=1.5, alpha=0.6)
    ax.add_patch(ell)

# --- MAIN RENDERING LOOP ---
def main():
    global trigger_render
    threading.Thread(target=video_receiver, daemon=True).start()
    threading.Thread(target=data_receiver, daemon=True).start()
    threading.Thread(target=action_sync_listener, daemon=True).start()
    
    plt.ion(); fig, ax = plt.subplots(figsize=(10,10))
    # Initial path point
    path_history.append((slam.mu[0], slam.mu[1]))

    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            s_angle, d_val = current_servo, current_dist
            needs_redraw = trigger_render
            trigger_render = False # Reset flag

        if frame is not None:
            # 1. Perception (YOLO)
            results = yolo(frame, verbose=False, conf=0.5)
            if len(results[0].boxes) > 0 and d_val < 250:
                label = results[0].names[int(results[0].boxes.cls[0])]
                box = results[0].boxes.xyxy[0].cpu().numpy()
                phi = np.radians(s_angle - 90)
                is_new, thumb, name = slam.update(label, d_val, phi, frame, box)
                if is_new and thumb is not None:
                    landmark_thumbs[name] = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                    needs_redraw = True # Redraw if new object found

            # 2. Render ONLY if moved or new perception (Saves CPU for priority)
            ax.clear()
            ax.set_facecolor('#0a0a0f')
            ax.set_xlim(0, 800); ax.set_ylim(0, 800)

            # --- A. HEATMAP & PATH LINE ---
            rx, ry, rt = slam.mu[:3]
            # Draw Path Line (Crystal Clear Trail)
            if len(path_history) > 1:
                pts = np.array(path_history)
                ax.plot(pts[:,0], pts[:,1], color='cyan', lw=1, alpha=0.5, zorder=1)

            # --- B. ROBOT ANIMATION ---
            draw_ellipse(ax, (rx, ry), slam.P[:2, :2], 'cyan')
            # 20x40 Chassis
            rect = plt.Rectangle((rx-20, ry-10), 40, 20, angle=np.degrees(rt), 
                                 color='cyan', alpha=0.9, zorder=10)
            ax.add_patch(rect)
            
            # Short Directional Nose (No more long line)
            ax.plot([rx, rx + 25 * np.cos(rt)], [ry, ry + 25 * np.sin(rt)], color='white', lw=3, zorder=11)

            # --- C. LANDMARKS ---
            for name, idx in slam.landmark_map.items():
                lx, ly = slam.mu[idx], slam.mu[idx+1]
                draw_ellipse(ax, (lx, ly), slam.P[idx:idx+2, idx:idx+2], 'lime')
                if name in landmark_thumbs:
                    img_box = OffsetImage(landmark_thumbs[name], zoom=0.2)
                    ab = AnnotationBbox(img_box, (lx, ly), frameon=True, bboxprops=dict(edgecolor='lime', lw=2))
                    ax.add_artist(ab)

            plt.pause(0.001) # Keeps the UI responsive
            cv2.imshow("Robot Vision", results[0].plot())
        
        if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    main()