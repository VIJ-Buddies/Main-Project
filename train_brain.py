import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import collections
import os
from rl_brain import IntelligentQNet, EKFSLAM

# --- CONFIGURATION FOR FAST TRAINING ---
SCREEN_SIZE = 600
GRID_SIZE = 30  
BATCH_SIZE = 128        # Increased to learn from more examples at once
GAMMA = 0.99
EPSILON_DECAY = 0.99    # Aggressive Decay (Stops exploring randomly much faster)
TARGET_UPDATE_FREQ = 5  # Update brain more often (Was 10)
TOTAL_EPOCHS = 500      # Reduced from 2000 (Faster Results)
SENSOR_DIM = 7

# --- HYPERPARAMETERS ---
MAX_STEPS_PER_EPISODE = 1500  # Reduced slightly to prevent wasting time on bad runs
STUCK_THRESHOLD = 200         # Give up faster if stuck to start a new, better episode

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, img, sens, act, rew, n_img, n_sens, done):
        self.buffer.append((img, sens, act, rew, n_img, n_sens, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        img, sens, act, rew, n_img, n_sens, done = zip(*batch)
        return (torch.cat(img), torch.cat(sens), torch.LongTensor(act), 
                torch.FloatTensor(rew), torch.cat(n_img), torch.cat(n_sens), 
                torch.FloatTensor(done))

class IntelligentEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()
        self.reset()

    def _generate_grid_room(self):
        grid_map = np.zeros((20, 20), dtype=int)
        obstacles = []
        # Outer Walls
        for i in range(20):
            grid_map[i, 0] = 1; grid_map[i, 19] = 1
            grid_map[0, i] = 1; grid_map[19, i] = 1
        # Random Obstacles
        for _ in range(25): 
            ox, oy = random.randint(2, 17), random.randint(2, 17)
            grid_map[ox, oy] = 1
        # Convert to pixels
        for x in range(20):
            for y in range(20):
                if grid_map[x, y] == 1:
                    cx, cy = x * GRID_SIZE + 15, y * GRID_SIZE + 15
                    obstacles.append([cx, cy])
        return obstacles, grid_map

    def reset(self):
        self.obstacles, self.grid_map = self._generate_grid_room()
        while True:
            sx, sy = random.randint(2, 17), random.randint(2, 17)
            if self.grid_map[sx, sy] == 0:
                pixel_x, pixel_y = sx * GRID_SIZE + 15, sy * GRID_SIZE + 15
                self.slam = EKFSLAM(pixel_x, pixel_y)
                break
        self.coverage = np.zeros((20, 20))
        self.step_count = 0
        self.steps_since_new_tile = 0
        self.total_covered = 0
        return self._get_obs()

    def _get_obs(self):
        cam_surf = pygame.Surface((160, 120))
        rx, ry, rt = self.slam.mu[:3]
        for o in self.obstacles:
            dx, dy = o[0]-rx, o[1]-ry
            dist = np.hypot(dx, dy)
            angle = (np.arctan2(dy, dx) - rt + np.pi) % (2*np.pi) - np.pi
            if dist < 250 and abs(angle) < 0.9: 
                tx = 80 + (angle/0.9) * 80
                size = max(5, int(50 - (dist/5)))
                pygame.draw.circle(cam_surf, (255, 255, 255), (int(tx), 60), size)
        img_arr = pygame.surfarray.array3d(cam_surf)
        gray = np.mean(img_arr, axis=2) / 255.0
        img_t = torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0)
        
        gx, gy = int(rx//GRID_SIZE), int(ry//GRID_SIZE)
        local_visited = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= gx+i < 20 and 0 <= gy+j < 20:
                    local_visited += self.coverage[gx+i, gy+j]
        density = local_visited / 25.0
        stag_val = min(1.0, self.steps_since_new_tile / 100.0)
        sensor_v = torch.FloatTensor([[rx/SCREEN_SIZE, ry/SCREEN_SIZE, np.cos(rt), np.sin(rt), 
                                       self.step_count/MAX_STEPS_PER_EPISODE, density, stag_val]])
        return img_t, sensor_v

    def step(self, action):
        self.step_count += 1
        self.steps_since_new_tile += 1
        actions = ['forward', 'left', 'right', 'backward']
        self.slam.predict(actions[action])
        rx, ry, rt = self.slam.mu[:3]
        
        reward = 0
        done = False
        gx, gy = int(rx // GRID_SIZE), int(ry // GRID_SIZE)
        
        if not (0 <= gx < 20 and 0 <= gy < 20):
            reward = -100.0; done = True
        elif self.grid_map[gx, gy] == 1:
            reward = -100.0; done = True
        else:
            # --- INTELLIGENT REWARDS ---
            if action == 0: 
                reward += 0.5   
            elif action in [1, 2]: 
                reward -= 0.1   
            elif action == 3: 
                reward -= 2.0   

            if self.coverage[gx, gy] == 0:
                reward += 50.0 
                self.coverage[gx, gy] = 1
                self.steps_since_new_tile = 0
                self.total_covered += 1
            else:
                reward -= 0.2 

            if self.steps_since_new_tile > STUCK_THRESHOLD:
                reward -= 100 
                done = True
            
            free_cells = 400 - np.sum(self.grid_map)
            if self.total_covered > (free_cells * 0.90):
                reward += 500
                print(f">>> ROOM CLEARED! ({self.total_covered}/{int(free_cells)}) <<<")
                done = True

        if self.step_count > MAX_STEPS_PER_EPISODE: done = True
        return self._get_obs(), reward, done

    def render(self):
        self.screen.fill((20, 20, 30))
        for x in range(20):
            for y in range(20):
                if self.grid_map[x, y] == 1: 
                    pygame.draw.rect(self.screen, (100, 50, 50), (x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE))
                elif self.coverage[x, y] == 1: 
                    pygame.draw.rect(self.screen, (0, 50, 0), (x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        rx, ry, rt = self.slam.mu[:3]
        pygame.draw.circle(self.screen, (0, 255, 255), (int(rx), int(ry)), 12)
        pygame.draw.line(self.screen, (255, 255, 0), (rx, ry), (rx + 25*np.cos(rt), ry + 25*np.sin(rt)), 2)
        pygame.display.flip()

def save_model(model, filename="intelligent_robot.pth"):
    print(f"\nSaving model to {filename}...")
    torch.save(model.state_dict(), filename)
    print("Save Complete.")

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Uses your Mac's GPU!
    else:
        device = torch.device("cpu")    
    print(f"Training on {device} | Fast Mode: {TOTAL_EPOCHS} Epochs")
    
    env = IntelligentEnv()
    policy_net = IntelligentQNet(sensor_dim=SENSOR_DIM).to(device)
    target_net = IntelligentQNet(sensor_dim=SENSOR_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Slightly higher LR for faster convergence
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005) 
    memory = ReplayBuffer()
    epsilon = 1.0
    MODEL_PATH = "intelligent_robot.pth"

    if os.path.exists(MODEL_PATH):
        print(f"\nFound existing model: {MODEL_PATH}")
        choice = input("Do you want to (C)ontinue training or start (N)ew? ").lower()
        if choice == 'c':
            try:
                policy_net.load_state_dict(torch.load(MODEL_PATH))
                target_net.load_state_dict(policy_net.state_dict())
                epsilon = 0.3 
                print(">>> Model Loaded. Resuming...")
            except: print(">>> Error loading. Starting New.")

    try:
        for e in range(TOTAL_EPOCHS):
            img, sens = env.reset()
            total_rew = 0
            
            for t in range(MAX_STEPS_PER_EPISODE):
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT: raise KeyboardInterrupt

                if random.random() < epsilon:
                    action = np.random.choice([0, 1, 2, 3], p=[0.4, 0.25, 0.25, 0.1])
                else:
                    with torch.no_grad():
                        action = policy_net(img.to(device), sens.to(device)).argmax().item()

                (n_img, n_sens), rew, done = env.step(action)
                memory.push(img, sens, action, rew, n_img, n_sens, done)

                if len(memory.buffer) > BATCH_SIZE:
                    b_img, b_sens, b_act, b_rew, bn_img, bn_sens, b_done = memory.sample(BATCH_SIZE)
                    b_img, b_sens = b_img.to(device), b_sens.to(device)
                    b_act, b_rew = b_act.to(device), b_rew.to(device)
                    bn_img, bn_sens, b_done = bn_img.to(device), bn_sens.to(device), b_done.to(device)

                    current_q = policy_net(b_img, b_sens).gather(1, b_act.unsqueeze(1))
                    next_act = policy_net(bn_img, bn_sens).argmax(1, keepdim=True)
                    next_q = target_net(bn_img, bn_sens).gather(1, next_act).squeeze().detach()
                    target_q = b_rew + (GAMMA * next_q * (1 - b_done))

                    loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                img, sens = n_img, n_sens
                total_rew += rew
                
                # Render less frequently to speed up training loop
                if t % 50 == 0: env.render() 
                if done: break

            epsilon = max(0.05, epsilon * EPSILON_DECAY)
            
            if e % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"Ep: {e} | Eps: {epsilon:.2f} | Rew: {total_rew:.1f} | Tiles: {env.total_covered}")
                torch.save(policy_net.state_dict(), MODEL_PATH)

    except KeyboardInterrupt:
        save_model(policy_net, MODEL_PATH)
        return

    save_model(policy_net, MODEL_PATH)

if __name__ == "__main__":
    train()