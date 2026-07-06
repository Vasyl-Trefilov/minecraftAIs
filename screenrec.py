import subprocess
import cv2
import time
import os
import numpy as np
import pandas as pd
from pynput import mouse, keyboard

FPS = 20
FRAME_TIME = 1.0 / FPS
DATA_DIR = "pvp_dataset"
FRAMES_DIR = os.path.join(DATA_DIR, "frames")

os.makedirs(FRAMES_DIR, exist_ok=True)

keys_pressed = {'w': 0, 'a': 0, 's': 0, 'd': 0, 'space': 0, 'shift': 0}
mouse_state = {'left': 0, 'right': 0, 'delta_x': 0, 'delta_y': 0}

def on_press(key):
    try:
        k = key.char.lower()
    except AttributeError:
        k = key.name.lower()
    if k in keys_pressed:
        keys_pressed[k] = 1

def on_release(key):
    try:
        k = key.char.lower()
    except AttributeError:
        k = key.name.lower()
    if k in keys_pressed:
        keys_pressed[k] = 0

def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        mouse_state['left'] = int(pressed)
    if button == mouse.Button.right:
        mouse_state['right'] = int(pressed)

last_x, last_y = None, None
center_x, center_y = None, None

def on_move(x, y):
    global last_x, last_y, center_x, center_y
    if center_x is not None and center_y is not None:
        if x == center_x and y == center_y:
            last_x, last_y = x, y
            return
            
    if last_x is not None and last_y is not None:
        mouse_state['delta_x'] += (x - last_x)
        mouse_state['delta_y'] += (y - last_y)
    last_x, last_y = x, y

keyboard.Listener(on_press=on_press, on_release=on_release).start()
mouse.Listener(on_move=on_move, on_click=on_click).start()

print("Recording in 3 seconds... Switch to Minecraft!")
time.sleep(3)

from pynput.mouse import Controller as MouseController
temp_mouse = MouseController()
center_x, center_y = temp_mouse.position
print(f"Detected screen/window center at: ({center_x}, {center_y})")
print("Warp events to this coordinate will be filtered out to prevent data corruption.")

csv_path = os.path.join(DATA_DIR, "labels.csv")
existing_df = None
start_frame_count = 0

if os.path.exists(csv_path):
    try:
        existing_df = pd.read_csv(csv_path)
        if len(existing_df) > 0:
            last_frame_name = existing_df['frame'].iloc[-1]
            last_num = int(last_frame_name.split('_')[1].split('.')[0])
            start_frame_count = last_num + 1
            print(f"Found existing dataset with {len(existing_df)} entries. Resuming from frame {start_frame_count:06d}...")
    except Exception as e_csv:
        print(f"Error reading existing labels.csv: {e_csv}. Starting from scratch.")
        start_frame_count = 0

dataset_log = []
frame_count = start_frame_count

try:
    print("RECORDING (Press Ctrl+C in terminal to stop)")
    while True:
        start_time = time.time()

        result = subprocess.run(['grim', '-'], capture_output=True, timeout=1)
        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_np is None:
            print("ERROR: grim failed to capture screen")
            break
        img_np = cv2.resize(img_np, (320, 180))

        frame_name = f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(os.path.join(FRAMES_DIR, frame_name), img_np)

        dataset_log.append({
            "frame": frame_name,
            "w": keys_pressed['w'], "a": keys_pressed['a'], "s": keys_pressed['s'], "d": keys_pressed['d'],
            "space": keys_pressed['space'], "shift": keys_pressed['shift'],
            "l_click": mouse_state['left'], "r_click": mouse_state['right'],
            "mouse_dx": mouse_state['delta_x'], "mouse_dy": mouse_state['delta_y']
        })

        mouse_state['delta_x'] = 0
        mouse_state['delta_y'] = 0
        frame_count += 1

        elapsed = time.time() - start_time
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

except KeyboardInterrupt:
    print("Stopped recording. Saving data...")
    df = pd.DataFrame(dataset_log)
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset. Total frames: {len(df)} (Added {frame_count - start_frame_count} new frames)")
