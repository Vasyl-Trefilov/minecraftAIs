import subprocess
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key, Listener
import evdev
from evdev import ecodes as e

FPS = 20
FRAME_TIME = 1.0 / FPS
CONFIDENCE_THRESHOLD = 0.5

DEBUG_DIR = "tmpscreens"
os.makedirs(DEBUG_DIR, exist_ok=True)

@tf.keras.utils.register_keras_serializable()
class AddCoords(tf.keras.layers.Layer):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h = 180
        w = 320

        y_coords = tf.linspace(-1.0, 1.0, h)
        x_coords = tf.linspace(-1.0, 1.0, w)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)

        x_grid = tf.expand_dims(tf.expand_dims(x_grid, axis=0), axis=-1)
        y_grid = tf.expand_dims(tf.expand_dims(y_grid, axis=0), axis=-1)

        x_grid = tf.tile(x_grid, [batch_size, 1, 1, 1])
        y_grid = tf.tile(y_grid, [batch_size, 1, 1, 1])

        return tf.concat([inputs, x_grid, y_grid], axis=-1)

print("Loading AI Model...")
model = tf.keras.models.load_model("chop_bot_v1.keras", compile=False)
print("Model Loaded!")

FRAME_STACK = 4
frame_buffer = []

try:
    cap = {
        e.EV_REL: [e.REL_X, e.REL_Y],
        e.EV_KEY: [
            e.BTN_LEFT,
            e.KEY_W, e.KEY_A, e.KEY_S, e.KEY_D
        ]
    }
    ui = evdev.UInput(cap, name='virtual-mouse-keyboard')
    print("Virtual input device created successfully using uinput.")
except Exception as uinput_err:
    print(f"Warning: Could not create uinput device ({uinput_err}). Falling back to pynput.")
    ui = None

mouse = MouseController()
keyboard_out = KeyboardController()

KEY_MAP = ['w', 'a', 's', 'd', Button.left]
EVDEV_KEY_MAP = [
    e.KEY_W,
    e.KEY_A,
    e.KEY_S,
    e.KEY_D,
    e.BTN_LEFT
]
current_button_states = [False] * 5

emergency_stop = False

def on_press(key):
    global emergency_stop
    if key == Key.esc:
        emergency_stop = True
        return False

listener = Listener(on_press=on_press)
listener.start()

print("\n" + "=" * 40)
print("BOT IS READY.")
print("Switch to Minecraft.")
print("PRESS 'ESC' TO EMERGENCY STOP")
print("Starting in 5 seconds...")
print("=" * 40)
time.sleep(4)

print("Initializing frame buffer...")
result = subprocess.run(['grim', '-'], capture_output=True, timeout=1)
img_array = np.frombuffer(result.stdout, dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
if img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (320, 180))
else:
    img_resized = np.zeros((180, 320, 3), dtype=np.uint8)

for _ in range(FRAME_STACK):
    frame_buffer.append(img_resized)

time.sleep(1)
frame_count = 0
try:
    while True:
        start_time = time.time()

        if emergency_stop:
            print("\nEMERGENCY STOP ACTIVATED")
            break

        result = subprocess.run(['grim', '-'], capture_output=True, timeout=1)
        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("ERROR: grim failed to capture screen")
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (320, 180))

        frame_buffer.append(img_resized)
        if len(frame_buffer) > FRAME_STACK:
            frame_buffer.pop(0)

        stacked_input = np.concatenate(frame_buffer, axis=-1)
        input_tensor = np.expand_dims(stacked_input, axis=0)

        predictions = model(input_tensor, training=False)
        keyboard_pred = predictions[0].numpy()[0]
        mouse_pred = predictions[1].numpy()[0]

        for i, key_prob in enumerate(keyboard_pred):
            should_press = key_prob > CONFIDENCE_THRESHOLD

            if should_press and not current_button_states[i]:
                if ui is not None:
                    ui.write(e.EV_KEY, EVDEV_KEY_MAP[i], 1)
                    ui.syn()
                else:
                    if type(KEY_MAP[i]) == Button:
                        mouse.press(KEY_MAP[i])
                    else:
                        keyboard_out.press(KEY_MAP[i])
                current_button_states[i] = True

            elif not should_press and current_button_states[i]:
                if ui is not None:
                    ui.write(e.EV_KEY, EVDEV_KEY_MAP[i], 0)
                    ui.syn()
                else:
                    if type(KEY_MAP[i]) == Button:
                        mouse.release(KEY_MAP[i])
                    else:
                        keyboard_out.release(KEY_MAP[i])
                current_button_states[i] = False

        MOUSE_SCALE = 120
        raw_dx = mouse_pred[0] * MOUSE_SCALE
        raw_dy = mouse_pred[1] * MOUSE_SCALE

        DEADZONE = 0.0
        if abs(raw_dx) < DEADZONE:
            raw_dx = 0
        if abs(raw_dy) < DEADZONE:
            raw_dy = 0

        SENSITIVITY = 1.0
        dx = int(np.round(raw_dx * SENSITIVITY))
        dy = int(np.round(raw_dy * SENSITIVITY))

        if dx != 0 or dy != 0:
            if ui is not None:
                ui.write(e.EV_REL, e.REL_X, dx)
                ui.write(e.EV_REL, e.REL_Y, dy)
                ui.syn()
            else:
                mouse.move(dx, dy)

        key_labels = ['w', 'a', 's', 'd', 'L']
        key_names = ''.join(key_labels[i] if current_button_states[i] else ' _' for i in range(5))
        if frame_count % 10 == 0:
            print(f"Mouse DX: {dx}, DY: {dy} | Keys: {key_names[:6]} | Click: {current_button_states[4]}")

        if frame_count % 30 == 0:
            debug = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            cv2.putText(debug, f"dx={dx} dy={dy}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            button_str = f"w:{int(current_button_states[0])} a:{int(current_button_states[1])} s:{int(current_button_states[2])} d:{int(current_button_states[3])}"
            cv2.putText(debug, button_str, (5, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            click_str = f"click:{int(current_button_states[4])}"
            cv2.putText(debug, click_str, (5, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_{frame_count:06d}.jpg"), debug)

        frame_count += 1

        elapsed = time.time() - start_time
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    for i, is_pressed in enumerate(current_button_states):
        if is_pressed:
            if ui is not None:
                ui.write(e.EV_KEY, EVDEV_KEY_MAP[i], 0)
                ui.syn()
            else:
                if type(KEY_MAP[i]) == Button:
                    mouse.release(KEY_MAP[i])
                else:
                    keyboard_out.release(KEY_MAP[i])
    if ui is not None:
        ui.close()
    print("Keys released. Bot safely shut down.")
