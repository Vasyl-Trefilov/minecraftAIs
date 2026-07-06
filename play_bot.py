import mss
import cv2
import time
import numpy as np
import tensorflow as tf
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import keyboard as kb_listener

FPS = 20
FRAME_TIME = 1.0 / FPS
CONFIDENCE_THRESHOLD = 0.5

print("Loading AI Model...")
model = tf.keras.models.load_model("chopbotv1stable.keras")
print("Model Loaded!")

mouse = MouseController()
keyboard = KeyboardController()

KEY_MAP = [
    'w', 'a', 's', 'd',
    Key.space, Key.shift,
    Button.left, Button.right
]

current_button_states = [False] * 8

sct = mss.mss()
monitor = sct.monitors[1]

print("\n" + "=" * 40)
print("BOT IS READY.")
print("Switch to Minecraft.")
print("PRESS AND HOLD 'ESC' TO EMERGENCY STOP")
print("Starting in 5 seconds...")
print("=" * 40)
time.sleep(5)

try:
    while True:
        start_time = time.time()

        if kb_listener.is_pressed('esc'):
            print("\nEMERGENCY STOP ACTIVATED")
            break

        img = sct.grab(monitor)

        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        img_resized = cv2.resize(img_bgr, (320, 180))
        img_normalized = img_resized.astype(np.float32) / 255.0

        input_tensor = np.expand_dims(img_normalized, axis=0)

        predictions = model(input_tensor, training=False)

        keyboard_pred = predictions[0].numpy()[0]
        mouse_pred = predictions[1].numpy()[0]

        for i, key_prob in enumerate(keyboard_pred):
            should_press = key_prob > CONFIDENCE_THRESHOLD

            if should_press and not current_button_states[i]:
                if type(KEY_MAP[i]) == Button:
                    mouse.press(KEY_MAP[i])
                else:
                    keyboard.press(KEY_MAP[i])
                current_button_states[i] = True

            elif not should_press and current_button_states[i]:
                if type(KEY_MAP[i]) == Button:
                    mouse.release(KEY_MAP[i])
                else:
                    keyboard.release(KEY_MAP[i])
                current_button_states[i] = False

        dx = int(mouse_pred[0] * 100)
        dy = int(mouse_pred[1] * 100)

        if abs(dx) > 0 or abs(dy) > 0:
            mouse.move(dx, dy)

        elapsed = time.time() - start_time
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    for i, is_pressed in enumerate(current_button_states):
        if is_pressed:
            if type(KEY_MAP[i]) == Button:
                mouse.release(KEY_MAP[i])
            else:
                keyboard.release(KEY_MAP[i])
    print("Keys released. Bot safely shut down.")