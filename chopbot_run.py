import subprocess
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision

mixed_precision.set_global_policy('float32')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MODEL_PATH = "chopbot_v1_last.keras"
MOUSE_CLIP_PATH = "mouse_clip.txt"
IMG_H, IMG_W = 180, 320
FRAME_STACK = 4

EMA_ALPHA = 0.5
MAX_DX_PER_FRAME = 40.0
MAX_DY_PER_FRAME = 40.0
DEADZONE_PX = 1.5
MOVE_PROB_THRESHOLD = 0.5

with open(MOUSE_CLIP_PATH) as f:
    MOUSE_CLIP = float(f.read().strip())


@tf.keras.utils.register_keras_serializable()
class AddCoords(layers.Layer):
    def call(self, inputs):
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        batch_size = tf.shape(inputs)[0]
        y_coords = tf.linspace(-1.0, 1.0, h)
        x_coords = tf.linspace(-1.0, 1.0, w)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        x_grid = tf.cast(tf.tile(x_grid[None, :, :, None], [batch_size, 1, 1, 1]), inputs.dtype)
        y_grid = tf.cast(tf.tile(y_grid[None, :, :, None], [batch_size, 1, 1, 1]), inputs.dtype)
        return tf.concat([inputs, x_grid, y_grid], axis=-1)


model = tf.keras.models.load_model(
    MODEL_PATH, compile=False, custom_objects={'AddCoords': AddCoords}
)


@tf.function
def infer(input_tensor):
    return model(input_tensor, training=False)


from evdev import UInput, ecodes as e
KEY_MAP = {'w': e.KEY_W, 'a': e.KEY_A, 's': e.KEY_S, 'd': e.KEY_D, 'l_click': e.BTN_LEFT}
CAPABILITIES = {e.EV_KEY: list(KEY_MAP.values()) + [e.BTN_RIGHT], e.EV_REL: [e.REL_X, e.REL_Y]}
ui = UInput(CAPABILITIES, name="chopbot-virtual-input")
_prev_key_state = {k: 0 for k in KEY_MAP}


def send_mouse_delta(dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx != 0: ui.write(e.EV_REL, e.REL_X, dx)
    if dy != 0: ui.write(e.EV_REL, e.REL_Y, dy)
    ui.syn()


def send_keys(w, a, s, d, l_click):
    global _prev_key_state
    new_state = {'w': w, 'a': a, 's': s, 'd': d, 'l_click': l_click}
    changed = False
    for name, val in new_state.items():
        if val != _prev_key_state[name]:
            ui.write(e.EV_KEY, KEY_MAP[name], val)
            changed = True
    if changed: ui.syn()
    _prev_key_state = new_state


def capture_frame():
    for attempt in range(3):
        try:
            result = subprocess.run(['grim', '-'], capture_output=True, timeout=1)
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("grim returned an unparsable frame")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, (IMG_W, IMG_H))
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(0.02)


def main():
    frame_buffer = deque(maxlen=FRAME_STACK)
    for _ in range(FRAME_STACK):
        frame_buffer.append(capture_frame())
    ema_dx, ema_dy = 0.0, 0.0

    try:
        while True:
            t0 = time.time()
            frame_buffer.append(capture_frame())
            stacked_input = np.concatenate(list(frame_buffer), axis=-1)
            input_tensor = np.expand_dims(stacked_input, axis=0)  # [1, 180, 320, 12] uint8

            keyboard_pred, move_pred, mouse_pred = infer(input_tensor)
            keyboard_pred = keyboard_pred.numpy()[0]
            move_prob = float(move_pred.numpy()[0, 0])
            mouse_pred = mouse_pred.numpy()[0]

            if move_prob > MOVE_PROB_THRESHOLD:
                raw_dx = float(mouse_pred[0]) * MOUSE_CLIP
                raw_dy = float(mouse_pred[1]) * MOUSE_CLIP
            else:
                raw_dx = 0.0
                raw_dy = 0.0

            ema_dx = EMA_ALPHA * raw_dx + (1 - EMA_ALPHA) * ema_dx
            ema_dy = EMA_ALPHA * raw_dy + (1 - EMA_ALPHA) * ema_dy
            dx = float(np.clip(ema_dx, -MAX_DX_PER_FRAME, MAX_DX_PER_FRAME))
            dy = float(np.clip(ema_dy, -MAX_DY_PER_FRAME, MAX_DY_PER_FRAME))

            if abs(dx) < DEADZONE_PX: dx = 0.0
            if abs(dy) < DEADZONE_PX: dy = 0.0
            send_mouse_delta(dx, dy)

            w, a, s, d, l_click = (keyboard_pred > 0.5).astype(int)
            send_keys(w, a, s, d, l_click)
            print(f"step: {(time.time() - t0) * 1000:.1f}ms | keys: {w}{a}{s}{d}{l_click} | "
                  f"move_prob: {move_prob:.2f} | mouse: {dx:.0f},{dy:.0f}")
    except KeyboardInterrupt:
        pass
    finally:
        for name, code in KEY_MAP.items():
            if _prev_key_state.get(name, 0): ui.write(e.EV_KEY, code, 0)
        ui.syn()
        ui.close()


if __name__ == "__main__":
    main()