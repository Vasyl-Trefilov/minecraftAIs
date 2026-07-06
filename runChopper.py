import subprocess
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

MODEL_PATH = "chop_modelstatic.keras"
MOUSE_CLIP_PATH = "mouse_clip.txt"

IMG_H, IMG_W = 180, 320
FRAME_STACK = 4

EMA_ALPHA = 0.5
MAX_DX_PER_FRAME = 40.0
MAX_DY_PER_FRAME = 40.0
DEADZONE_PX = 1.5

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

@tf.keras.utils.register_keras_serializable()
class SpatialSoftArgmax(layers.Layer):
    def build(self, input_shape):
        h, w = input_shape[1], input_shape[2]
        y_coords = tf.linspace(-1.0, 1.0, h)
        x_coords = tf.linspace(-1.0, 1.0, w)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        self.x_grid = tf.reshape(tf.cast(x_grid, tf.float32), [1, h * w, 1])
        self.y_grid = tf.reshape(tf.cast(y_grid, tf.float32), [1, h * w, 1])

    def call(self, inputs):
        b = tf.shape(inputs)[0]
        h, w, c = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        flat = tf.reshape(inputs, [b, h * w, c])
        flat = tf.cast(flat, tf.float32)
        attn = tf.nn.softmax(flat, axis=1)
        exp_x = tf.reduce_sum(attn * self.x_grid, axis=1)
        exp_y = tf.reduce_sum(attn * self.y_grid, axis=1)
        return tf.concat([exp_x, exp_y], axis=-1)

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'AddCoords': AddCoords, 'SpatialSoftArgmax': SpatialSoftArgmax})

@tf.function
def infer(input_tensor):
    return model(input_tensor, training=False)

def send_mouse_delta(dx, dy):
    pass

def send_keys(w, a, s, d, l_click):
    pass

def capture_frame():
    result = subprocess.run(['grim', '-'], capture_output=True, timeout=1)
    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_W, IMG_H))

def main():
    frame_buffer = deque(maxlen=FRAME_STACK)
    for _ in range(FRAME_STACK):
        frame_buffer.append(capture_frame())

    ema_dx, ema_dy = 0.0, 0.0

    while True:
        t0 = time.time()

        frame_buffer.append(capture_frame())
        stacked_input = np.concatenate(list(frame_buffer), axis=-1)
        input_tensor = np.expand_dims(stacked_input, axis=0)

        keyboard_pred, mouse_pred = infer(input_tensor)
        keyboard_pred = keyboard_pred.numpy()[0]
        mouse_pred = mouse_pred.numpy()[0]

        raw_dx = float(mouse_pred[0]) * MOUSE_CLIP
        raw_dy = float(mouse_pred[1]) * MOUSE_CLIP

        ema_dx = EMA_ALPHA * raw_dx + (1 - EMA_ALPHA) * ema_dx
        ema_dy = EMA_ALPHA * raw_dy + (1 - EMA_ALPHA) * ema_dy

        dx = float(np.clip(ema_dx, -MAX_DX_PER_FRAME, MAX_DX_PER_FRAME))
        dy = float(np.clip(ema_dy, -MAX_DY_PER_FRAME, MAX_DY_PER_FRAME))

        if abs(dx) < DEADZONE_PX:
            dx = 0.0
        if abs(dy) < DEADZONE_PX:
            dy = 0.0

        send_mouse_delta(dx, dy)

        w, a, s, d, l_click = (keyboard_pred > 0.5).astype(int)
        send_keys(w, a, s, d, l_click)

if __name__ == "__main__":
    main()