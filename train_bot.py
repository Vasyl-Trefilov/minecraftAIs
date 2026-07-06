import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision, regularizers

DATA_DIR = "pvp_dataset"
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")

IMG_H, IMG_W = 180, 320
FRAME_STACK = 4
BATCH_SIZE = 64
KEY_COLS = ['w', 'a', 's', 'd', 'l_click']
MOUSE_CLIP_PERCENTILE = 99.0
GPU_MEM_LIMIT_MB = 8196

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT_MB)]
        )
    except RuntimeError:
        pass

mixed_precision.set_global_policy('mixed_float16')

df = pd.read_csv(CSV_PATH).sort_values('frame').reset_index(drop=True)
columns_to_shift = KEY_COLS + ['mouse_dx', 'mouse_dy']
df[columns_to_shift] = df[columns_to_shift].shift(-2)
df = df.dropna().reset_index(drop=True)

image_paths = np.array([os.path.join(FRAMES_DIR, f) for f in df['frame'].values])
keyboard_labels = df[KEY_COLS].values.astype('float32')
mouse_raw = df[['mouse_dx', 'mouse_dy']].values.astype('float32')
N = len(df)

mouse_clip = float(np.percentile(np.abs(mouse_raw), MOUSE_CLIP_PERCENTILE))
mouse_clip = max(mouse_clip, 1.0)
print(f"Using mouse_clip = {mouse_clip:.2f}px (the {MOUSE_CLIP_PERCENTILE}th percentile of |delta|)")
mouse_raw = np.clip(mouse_raw, -mouse_clip, mouse_clip)
mouse_labels = mouse_raw / mouse_clip

with open("mouse_clip.txt", "w") as f:
    f.write(str(mouse_clip))

print("Pre-loading frames into RAM...")
all_images = np.zeros((N, IMG_H, IMG_W, 3), dtype=np.uint8)
for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_images[idx] = cv2.resize(img_rgb, (IMG_W, IMG_H))
print("RAM loading complete.")

with tf.device('/CPU:0'):
    all_images_tf = tf.Variable(all_images, trainable=False, dtype=tf.uint8)
    keyboard_labels_tf = tf.Variable(keyboard_labels, trainable=False, dtype=tf.float32)
    mouse_labels_tf = tf.Variable(mouse_labels, trainable=False, dtype=tf.float32)

valid_starts = np.arange(0, N - FRAME_STACK + 1)
chunk_size = 100
num_chunks = N // chunk_size
chunk_indices = np.arange(num_chunks)
np.random.seed(42)
np.random.shuffle(chunk_indices)
split_idx = int(0.85 * num_chunks)
train_chunks = set(chunk_indices[:split_idx])

train_starts_list, val_starts_list = [], []
for start in valid_starts:
    if (start // chunk_size) in train_chunks:
        train_starts_list.append(start)
    else:
        val_starts_list.append(start)
train_starts = np.array(train_starts_list, dtype=np.int32)
val_starts = np.array(val_starts_list, dtype=np.int32)

FLIP_KEY_PERM = tf.constant([0, 3, 2, 1, 4])
FLIP_MOUSE_SIGN = tf.constant([-1.0, 1.0])

def load_batch(start_batch):
    offsets = tf.range(FRAME_STACK)
    idx = start_batch[:, None] + offsets[None, :]
    imgs = tf.gather(all_images_tf, idx)
    imgs = tf.transpose(imgs, perm=[0, 2, 3, 1, 4])
    b = tf.shape(imgs)[0]
    imgs = tf.reshape(imgs, [b, IMG_H, IMG_W, 3 * FRAME_STACK])
    last_idx = start_batch + FRAME_STACK - 1
    k = tf.gather(keyboard_labels_tf, last_idx)
    m = tf.gather(mouse_labels_tf, last_idx)
    return imgs, (k, m)

def augment_batch(imgs, labels):
    k, m = labels
    b = tf.shape(imgs)[0]
    flip_mask = tf.random.uniform([b]) > 0.5
    flipped_imgs = tf.image.flip_left_right(imgs)
    imgs = tf.where(flip_mask[:, None, None, None], flipped_imgs, imgs)
    flipped_m = m * FLIP_MOUSE_SIGN
    m = tf.where(flip_mask[:, None], flipped_m, m)
    flipped_k = tf.gather(k, FLIP_KEY_PERM, axis=1)
    k = tf.where(flip_mask[:, None], flipped_k, k)
    return imgs, (k, m)

def make_dataset(starts, training):
    ds = tf.data.Dataset.from_tensor_slices(starts)
    if training:
        ds = ds.shuffle(buffer_size=len(starts), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.map(load_batch, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_dataset = make_dataset(train_starts, training=True)
val_dataset = make_dataset(val_starts, training=False)

def weighted_bce(pos_weight_vec):
    pw = tf.constant(pos_weight_vec)
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-(pw * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))
    return loss_fn

def make_mouse_loss(saturation_weight=0.05):
    def loss_fn(y_true, y_pred):
        weight = 1.0 + 1.5 * tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True)
        mse = tf.reduce_mean(weight * tf.square(y_true - y_pred))
        saturation_penalty = tf.reduce_mean(tf.square(y_pred) * tf.square(y_pred))
        return mse + saturation_weight * saturation_penalty
    return loss_fn

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

def conv_block(feats, filters, reg):
    feats = layers.Conv2D(filters, 3, padding='same', activation='gelu', kernel_regularizer=reg)(feats)
    feats = layers.Conv2D(filters, 3, strides=2, padding='same', activation='gelu', kernel_regularizer=reg)(feats)
    feats = layers.BatchNormalization()(feats)
    return feats

def build_chop_model():
    reg = regularizers.l2(1e-4)
    inputs = layers.Input(shape=(IMG_H, IMG_W, 3 * FRAME_STACK), dtype=tf.uint8)

    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.RandomZoom(height_factor=(-0.1, 0.0), width_factor=(-0.1, 0.0), fill_mode='constant')(x)
    x = layers.RandomBrightness(factor=0.15)(x)
    x = layers.RandomContrast(factor=0.15)(x)

    stem = layers.Conv2D(16, 3, strides=2, padding='same', activation='gelu', kernel_regularizer=reg)(x)
    stem = layers.BatchNormalization()(stem)

    xk = conv_block(stem, 24, reg)
    xk = conv_block(xk, 48, reg)
    xk = conv_block(xk, 96, reg)
    xk = conv_block(xk, 128, reg)
    xk = layers.GlobalAveragePooling2D()(xk)
    xk = layers.Dropout(0.3)(xk)
    keyboard_out = layers.Dense(len(KEY_COLS), activation='sigmoid', name='keyboard', dtype='float32')(xk)

    xm = AddCoords()(stem)
    xm = conv_block(xm, 32, reg)
    xm = conv_block(xm, 64, reg)
    xm = conv_block(xm, 128, reg)
    xm = conv_block(xm, 192, reg)
    heatmaps = layers.Conv2D(8, 1, activation=None, kernel_regularizer=reg)(xm)
    coords = SpatialSoftArgmax()(heatmaps)
    m_branch = layers.Dense(64, activation='gelu', kernel_regularizer=reg)(coords)
    mouse_out = layers.Dense(
        2, activation='tanh', name='mouse', dtype='float32',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
    )(m_branch)

    return Model(inputs=inputs, outputs=[keyboard_out, mouse_out])

model = build_chop_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
    loss={'keyboard': weighted_bce([1.7, 10.0, 10.0, 10.0, 1.0]), 'mouse': make_mouse_loss()},
    loss_weights={'keyboard': 1.0, 'mouse': 3.0},
    metrics={'keyboard': 'accuracy', 'mouse': 'mae'},
    jit_compile=True,
)
model.summary()

model.fit(train_dataset, validation_data=val_dataset, epochs=100)
model.save("chop_model.keras")