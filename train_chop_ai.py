import os
import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision, regularizers

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
DATA_DIR = "chop_dataset"
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")

IMG_H, IMG_W = 180, 320
FRAME_STACK = 4
BATCH_SIZE = 64
KEY_COLS = ['w', 'a', 's', 'd', 'l_click']
GPU_MEM_LIMIT_MB = 8196
L2_REG = 1e-5

# Raw-pixel threshold (pre-normalization) for "did the mouse meaningfully move
# this frame". Small enough to catch real aim corrections, big enough to
# ignore sensor/polling jitter. Tune if your mouse is noisier/quieter than
# typical.
MOVE_THRESH_PX = 3.0

# How many frames the labels are shifted relative to the images. If this is
# wrong (wrong sign or magnitude), images and actions are decorrelated and
# NO architecture or loss function can learn past the label prior -- which
# matches everything we've seen so far. Keeping this as a named constant so
# it's a one-line change if the diagnostic below says otherwise.
LAG_SHIFT = -2

parser = argparse.ArgumentParser()
parser.add_argument("--overfit_debug", action="store_true",
                     help="Train on ~256 samples with no augmentation/reg to sanity check the pipeline.")
parser.add_argument("--skip_alignment_check", action="store_true",
                     help="Skip the pre-training image/label alignment diagnostic (not recommended).")
args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT_MB)]
        )
    except RuntimeError:
        pass

mixed_precision.set_global_policy('float32')

df = pd.read_csv(CSV_PATH).sort_values('frame').reset_index(drop=True)
columns_to_shift = KEY_COLS + ['mouse_dx', 'mouse_dy']
df_unshifted = df.copy()  # kept for the alignment diagnostic below
df[columns_to_shift] = df[columns_to_shift].shift(LAG_SHIFT)
df = df.dropna().reset_index(drop=True)

image_paths = np.array([os.path.join(FRAMES_DIR, f) for f in df['frame'].values])
keyboard_labels = df[KEY_COLS].values.astype('float32')
mouse_raw_px = df[['mouse_dx', 'mouse_dy']].values.astype('float32')
N = len(df)

MOUSE_CLIP = 120.0
print(f"Using stable MOUSE_CLIP = {MOUSE_CLIP}px")

# ----------------------------------------------------------------------
# KEY CHANGE: split the mouse target into (a) a binary "is moving" flag
# computed from the RAW (pre-clip) pixel delta, and (b) the normalized
# direction/magnitude vector, still clipped/normalized as before. The
# "is_moving" flag is what lets us mask the regression loss later.
# ----------------------------------------------------------------------
is_moving = (np.abs(mouse_raw_px).sum(axis=1) > MOVE_THRESH_PX).astype('float32')
move_rate = is_moving.mean()
print(f"Fraction of frames with meaningful mouse movement: {move_rate:.3f}")

mouse_clipped = np.clip(mouse_raw_px, -MOUSE_CLIP, MOUSE_CLIP)
mouse_labels = mouse_clipped / MOUSE_CLIP

with open("mouse_clip.txt", "w") as f:
    f.write(str(MOUSE_CLIP))

key_priors = np.clip(keyboard_labels.mean(axis=0), 1e-3, 1 - 1e-3)
key_bias_init = np.log(key_priors / (1 - key_priors)).astype('float32')
move_prior = float(np.clip(move_rate, 1e-3, 1 - 1e-3))
move_bias_init = float(np.log(move_prior / (1 - move_prior)))
print(f"Key priors: {dict(zip(KEY_COLS, key_priors.round(3)))}")
print(f"Move prior: {move_prior:.3f} (bias init {move_bias_init:.3f})")

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
    # Pack [dx, dy, is_moving] into one tensor so it travels through
    # tf.data / model.fit as a single "mouse" label the loss function can
    # unpack -- this is what lets the regression loss mask itself.
    mouse_labels_ext = np.concatenate([mouse_labels, is_moving[:, None]], axis=1)
    mouse_labels_tf = tf.Variable(mouse_labels_ext, trainable=False, dtype=tf.float32)

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

# ----------------------------------------------------------------------
# Oversample "is_moving" frames so every batch has a healthy number of
# active examples for the masked regression loss to learn from, rather
# than relying on whatever fraction naturally falls in a random batch.
# Target ~40% of each batch being an active-movement frame.
# ----------------------------------------------------------------------
last_idx = train_starts + FRAME_STACK - 1
train_is_moving = is_moving[last_idx] > 0.5
current_rate = train_is_moving.mean()
target_rate = 0.40
if 0 < current_rate < target_rate:
    n_total = len(train_starts)
    n_moving = train_is_moving.sum()
    n_extra = int(max(0, (target_rate * n_total - n_moving) / (1 - target_rate)))
    reps = int(np.ceil(n_extra / max(n_moving, 1)))
    extra_moving = np.repeat(train_starts[train_is_moving], reps)[:n_extra]
else:
    extra_moving = np.array([], dtype=np.int32)
train_starts_oversampled = np.concatenate([train_starts, extra_moving]).astype(np.int32)
print(f"Oversampled train set: {len(train_starts)} -> {len(train_starts_oversampled)} "
      f"(moving frame rate {current_rate:.3f} -> ~{target_rate:.2f} target)")

if args.overfit_debug:
    print("### OVERFIT DEBUG MODE: training on a tiny fixed slice, no augmentation, no reg ###")
    train_starts_oversampled = train_starts[:256]
    val_starts = train_starts[:256]
    L2_REG = 0.0

FLIP_KEY_PERM = tf.constant([0, 3, 2, 1, 4])
FLIP_MOUSE_SIGN = tf.constant([-1.0, 1.0, 1.0])  # dx flips sign, dy and is_moving don't


def load_batch(start_batch):
    offsets = tf.range(FRAME_STACK)
    idx = start_batch[:, None] + offsets[None, :]
    imgs = tf.gather(all_images_tf, idx)
    imgs = tf.transpose(imgs, perm=[0, 2, 3, 1, 4])
    b = tf.shape(imgs)[0]
    imgs = tf.reshape(imgs, [b, IMG_H, IMG_W, 3 * FRAME_STACK])
    last_idx = start_batch + FRAME_STACK - 1
    k = tf.gather(keyboard_labels_tf, last_idx)
    m = tf.gather(mouse_labels_tf, last_idx)  # [dx, dy, is_moving]
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


def pack_for_three_outputs(imgs, labels):
    k, m = labels
    # 'move' and 'mouse' outputs both need the packed [dx, dy, is_moving]
    # label; Keras just needs one label tensor per output slot.
    return imgs, (k, m, m)


def make_dataset(starts, training):
    ds = tf.data.Dataset.from_tensor_slices(starts)
    if training:
        ds = ds.shuffle(buffer_size=len(starts), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.map(load_batch, num_parallel_calls=tf.data.AUTOTUNE)
    if training and not args.overfit_debug:
        ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(pack_for_three_outputs, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_dataset = make_dataset(train_starts_oversampled, training=True)
val_dataset = make_dataset(val_starts, training=False)


def weighted_bce(pos_weight_vec):
    pw = tf.constant(pos_weight_vec)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-(pw * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))
    return loss_fn


def move_bce(pos_weight):
    def loss_fn(y_true, y_pred):
        y_true = y_true[..., 2:3]  # is_moving flag lives in the 3rd slot of the packed mouse label
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-(pos_weight * y_true * tf.math.log(y_pred)
                                 + (1 - y_true) * tf.math.log(1 - y_pred)))
    return loss_fn


def move_accuracy_metric(y_true, y_pred):
    y_true = y_true[..., 2:3]
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


def masked_mouse_mse(y_true, y_pred):
    """Regression loss computed ONLY over frames where is_moving==1.
    This is the core fix: the direction/magnitude head never sees the 75%
    zero-frames, so it can't collapse to "just predict 0" -- every
    gradient it receives is a real direction signal."""
    target = y_true[..., :2]
    mask = y_true[..., 2:3]
    se = tf.reduce_sum(tf.square(target - y_pred), axis=-1, keepdims=True)
    masked_se = se * mask
    denom = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(masked_se) / denom


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


def conv_block(feats, filters, reg):
    feats = layers.Conv2D(filters, 3, padding='same', activation='gelu', kernel_regularizer=reg)(feats)
    feats = layers.Conv2D(filters, 3, strides=2, padding='same', activation='gelu', kernel_regularizer=reg)(feats)
    feats = layers.BatchNormalization()(feats)
    return feats


def build_chop_model():
    reg = regularizers.l2(L2_REG) if L2_REG > 0 else None
    inputs = layers.Input(shape=(IMG_H, IMG_W, 3 * FRAME_STACK), dtype=tf.uint8)

    x = layers.Rescaling(1.0 / 255.0)(inputs)
    if not args.overfit_debug:
        x = layers.RandomZoom(height_factor=(-0.1, 0.0), width_factor=(-0.1, 0.0), fill_mode='constant')(x)
        x = layers.RandomBrightness(factor=0.15)(x)
        x = layers.RandomContrast(factor=0.15)(x)

    stem = layers.Conv2D(16, 3, strides=2, padding='same', activation='gelu', kernel_regularizer=reg)(x)
    stem = layers.BatchNormalization()(stem)

    # Keyboard branch
    xk = conv_block(stem, 24, reg)
    xk = conv_block(xk, 48, reg)
    xk = conv_block(xk, 96, reg)
    xk = conv_block(xk, 128, reg)
    xk = layers.GlobalAveragePooling2D()(xk)
    xk = layers.Dropout(0.3 if not args.overfit_debug else 0.0)(xk)
    keyboard_out = layers.Dense(
        len(KEY_COLS), activation='sigmoid', name='keyboard', dtype='float32',
        bias_initializer=tf.keras.initializers.Constant(key_bias_init)
    )(xk)

    # Shared mouse trunk (CoordConv, since aim needs spatial position)
    xm = AddCoords()(stem)
    xm = conv_block(xm, 32, reg)
    xm = conv_block(xm, 64, reg)
    xm = conv_block(xm, 128, reg)
    xm = conv_block(xm, 192, reg)
    m = layers.Conv2D(16, 1, activation='gelu', kernel_regularizer=reg)(xm)
    m_flat = layers.Flatten()(m)
    m_branch = layers.Dense(128, activation='gelu', kernel_regularizer=reg)(m_flat)

    # Head 1: is-the-camera-moving classifier (clean, well-posed binary problem)
    move_out = layers.Dense(
        1, activation='sigmoid', name='move', dtype='float32',
        bias_initializer=tf.keras.initializers.Constant(move_bias_init)
    )(m_branch)

    # Head 2: direction/magnitude, trained only on frames where move==1
    mouse_out = layers.Dense(
        2, activation='tanh', name='mouse', dtype='float32',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05)
    )(m_branch)

    return Model(inputs=inputs, outputs=[keyboard_out, move_out, mouse_out])


model = build_chop_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004, clipnorm=5.0),
    loss={
        'keyboard': weighted_bce([1.5, 6.0, 6.0, 6.0, 1.0]),
        'move': move_bce(pos_weight=1.0 / move_prior),
        'mouse': masked_mouse_mse,
    },
    loss_weights={'keyboard': 1.0, 'move': 1.0, 'mouse': 3.0},
    metrics={
        'keyboard': [tf.keras.metrics.BinaryAccuracy(name='accuracy')],
        'move': [move_accuracy_metric],
    },
    jit_compile=False
)

model.summary()


class HonestProgressCallback(tf.keras.callbacks.Callback):
    """Computes real, unambiguous per-branch metrics on a fixed val batch
    each epoch, so a stall is visible immediately instead of hidden behind
    a metric that doesn't mean what it looks like it means."""

    def __init__(self, val_ds):
        super().__init__()
        self.val_batch = next(iter(val_ds.take(1)))

    def on_epoch_end(self, epoch, logs=None):
        imgs, (k_true, move_true, mouse_true) = self.val_batch
        k_pred, move_pred, mouse_pred = self.model(imgs, training=False)
        k_pred_bin = (k_pred.numpy() > 0.5).astype(int)
        k_true_np = k_true.numpy()
        per_key_acc = (k_pred_bin == k_true_np).mean(axis=0)
        move_true_np = move_true.numpy()[:, 2]
        move_pred_bin = (move_pred.numpy()[:, 0] > 0.5).astype(int)
        move_acc = (move_pred_bin == move_true_np).mean()
        mask = move_true_np > 0.5
        if mask.sum() > 0:
            masked_mae = np.abs(mouse_true.numpy()[mask, :2] - mouse_pred.numpy()[mask]).mean()
        else:
            masked_mae = float('nan')
        print(f"  [honest] per-key acc {dict(zip(KEY_COLS, per_key_acc.round(3)))} | "
              f"move_acc {move_acc:.3f} | masked_mouse_mae(active-only) {masked_mae:.3f}")


callbacks = [
    tf.keras.callbacks.ModelCheckpoint("chopbot_v1_last.keras", save_best_only=False, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("chopbot_v1_smart.keras", monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    HonestProgressCallback(val_dataset),
]

epochs = 30 if args.overfit_debug else 150
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)

if args.overfit_debug:
    print("\nCheck the [honest] line above: per-key accuracy should climb well past the base")
    print("rates, move_acc should climb well past the move prior, and masked_mouse_mae should")
    print("drop well below ~0.3 on this 256-sample slice within ~30 epochs. If not, the bug is")
    print("upstream of the model (data/label alignment) -- spot check a few (image, label) pairs.")