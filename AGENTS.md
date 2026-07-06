# Minecraft PvP AI Bot

Imitation learning: record human gameplay → train TF model → run bot.

## Setup

```bash
source venv/bin/activate
```

Python 3.11 virtual env with all deps (TF, OpenCV, pynput, mss, pandas, numpy).

## Pipeline

### 1. Record data — `screenrec.py`

Captures screen via `grim` (NOT `mss` — this runs under **Hyprland/Wayland**, `mss` returns black frames).

- Captures frames at `320×180` → saved to `pvp_dataset/frames/`
- Mouse delta accumulated between consecutive `on_move` events (pixel deltas, not center-relative)
- Label CSV at `pvp_dataset/labels.csv` with columns:
  `frame,w,a,s,d,space,shift,l_click,r_click,mouse_dx,mouse_dy`
- Key/Label mapping: `['w','a','s','d','space','shift','left_click','right_click']`
- Press Ctrl+C to stop

### 2. Train — `train_bot.py`

```bash
python train_bot.py
```

Multi-task CNN: 3 Conv2D blocks → 2 Dense heads. Details:
- **Input**: `180×320×3` normalized to `[0,1]`
- **Keyboard head**: 8-unit `sigmoid` (independent multi-label), loss=`binary_crossentropy`
- **Mouse head**: 2-unit `tanh` (bounded to `[-1,1]`), loss=`mse`
- **Mouse targets**: raw deltas clipped to `±300`, then divided by `300` → `[-1,1]`
- Batch size 32, 100 epochs, Adam `lr=0.0005`
- Saves to `pvp_bot_v1.keras`

### 3. Run bot — `aiPvpPlay.py`

```bash
python aiPvpPlay.py
```

- Uses `grim` for screen capture (same as recording)
- Normalizes mouse model output: `dx = int(pred[0] * 300 * 0.4)` (scale=300, sensitivity=0.4)
- DEADZONE of 4 pixels: `|raw_dx|<4` → `dx=0`
- Confidence threshold 0.5 on sigmoid keyboard outputs
- Press `ESC` for emergency stop (frees all keys/mouse buttons)
- Debug frames (every 30th frame with prediction overlay) saved to `tmpscreens/`

## 🚨 Critical gotchas

- **`play_bot.py` uses `mss`** — returns black frames on Wayland. Use `aiPvpPlay.py` instead.
- **Training normalization must match bot inference**: `MOUSE_CLIP=300` in training vs `MOUSE_SCALE=300` in bot. If you change one, change both.
- **Mouse head must use `tanh` activation**, not `linear`. Linear lets predictions blow up to ±100+, causing insane spinning.
- **BGR→RGB required after `cv2.imdecode`** — training uses `tf.image.decode_jpeg` (RGB), but `cv2.imdecode` returns BGR. Feed BGR pixel data into a model trained on RGB → tanh saturates → always predicts ±1. Add `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before model inference.
- `pvp_bot_v1.keras` is ~176MB — gitignore-worthy.
- Training data is 8720 frames (~7.5 min) of wood-chopping. `l_click=1` on 83% of frames, `w=1` on 23%, other keys are rare. Bot will heavily bias toward left-click and walk-forward.

## Files

| File | Purpose |
|------|---------|
| `screenrec.py` | Record gameplay + inputs → dataset |
| `train_bot.py` | Train TF model |
| `aiPvpPlay.py` | Run trained bot (grim, debug frames, ESC stop) |
| `play_bot.py` | OLD bot script — broken on Wayland (uses mss) |
| `pvp_dataset/labels.csv` | Frame-to-action labels |
| `pvp_dataset/frames/` | 8720 JPEG frames at 320×180 |
| `pvp_bot_v1.keras` | Saved model |
| `tmpscreens/` | Debug output from bot runs |
