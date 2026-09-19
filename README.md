# Auto Fishing for Play Together

**Author:** tannguyen2001 (`ntan7928@gmail.com`)  
**Social:** [Facebook](https://www.facebook.com/vantan01222/) · [LinkedIn](https://www.linkedin.com/in/tannv2k1/)

A computer-vision fishing bot for **Play Together** running on **BlueStacks**. It watches the game screen with a YOLOv8 model, reacts to bite cues in near real time, and drives the full catch–store–repair loop without manual input.

---

## Overview

Fishing in Play Together is a timing-sensitive mini-game: cast, wait for the bite indicator, reel in, then confirm **Store / Bảo quản**. Rods also wear down and must be repaired. This project automates that loop by combining:

- **YOLOv8** object detection (Roboflow dataset → local / GPU-trained weights)
- **Low-latency screen capture** from the BlueStacks window (`mss`)
- **Windows input** focused on the BlueStacks App Player / Keymap Overlay
- A **finite-state controller** that handles success, miss, and repair paths

Optional **ADB** capture remains available for calibration and debugging.

---

## Features

| Capability | Behavior |
|---|---|
| Auto cast | Clicks the cast control on a configurable schedule |
| Bite detection | Detects `exclamation-mark` inside a tight ROI for speed |
| Fast reel | Clicks the reel button immediately on bite (no per-jerk refocus) |
| Store / keep fish | Waits until `store-button` appears, then clicks it |
| Missed bite | Detects `too-late-run-away` and returns to casting |
| Rod repair | Opens the bag, navigates tool UI, confirms paid/free repair, dismisses done dialog |
| Calibration tools | `--pick`, `--preview`, `--click` for coordinate setup |
| Configurable timings | Delays, timeouts, confidence, ROI, and paid-repair policy via `config.yaml` |

---

## Detection classes

Trained on the [play-together2](https://universe.roboflow.com/rick-hsu-nv7rx/play-together2) Roboflow dataset (10 classes):

| Class | Role in the bot |
|---|---|
| `exclamation-mark` | Fish bite cue → reel |
| `store-button` | Keep / Store fish popup |
| `too-late-run-away` | Missed timing toast |
| `tool-is-broken` | Broken rod bubble → enter repair |
| `tool-tab` | Tools tab in the inventory |
| `repair-button` | Repair action on the rod slot |
| `paid-repair-button` / `paid-repair-text` | Confirm paid repair dialog |
| `repair-done-button` / `repair-done-text` | Repair finished confirmation |

---

## Control flow

```
CAST
  → WAITING_BITE  (scan ROI for !; periodic full-frame UI check)
       ├─ bite        → jerk → AFTER_CATCH
       ├─ tool broken → REPAIR
       └─ too late / leftover store → handle → COOLDOWN

AFTER_CATCH  (full-frame UI, never click Store early)
  ├─ store-button        → click → COOLDOWN
  ├─ too-late-run-away   → wait toast → COOLDOWN
  ├─ tool-is-broken      → REPAIR
  └─ timeout             → fallback Store coords → COOLDOWN

REPAIR
  open bag → tool-tab → repair-button
  → paid-repair-button (optional) → repair-done-button → Esc → COOLDOWN

COOLDOWN → CAST
```

---

## Architecture

```
main.py / train.py          Thin CLI wrappers (python -m autofishing)
config.yaml                 Runtime settings (backend, ROI, coords, timings)
autofishing/
  cli/                      Argument parsing + tool commands
  bot/                      State machine, factory, preview
  detection/                YOLOv8 wrapper
  capture/                  mss screen, BlueStacks window, ADB
  input/                    Click / key backends
  training/                 Local train entry
  config.py                 YAML → typed BotSettings
  constants.py              UI class names
models/best.pt              Inference weights
datasets/                   YOLO dataset
kaggle_train.ipynb          GPU training notebook
src/                        Deprecated import shims → autofishing.*
```

**Why window capture instead of ADB for the live loop?**  
ADB `screencap` adds hundreds of milliseconds per frame. Capturing the BlueStacks overlay with `mss` keeps grab latency in the low tens of milliseconds so reeling stays competitive.

---

## Requirements

- Windows 10/11
- Python 3.10+
- BlueStacks with Play Together running (Keymap Overlay visible for reliable clicks)
- Trained weights at `models/best.pt`
- Dependencies in `requirements.txt` (`ultralytics`, `mss`, `opencv-python`, `pyautogui`, …)

Optional:

- Android SDK `adb` for `--pick` / ADB backend
- Roboflow API key for dataset download helpers
- NVIDIA GPU / Kaggle for faster training

---

## Quick start

```powershell
cd AutoFishingPlayTogether
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Place weights at models/best.pt, calibrate coords if needed
python main.py --pick      # click UI → print coords
python main.py --preview   # live capture + latency overlay
python main.py             # run the bot
# equivalent: python -m autofishing
```

Calibrate at least:

- `cast_x` / `cast_y` — cast button  
- `reel_x` / `reel_y` — reel / jerk button  
- `store_x` / `store_y` — Store fallback  
- `bag_x` / `bag_y` — backpack icon (required for auto-repair)

Logical resolution defaults to **1600×900** (matches typical BlueStacks overlay mapping).

---

## Configuration highlights

See `config.yaml`:

- `backend: window` — recommended live path  
- `capture` ROI — crop around the character head for bite detection  
- `model.imgsz` / `bot.ui_imgsz` — small for bite speed, larger for UI popups  
- `allow_paid_repair` — set `false` to skip star-cost repairs  
- Timing knobs: `cast_delay_ms`, `after_catch_min_ms`, `after_catch_timeout_ms`, `cooldown_after_bite_ms`

---

## Training

1. Download or unzip the Roboflow YOLO export into `datasets/`.
2. Train locally with `train.py`, or use `kaggle_train.ipynb` on a GPU runtime.
3. Copy the best checkpoint to `models/best.pt`.

Validate on a real screenshot:

```powershell
python main.py --test path\to\screenshot.png
```

---

## Disclaimer

This project is for **personal research and educational use**. Automating online games may violate the publisher’s Terms of Service and can result in account penalties. Use at your own risk. The author is not affiliated with Play Together or BlueStacks.
