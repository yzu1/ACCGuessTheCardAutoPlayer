# ACCGuessTheCardAutoPlayer
Self-learning bot for roblox blurred-card guessing game on Anime Card Clash. 
It reads the four answer buttons, compares the blurred card with every card it has seen before, and clicks the best match. When the game reveals the answer, the bot saves that card's pictures, so it gets more accurate the more it plays.


## How it works
1. Finds the game on screen by its orange timer bar, so there's nothing to calibrate.
2. Reads the four names with OCR (Tesseract).
3. Compares the blurred card with its saved pictures. If none match, it rules out the cards it knows and guesses among the names it hasn't seen yet.
4. After the reveal, saves the sharp picture to `cards/<Name>.png` and the blurred one to `cards/blurred/<Name>__N.png`.
5. When a game ends, clicks **Play again** (never "Continue for R$"). Stops after round 215, round cap being subjected to user change.


## Setup

pip install mss pillow numpy pyautogui pytesseract

Python 3.xx
Tesseract OCR (https://github.com/UB-Mannheim/tesseract/wiki) If Python can't find it, set `TESSERACT_PATH` at the top of `autoplayer.py`, for example `r"C:\Program Files\Tesseract-OCR\tesseract.exe"`.

## Card Database 

The bot can start from nothing and learn from played iterations overtime, or you can skip the learning by downloading the database on release of about 600 cards:

1. Download "cards.zip" from this repo's release page.
2. Unzip it into the folder that holds autoplayer.py, so you end up with autoplayer.py and cards in the same folder.
3. The first start takes the longest while it prepares the pictures, subsequent stats have a cache to it takes a few seconds.


##Usage

Open the game, then run these from a terminal in the project folder:

| Command | What it does |
|---|---|
| `python autoplayer.py snap` | Checks it can find the game (saves crops to `debug/`) |
| `python autoplayer.py run` | Plays and learns |
| `python autoplayer.py run --watch` | You play; the bot learns and shows what it would pick |
| `python autoplayer.py --monitor 2 run` | Use this if the game is on your second screen |
| `python autoplayer.py rename "Old Name" "New Name"` | Fixes a misread card name (stop the bot first) |
| `python autoplayer.py test shot.png` | Runs the `snap` check on a saved screenshot |

To stop it, press **Ctrl+C** or move the mouse into a corner of the screen.

**Settings** at the top of `autoplayer.py`: `MAX_ROUND` (the last round it plays, 215) and `ANSWER_WAIT` (how many seconds it waits before clicking).

**Fixing names:** the OCR sometimes misreads letters, such as B as G. `rename` moves all the pictures to the right name and remembers the misreading, so next time it maps straight to that name. To move a single wrongly saved picture instead, use its file name: `rename "Shadow Knight__2" "Shadow Ant"`.
