"""
Self-learning auto-player for the hackathon's blurred-card guessing game.

No card images needed up front: the bot learns from the game's own reveals.
Each round:
  1. Screenshot the game, read the four option names with OCR.
  2. Compare the blurred card against everything it has learned so far:
       - sharp reveal images of known cards (blurred at several strengths)
       - blurred versions it has seen before
     If a known card clearly matches, click it. If not, rule out the known cards and
     guess among the names it hasn't seen yet (process of elimination).
  3. When the game turns the correct button green and unblurs the card, save:
       cards/<Name>.png               the sharp card (<Name>__2.png, ... if the name has other pictures)
       cards/blurred/<Name>__N.png    the blurred version from this round
  4. Wait for the next round.
It starts by guessing and gets close to perfect once it has seen each card.

Setup (once):
  pip install mss pillow numpy pyautogui pytesseract
  Install Tesseract OCR:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki  (then set TESSERACT_PATH below if needed)
    macOS:   brew install tesseract

Commands:
  python autoplayer.py snap               # check it can find the game; saves crops to debug/
  python autoplayer.py run                # play and learn; skips the continue offer and clicks "Play again" after each game;
                                          # stops for good after game round 215 (MAX_ROUND)
  python autoplayer.py run --watch        # don't click: YOU play, the bot just learns and shows its guesses
  python autoplayer.py test shot.png      # same check on a saved screenshot
  python autoplayer.py --monitor 2 run    # if the game is on your second screen
  python autoplayer.py rename "Gerserk Mode" "Berserk Mode"   # fix a misread card name (stop the bot first; the misreading is remembered)
  python autoplayer.py rename "Shadow Knight__2" "Shadow Ant" # move one wrongly named picture (+ its blurred ones)
The game is found automatically each frame by locating the timer bar - no calibration needed.
Stop with Ctrl+C, or slam the mouse into a screen corner (pyautogui failsafe).
"""
import argparse
import difflib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ---------------------------------------------------------------- settings
TESSERACT_PATH = None  # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe" if not on PATH

CARDS_DIR = Path("cards")
BLURRED_DIR = CARDS_DIR / "blurred"
ALIASES_FILE = CARDS_DIR / "aliases.json"  # misread spelling -> the right name, written by the rename command
FEATURE_CACHE = CARDS_DIR / "features.npz"  # blur grids of every saved picture, so startup doesn't redo them
DEBUG_DIR = Path("debug")

# Layout in pixels relative to the timer bar's top-left corner, at a bar length of 1190 px.
# (Measured from your screenshots; scaled automatically if the bar is a different size.)
BAR_LENGTH = 1190
CARD_OFFSET = (56, 101, 463, 650)            # inside the card's coloured border
ROUND_OFFSET = (-10, -50, 260, -4)           # the orange "Round N" label above the bar
BUTTON_OFFSETS = [
    (552, 312, 841, 383),  # top-left
    (888, 312, 1178, 383),  # top-right
    (552, 408, 841, 479),  # bottom-left
    (888, 408, 1178, 479),  # bottom-right
]

WORK_SIZE = (300, 400)                   # card is normalised to this size before blurring
COMPARE_SIZE = (24, 32)                  # tiny colour grid used for comparisons
BLUR_RADII = [0, 4, 8, 12, 16, 22, 30]   # blur strengths tried on learned images
KNOWN_MATCH_MAX = 20.0                   # distance below which a learned card counts as a confident match
SAME_IMAGE_MAX = 15.0                    # sharp pictures closer than this are the same image (different cards measure 28+)
NAME_LETTERS_PER_TYPO = 6                # OCR text counts as a known name if it's off by 1 letter per 6 (at least 1)
ANSWER_DELAY = 0.5                       # wait after a new round shows up before reading it for real
MAX_ROUND = 215                          # hard cap: answer up to this game round, then stop the bot
ANSWER_WAIT = (0.75,1.5)                # click a random time in this range (seconds) after a round appears
REVEAL_TIMEOUT = 20.0                    # max seconds to wait for the green reveal
REVEAL_SETTLE = 0.3                      # wait after green appears so the unblur has finished
POLL_INTERVAL = 0.1
CLICK_HOLD = 0.05                        # pause between nudging the mouse, pressing and releasing
REPLAY_CHECK_INTERVAL = 0.25             # how often to look for "Give up" / "Play again" when no round is showing
REPLAY_COOLDOWN = 1.0                   # wait after clicking one before looking again


# ---------------------------------------------------------------- image helpers
_last_bar = None  # (x, y, length) of the timer bar the last time it had time left on it


def find_timer_bar(shot):
    """Find the timer bar (orange fill + black track) near the top. Returns (x, y, length) or None.
    The bar has to start with orange fill, so black strips can't pass for it (Roblox's title bar turns pure
    black depending on window focus). Once time runs out the bar is all black, so then it's checked at the
    spot it was last seen."""
    global _last_bar
    a = np.asarray(shot.convert("RGB"))[: int(shot.height * 0.3)].astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    orange = (r > 180) & (b < 120) & (r - g > 40)
    # The track is pure black; the dark band across the top of the game window is only nearly black.
    bar = orange | (a.max(axis=2) < 3)
    left, right = np.zeros_like(bar), np.zeros_like(bar)
    for s in (1, 2, 3):  # close the 1-2 px seam where the orange fill meets the black track
        left[:, s:] |= bar[:, :-s]
        right[:, :-s] |= bar[:, s:]
    bar |= left & right
    best = (0, 0, 0)  # length, x, y
    for y in range(0, bar.shape[0], 2):
        row = np.concatenate(([0], bar[y].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(row))
        starts, ends = edges[::2], edges[1::2]
        last = bar.shape[1] - 1
        starts_orange = orange[y, np.minimum(starts + 3, last)] & orange[y, np.minimum(starts + 8, last)]
        lengths = (ends - starts) * starts_orange
        if len(lengths) and lengths.max() > best[0]:
            i = int(np.argmax(lengths))
            best = (int(lengths[i]), int(starts[i]), y)
    length, x, y = best
    if length >= shot.width * 0.35:
        while y > 0 and bar[y - 1, x + 10]:  # walk up to the bar's top edge (left end avoids the timer text)
            y -= 1
        _last_bar = (x, y, length)
        return _last_bar
    if _last_bar:  # time ran out: is the all-black bar still where it was?
        x, y, length = _last_bar
        if y + 5 < bar.shape[0] and bar[y + 5, x + 10 : x + length - 10].mean() > 0.9:
            return _last_bar
    return None


def locate(shot):
    """Pixel boxes for the card and the four buttons in this screenshot, or None if no game is visible."""
    bar = find_timer_bar(shot)
    if bar is None:
        return None
    x, y, length = bar
    k = length / BAR_LENGTH
    place = lambda o: (int(x + o[0] * k), int(y + o[1] * k), int(x + o[2] * k), int(y + o[3] * k))
    return {"card": place(CARD_OFFSET), "buttons": [place(o) for o in BUTTON_OFFSETS], "round": place(ROUND_OFFSET)}


def card_image(shot, layout):
    return shot.crop(layout["card"]).resize(WORK_SIZE, Image.LANCZOS)


def to_features(img):
    """Shrink to a tiny Lab colour grid. The shrink is itself a strong blur, so this is blur-tolerant."""
    small = img.convert("RGB").resize(COMPARE_SIZE, Image.BILINEAR).convert("LAB")
    return np.asarray(small, dtype=np.float32)


def green_button(shot, layout):
    """Index of the button the game has highlighted green, or None."""
    for i, box in enumerate(layout["buttons"]):
        a = np.asarray(shot.crop(box).convert("RGB")).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        green = (g > 120) & (g - r > 35) & (g - b > 35)
        if green.mean() > 0.25:
            return i
    return None


def tesseract():
    """pytesseract, pointed at the Tesseract program."""
    import pytesseract
    default_win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    elif default_win.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_win)
    return pytesseract


def read_round(shot, layout):
    """The game's round number from the orange "Round N" label above the timer bar, or None if unreadable."""
    a = np.asarray(shot.crop(layout["round"]).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    bw = Image.fromarray(np.where((r > 180) & (b < 120) & (r - g > 40), 0, 255).astype("uint8"))
    bw = ImageOps.expand(bw.resize((bw.width * 2, bw.height * 2), Image.LANCZOS), 20, 255)
    text = tesseract().image_to_string(bw, config="--psm 7 -c tessedit_char_whitelist=Round0123456789")
    number = re.search(r"\d+", text)
    return int(number.group()) if number else None


def read_button(img):
    """OCR one answer button. Labels are white with a dark outline, so keep only the white pixels."""
    gray = np.asarray(img.convert("L")).astype(int)
    bw = Image.fromarray(np.where(gray > 240, 0, 255).astype("uint8"))
    bw = bw.resize((bw.width * 3, bw.height * 3), Image.LANCZOS)
    bw = ImageOps.expand(bw, 20, 255)
    # Read it as a block, not a single line: long names wrap onto two lines ("Awakened Sun Blaze / Shinigami").
    text = tesseract().image_to_string(bw, config="--psm 6")
    return re.sub(r"[^A-Za-z0-9 '\-]", "", " ".join(text.split())).strip()


def read_options(shot, layout):
    with ThreadPoolExecutor(4) as pool:  # each read runs Tesseract as its own process, so do all four at once
        return list(pool.map(read_button, [shot.crop(b) for b in layout["buttons"]]))


def green_areas(shot):
    """Bounding boxes of wide green blobs (button-shaped) anywhere on the screen."""
    a = np.asarray(shot.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > 120) & (g - r > 35) & (g - b > 35)
    boxes, active = [], []  # boxes are [x0, y0, x1, y1], grown row by row
    for y in range(0, green.shape[0], 2):
        active = [box for box in active if box[3] >= y - 4]
        row = np.concatenate(([0], green[y].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(row))
        for x0, x1 in zip(edges[::2], edges[1::2]):
            if x1 - x0 < 4:
                continue
            for box in active:
                if x0 < box[2] and x1 > box[0]:  # touches a blob from the rows above
                    box[0], box[2], box[3] = min(box[0], x0), max(box[2], x1), y
                    break
            else:
                box = [x0, y, x1, y]
                boxes.append(box)
                active.append(box)
    return [tuple(int(v) for v in box) for box in boxes
            if box[2] - box[0] > shot.width * 0.05 and box[2] - box[0] > 2 * (box[3] - box[1]) > 20]


def find_end_button(shot):
    """The end-screen button to click next: ("Give up", box), ("Play again", box) or None.
    "Give up" sits just under the green "Continue for R$" button and skips its countdown.
    A button is only returned when its text reads right, so the bot can never click "Continue for R$"."""
    for box in green_areas(shot):
        text = read_button(shot.crop(box)).lower()
        if difflib.get_close_matches(text, ["play again"], cutoff=0.7):
            return "Play again", box
        if "continue" in text:
            x0, y0, x1, y1 = box
            h = y1 - y0
            below = (x0, y1 + h // 8, x1, y1 + h + h // 4)
            if difflib.get_close_matches(read_button(shot.crop(below)).lower(), ["give up"], cutoff=0.7):
                return "Give up", below
    return None


# ---------------------------------------------------------------- what the bot has learned
def blur_grids(img):
    """Feature grids of a picture at each blur strength in BLUR_RADII (the first one unblurred)."""
    base = img.convert("RGB").resize(WORK_SIZE, Image.LANCZOS)
    return [to_features(base.filter(ImageFilter.GaussianBlur(r)) if r else base) for r in BLUR_RADII]


def edit_distance(a, b):
    """Fewest single-letter changes (swap, add or remove a letter) that turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cached_blur_grids(paths):
    """blur_grids() of each picture, as {path: list of grids}. Blurring every picture 7 ways is most of the
    startup time, so the results are kept in FEATURE_CACHE and only pictures added or renamed since the last
    run get blurred again. The grids are whole numbers 0-255, so they're stored as uint8 without losing anything."""
    def key(path):
        st = path.stat()
        return f"{path.parent.name}/{path.name}|{st.st_size}|{st.st_mtime_ns}"

    keys = [key(p) for p in paths]
    cache = {}
    if FEATURE_CACHE.exists():
        try:
            with np.load(FEATURE_CACHE) as data:
                cache = dict(zip(data["keys"].tolist(), data["grids"]))
        except Exception:  # damaged file - just rebuild it
            print("Picture cache unreadable, rebuilding it.")
    todo = [(k, p) for k, p in zip(keys, paths) if k not in cache]
    if todo:
        print(f"Preparing {len(todo)} new pictures (saved for next time)...")
        with ThreadPoolExecutor() as pool:  # Pillow releases the GIL while blurring, so threads run in parallel
            done = pool.map(lambda p: np.stack(blur_grids(Image.open(p))).astype(np.uint8), [p for _, p in todo])
            cache.update(zip([k for k, _ in todo], done))
    if keys and (todo or len(cache) != len(keys)):  # something new or something gone: rewrite the cache
        tmp = FEATURE_CACHE.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), grids=np.stack([cache[k] for k in keys]))
        tmp.replace(FEATURE_CACHE)  # swap in whole, so a crash mid-write can't leave a broken cache
    return {p: list(cache[k].astype(np.float32)) for k, p in zip(keys, paths)}


def load_aliases():
    """Misread spellings fixed with the rename command, lower-cased, mapped to the name they were renamed to."""
    return json.loads(ALIASES_FILE.read_text()) if ALIASES_FILE.exists() else {}


def next_free(folder, name, plain_first=False):
    """Unused path for another picture of name: <name>.png first if plain_first, then <name>__2.png, ...
    (blurred pictures start at <name>__1.png)."""
    if plain_first and not (folder / f"{name}.png").exists():
        return folder / f"{name}.png"
    k = 2 if plain_first else 1
    while (folder / f"{name}__{k}.png").exists():
        k += 1
    return folder / f"{name}__{k}.png"


class CardMemory:
    def __init__(self):
        CARDS_DIR.mkdir(exist_ok=True)
        BLURRED_DIR.mkdir(exist_ok=True)
        self.feats = {}  # name -> list of feature grids
        self.images = {}  # name -> unblurred grid of each different sharp picture saved for it
        self.aliases = load_aliases()
        sharp, blurred = sorted(CARDS_DIR.glob("*.png")), sorted(BLURRED_DIR.glob("*.png"))
        grids = cached_blur_grids(sharp + blurred)
        for path in sharp:
            name = path.stem.split("__")[0]  # extra pictures of a name are saved as <Name>__2.png, <Name>__3.png, ...
            self.feats.setdefault(name, []).extend(grids[path])
            self.images.setdefault(name, []).append(grids[path][0])
        for path in blurred:
            self.feats.setdefault(path.stem.split("__")[0], []).extend(grids[path])

    def _add_feats(self, name, img):
        grids = blur_grids(img)
        self.feats.setdefault(name, []).extend(grids)
        return grids[0]  # the unblurred grid (BLUR_RADII starts at 0)

    @property
    def names(self):
        return list(self.feats)

    def resolve(self, text):
        """Map OCR text to a known card name, or return the cleaned text as a new name.
        Only a near-identical spelling counts as the same card (a misread letter or two, same numbers): real
        names can differ by a word or a digit, like "Galactic Tyrant" / "Awakened Galactic Tyrant" or
        "Shadow Ant" / "Shadow Knight". Misreadings fixed with the rename command count as spellings of the name
        they were renamed to, so one the OCR keeps making (like "Gig Grain" for "Big Brain") maps back."""
        if len(text) < 3:
            return None
        t = text.lower()
        allowed = max(1, len(t) // NAME_LETTERS_PER_TYPO)
        best = None  # (edits, name)
        spellings = [(name.lower(), name) for name in self.feats]
        spellings += [(old, name) for old, name in self.aliases.items() if name in self.feats]
        for n, name in spellings:
            if abs(len(n) - len(t)) > allowed or re.sub(r"\D", "", n) != re.sub(r"\D", "", t):
                continue
            edits = edit_distance(t, n)
            if edits <= allowed and (best is None or edits < best[0]):
                best = (edits, name)
        return best[1] if best else text

    def distance(self, name, card_feat):
        return min(float(np.abs(f - card_feat).mean()) for f in self.feats[name])

    def learn(self, name, blurred_img, sharp_img):
        """Save this round's pictures. One name can have several different card pictures, so the sharp one is
        kept whenever it differs from every picture already saved for that name. Returns how many different
        pictures the name has now if this one was new, else None."""
        sharp = to_features(sharp_img.convert("RGB").resize(WORK_SIZE, Image.LANCZOS))
        saved = self.images.setdefault(name, [])
        new_image = all(float(np.abs(f - sharp).mean()) > SAME_IMAGE_MAX for f in saved)
        if new_image:
            sharp_img.save(next_free(CARDS_DIR, name, plain_first=True))
            saved.append(self._add_feats(name, sharp_img))
        blurred_img.save(next_free(BLURRED_DIR, name))
        self._add_feats(name, blurred_img)
        return len(saved) if new_image else None


# ---------------------------------------------------------------- the decision
def decide(shot, layout, mem):
    """Pick a button. Returns (index, names, report_lines) or None if no round is showing."""
    texts = read_options(shot, layout)
    names = [mem.resolve(t) for t in texts]
    if sum(n is not None for n in names) < 2:
        return None
    card_feat = to_features(card_image(shot, layout))
    known = {i: mem.distance(n, card_feat) for i, n in enumerate(names) if n in mem.feats}
    unknown = [i for i, n in enumerate(names) if n and n not in mem.feats]

    if known and (min(known.values()) <= KNOWN_MATCH_MAX or not unknown):
        pick, why = min(known, key=known.get), "closest learned card"
    elif unknown:
        pick, why = random.choice(unknown), "no learned card matches -> guessing an unseen one"
    else:
        return None

    lines = []
    for i, (t, n) in enumerate(zip(texts, names)):
        score = f"{known[i]:6.1f}" if i in known else (" new  " if i in unknown else "  ??  ")
        mark = "  <== pick" if i == pick else ""
        lines.append(f"   [{i}] {score}  {n or repr(t)}{mark}")
    lines.append(f"   ({why})")
    return pick, names, lines


# ---------------------------------------------------------------- screen access
def grab(sct, monitor):
    raw = sct.grab(monitor)
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def click_box(monitor, shot, box):
    """Click the middle of a box. Roblox misses a cursor that teleports and clicks in the same instant
    (what pyautogui.click does), so wiggle the mouse with real input first and hold the button briefly."""
    import pyautogui
    ratio = shot.width / monitor["width"]  # handles Retina / high-DPI screens
    l, t, r, b = box
    pyautogui.moveTo(int(monitor["left"] + (l + r) / 2 / ratio), int(monitor["top"] + (t + b) / 2 / ratio))
    if sys.platform == "win32":
        import ctypes
        for dx in (1, -1):  # relative MOUSEEVENTF_MOVE: a movement the game sees, unlike pyautogui's SetCursorPos
            ctypes.windll.user32.mouse_event(0x0001, dx, 0, 0, 0)
            time.sleep(CLICK_HOLD)
    pyautogui.mouseDown()
    time.sleep(CLICK_HOLD)
    pyautogui.mouseUp()


def save_debug(shot, layout):
    DEBUG_DIR.mkdir(exist_ok=True)
    shot.save(DEBUG_DIR / "full.png")
    if layout:
        shot.crop(layout["card"]).save(DEBUG_DIR / "card.png")
        for i, b in enumerate(layout["buttons"]):
            shot.crop(b).save(DEBUG_DIR / f"button{i}.png")


def describe(shot, mem):
    layout = locate(shot)
    if layout is None:
        found = find_end_button(shot)
        if found:
            print(f'No round showing, but found the "{found[0]}" button at {found[1]}.')
        else:
            print("Couldn't find the timer bar - is the game visible on this screen?")
        return
    save_debug(shot, layout)
    print(f"Game found. Crops saved to {DEBUG_DIR}/.  Green button: {green_button(shot, layout)}")
    result = decide(shot, layout, mem)
    print("\n".join(result[2]) if result else "No readable options (normal during a reveal).")


def cmd_snap(args):
    import mss
    with mss.mss() as sct:
        shot = grab(sct, sct.monitors[args.monitor])
    describe(shot, CardMemory())


def cmd_test(args):
    mem = CardMemory()
    print(f"{len(mem.names)} cards learned so far.")
    describe(Image.open(args.image).convert("RGB"), mem)


def move_pictures(paths, folder, name, plain_first=False):
    """Move pictures into folder as pictures of name, numbered after any it already has."""
    temps = [p.rename(p.with_name(f"~moving{i}.png")) for i, p in enumerate(paths)]  # frees the old names first
    for t in temps:
        t.rename(next_free(folder, name, plain_first))


def taken_from(sharp_paths, blurred_paths):
    """For each blurred capture, the sharp picture it was taken of (the one it looks most like)."""
    grids = [blur_grids(Image.open(p)) for p in sharp_paths]
    owners = []
    for b in blurred_paths:
        f = to_features(Image.open(b).convert("RGB").resize(WORK_SIZE, Image.LANCZOS))
        dists = [min(float(np.abs(g - f).mean()) for g in gs) for gs in grids]
        owners.append(sharp_paths[int(np.argmin(dists))])
    return owners


def cmd_rename(args):
    """Fix a misread name. OLD is a card name (all its pictures move, merged into NEW if that exists) or one
    picture of it like "Shadow Knight__2" (that picture and the blurred captures taken of it move)."""
    name, _, num = args.old.partition("__")
    of_name = lambda folder: [p for p in sorted(folder.glob("*.png")) if p.stem.split("__")[0].lower() == name.lower()]
    sharp, blurred = of_name(CARDS_DIR), of_name(BLURRED_DIR)
    if num:
        stem = name if num == "1" else args.old  # picture 1 is the plain <Name>.png
        one = [p for p in sharp if p.stem.lower() == stem.lower()]
        if not one:
            print(f'No saved picture "{args.old}".')
            return
        blurred = [b for b, owner in zip(blurred, taken_from(sharp, blurred)) if owner == one[0]]
        sharp = one
    if not sharp and not blurred:
        print(f'No saved pictures named "{args.old}".')
        return
    move_pictures(sharp, CARDS_DIR, args.new, plain_first=True)
    move_pictures(blurred, BLURRED_DIR, args.new)
    print(f'Moved {len(sharp)} sharp and {len(blurred)} blurred pictures from "{args.old}" to "{args.new}".')
    if not num and name.lower() != args.new.lower():  # remember the misreading so the bot maps it to the new name
        aliases = {old: (args.new if new.lower() == name.lower() else new) for old, new in load_aliases().items()}
        aliases[name.lower()] = args.new
        aliases.pop(args.new.lower(), None)  # renaming back undoes the old alias
        ALIASES_FILE.write_text(json.dumps(aliases, indent=1, sort_keys=True))


def wait_for(sct, monitor, condition, timeout):
    """Poll the screen until condition(shot, layout) is true. Returns (shot, layout) or (None, None)."""
    end = time.time() + timeout
    while time.time() < end:
        shot = grab(sct, monitor)
        layout = locate(shot)
        if layout and condition(shot, layout):
            return shot, layout
        time.sleep(POLL_INTERVAL)
    return None, None


def cmd_run(args):
    import mss
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0  # no built-in 0.1 s pause after every mouse call; click_box has its own timing
    mem = CardMemory()
    print(f"{len(mem.names)} cards learned so far. {'WATCH MODE (not clicking). ' if args.watch else ''}Ctrl+C to stop.")
    rounds = right = 0
    last_seen, announced, next_replay = time.time(), False, 0.0
    last_names = None

    with mss.mss() as sct:
        monitor = sct.monitors[args.monitor]
        while True:
            # 1. wait for a round: game visible, no green highlight, options readable
            shot = grab(sct, monitor)
            layout = locate(shot)
            if layout is None:
                # End screen / menus. Only buttons whose text reads "Give up" or "Play again" are clicked here,
                # so it can't hit "Continue for R$".
                if not announced and time.time() - last_seen > 3:
                    then = "Start a new game and I'll carry on" if args.watch else 'Looking for "Play again"'
                    print(f"\nNo game on screen (game over?). Score this session: {right}/{rounds}, "
                          f"{len(mem.names)} cards learned. {then}, or Ctrl+C to quit.")
                    announced = True
                if not args.watch and time.time() >= next_replay:
                    found = find_end_button(shot)
                    next_replay = time.time() + (REPLAY_COOLDOWN if found else REPLAY_CHECK_INTERVAL)
                    if found:
                        print(f'Clicking "{found[0]}".')
                        click_box(monitor, shot, found[1])
                time.sleep(POLL_INTERVAL)
                continue
            last_seen, announced = time.time(), False
            if green_button(shot, layout) is not None:
                time.sleep(POLL_INTERVAL)
                continue
            round_seen = time.time()
            texts = read_options(shot, layout)
            if sum(len(t) >= 3 for t in texts) < 2 or [mem.resolve(t) for t in texts] == last_names:
                time.sleep(POLL_INTERVAL)  # no round, or still the one just answered
                continue
            # The first read can catch the buttons while they're still appearing, so let the round settle and
            # read the names and the card again from a fresh frame.
            time.sleep(ANSWER_DELAY)
            shot = grab(sct, monitor)
            layout = locate(shot)
            result = layout and decide(shot, layout, mem)
            if not result or result[1] == last_names:
                continue
            pick, names, lines = result
            last_names = names
            game_round = read_round(shot, layout)
            if game_round and game_round > MAX_ROUND and not args.watch:
                print(f"\nRound {game_round} is past the cap of {MAX_ROUND} - stopping without answering.")
                return
            blurred = card_image(shot, layout)
            rounds += 1
            wait = random.uniform(*ANSWER_WAIT)
            print(f"\nRound {game_round or '?'}: picking {names[pick]}" + ("" if args.watch else f", clicking {wait:.1f}s in"))
            print("\n".join(lines))
            if not args.watch:
                time.sleep(max(0.0, round_seen + wait - time.time()))  # reading the options counts toward the wait
                click_box(monitor, shot, layout["buttons"][pick])

            # 2. wait for the reveal (correct button turns green), then learn from it
            revealed, rlayout = wait_for(sct, monitor, lambda s, l: green_button(s, l) is not None, REVEAL_TIMEOUT)
            if revealed is None:
                print("   no reveal seen (game over or paused?)")
            else:
                correct_idx = green_button(revealed, rlayout)
                time.sleep(REVEAL_SETTLE)
                sharp_shot = grab(sct, monitor)
                sharp = card_image(sharp_shot, locate(sharp_shot) or rlayout)
                correct = names[correct_idx]
                pictures = mem.learn(correct, blurred, sharp) if correct else None
                right += correct_idx == pick
                print(f"   answer: {correct}  -> {'RIGHT' if correct_idx == pick else 'wrong'}"
                      f"   [{right}/{rounds} correct this session, {len(mem.names)} cards learned]")
                if pictures and pictures > 1:
                    print(f"   new picture for {correct}: {pictures} different ones saved")
            if game_round and game_round >= MAX_ROUND and not args.watch:
                print(f"\nRound {game_round} done - that's the cap of {MAX_ROUND}, stopping.")
                return
            if revealed is None:
                continue

            # 3. wait for the highlight to clear before the next round
            end = time.time() + REVEAL_TIMEOUT
            while time.time() < end:
                s = grab(sct, monitor)
                l = locate(s)
                if l is None or green_button(s, l) is None:
                    break
                time.sleep(POLL_INTERVAL)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    p.add_argument("--monitor", type=int, default=1, help="which screen the game is on (1 = main)")
    sub.add_parser("snap").set_defaults(fn=cmd_snap)
    t = sub.add_parser("test")
    t.add_argument("image")
    t.set_defaults(fn=cmd_test)
    r = sub.add_parser("run")
    r.add_argument("--watch", action="store_true", help="don't click; learn while you play")
    r.set_defaults(fn=cmd_run)
    rn = sub.add_parser("rename")
    rn.add_argument("old")
    rn.add_argument("new")
    rn.set_defaults(fn=cmd_rename)
    args = p.parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
