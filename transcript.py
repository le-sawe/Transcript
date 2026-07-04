import sys
import re
import whisper
import time
import threading
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODEL_SIZE = "small"

_TS = re.compile(r'\[(?:(\d+):)?(\d+):(\d+\.\d+) -->')
UPDATE_EVERY = 10

_R = "\033[1;31m"   # bold red
_E = "\033[0m"      # reset


def _make_frame(star, h_bar):
    s = f"{_R}{star}{_E}"
    h = f"{_R}{h_bar * 27}{_E}"
    c = f"{_R}☭{_E}"
    return [
        f"  {s} {h} {s}",
        f"       ╔═════════════════════════╗",
        f"       ║   {c}                 {c}   ║",
        f"       ║                         ║",
        f"       ║    WORKERS  OF  THE     ║",
        f"       ║    WORLD,   UNITE!      ║",
        f"       ║                         ║",
        f"       ║   {c}                 {c}   ║",
        f"       ╚═════════════════════════╝",
        f"  {s} {h} {s}",
    ]


_FRAMES = [
    _make_frame("★", "═"),
    _make_frame("✦", "─"),
    _make_frame("☆", "═"),
    _make_frame("✧", "─"),
]
_ART_H = len(_FRAMES[0])


class _Spinner:
    def __init__(self, real_stdout):
        self._out = real_stdout
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            frame = _FRAMES[i % len(_FRAMES)]
            # save cursor → go up _ART_H lines → redraw → restore cursor
            buf = "\0337"
            buf += f"\033[{_ART_H}A"
            for line in frame:
                buf += f"\r{line}\033[K\n"
            buf += "\0338"
            self._out.write(buf)
            self._out.flush()
            i += 1
            time.sleep(0.45)


class _ProgressCapture:
    def __init__(self, pbar, real_stdout):
        self.pbar = pbar
        self._real = real_stdout
        self._last = 0

    def write(self, text):
        m = _TS.search(text)
        if m:
            h = int(m.group(1) or 0)
            pos = int(h * 3600 + int(m.group(2)) * 60 + float(m.group(3)))
            if pos - self._last >= UPDATE_EVERY:
                self.pbar.n = pos
                self.pbar.refresh()
                self._last = pos

    def flush(self):
        self._real.flush()


print(f"Loading Whisper model '{MODEL_SIZE}'...")
model = whisper.load_model(MODEL_SIZE)

videos = sorted(DATA_DIR.glob("*.mp4"))
if not videos:
    print("No .mp4 files found in data/")
    exit(1)

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Found {len(videos)} video(s) to transcribe.\n")

for i, video in enumerate(videos, 1):
    output = OUTPUT_DIR / video.with_suffix(".txt").name
    if output.exists():
        print(f"[{i}/{len(videos)}] Skipping (already done): {video.name}")
        continue

    print(f"[{i}/{len(videos)}] {video.name}")
    audio = whisper.load_audio(str(video))
    duration = int(len(audio) / whisper.audio.SAMPLE_RATE)
    print(f"    Duration: {duration // 60}m {duration % 60}s\n")

    # Print initial art frame — reserves space above the tqdm line
    for line in _FRAMES[0]:
        print(line)

    start = time.time()
    bar_fmt = "    {percentage:3.0f}%|{bar}| {n}s/{total}s  [elapsed {elapsed}, eta {remaining}]"
    real_out = sys.stdout

    with tqdm(total=duration, unit="s", ncols=72, bar_format=bar_fmt,
              file=real_out, dynamic_ncols=False) as pbar:
        spinner = _Spinner(real_out).start()
        sys.stdout = _ProgressCapture(pbar, real_out)
        result = model.transcribe(audio, verbose=True)
        sys.stdout = real_out
        spinner.stop()
        pbar.n = duration
        pbar.refresh()

    output.write_text(result["text"], encoding="utf-8")
    print(f"\n    Saved: {output.name}  ({round(time.time() - start)}s)\n")

print(f"\n{_R}☭  ALL DONE  ☭{_E}\n")
