#!/usr/bin/env python3
"""
Beemrus - curses step sequencer with sample-accurate mixing and dynamic sample add.
Requirements (in your project venv):
    pip install sounddevice soundfile numpy
Run from your project's activated venv in a terminal (Alacritty).
"""

import subprocess
import tempfile
import os
import time
import shutil
import curses
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd

# ----------------------------
# HELPERS: Yazi folder chooser
# ----------------------------
def choose_sample_folder_cli():
    """Launch Yazi and return selected folder path (or None)."""
    print("Opening Yazi please wait...")
    time.sleep(2)

    yazi_path = shutil.which("yazi")
    if not yazi_path:
        print("Oops! I couldn't find Yazi. Do you have it installed?")
        return None

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        chooser_path = tmp.name

    try:
        subprocess.run([yazi_path, "--chooser-file", chooser_path], check=False)
        if os.path.exists(chooser_path):
            with open(chooser_path, "r") as f:
                selected = f.read().strip()
            os.unlink(chooser_path)
            if selected and os.path.isdir(selected):
                print(f"Selected sample folder: {selected}")
                return selected
            elif selected:
                print(f"Selected file: {selected}")
                return os.path.dirname(selected)
            else:
                print("No selection made.")
                return None
        else:
            print("Chooser file not created — Yazi may have been closed.")
            return None
    except Exception as e:
        print("Error launching Yazi:", e)
        return None


def list_audio_files(folder_path):
    exts = (".wav", ".flac", ".aiff", ".ogg", ".mp3")
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
    except Exception:
        files = []
    files.sort()
    return files


def filter_samples(files, keyword):
    return [f for f in files if keyword.lower() in f.lower()]


def pick_sample_from_folder_cli(folder):
    """
    Simple CLI picker: prints files and asks for a number; returns full path or None.
    Assumes terminal (not curses) active.
    """
    files = list_audio_files(folder)
    if not files:
        print("No audio files found in folder.")
        return None

    print(f"\nFound {len(files)} samples in {folder}:")
    for idx, fname in enumerate(files, start=1):
        print(f"  [{idx}] {fname}")

    while True:
        sel = input("Enter number of sample to pick (or blank to cancel): ").strip()
        if sel == "":
            return None
        if sel.isdigit() and 1 <= int(sel) <= len(files):
            return os.path.join(folder, files[int(sel) - 1])
        print("Invalid selection. Try again.")


# ----------------------------
# AUDIO: Mixer (same as before)
# ----------------------------
def mono_and_dtype(arr):
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    return arr.astype(np.float32)


def simple_resample(arr, src_sr, dst_sr):
    if src_sr == dst_sr:
        return arr
    src_len = len(arr)
    dst_len = int(round(src_len * (dst_sr / src_sr)))
    if src_len == 0 or dst_len == 0:
        return np.zeros(dst_len, dtype=np.float32)
    src_pos = np.linspace(0, src_len - 1, num=src_len)
    dst_pos = np.linspace(0, src_len - 1, num=dst_len)
    res = np.interp(dst_pos, src_pos, arr).astype(np.float32)
    return res


class Mixer:
    def __init__(self, samples_map, steps, bpm, out_sr=None, blocksize=1024):
        # determine output sample rate
        try:
            dev = sd.default.device
            if isinstance(dev, tuple):
                devinfo = sd.query_devices(dev[1])
            else:
                devinfo = sd.query_devices(dev)
            sr_default = int(devinfo.get("default_samplerate", 44100))
        except Exception:
            sr_default = 44100
        self.sr = int(out_sr) if out_sr else sr_default

        self.samples_map = samples_map.copy()
        self.steps = steps
        self.bpm = bpm
        self.blocksize = blocksize

        self.step_time = 60.0 / float(self.bpm) / 4.0
        self.samples_per_step = max(1, int(round(self.sr * self.step_time)))

        self.sample_arrays = {}
        self.max_sample_len = 0
        self._load_all_samples()

        self.pattern = {label: [0] * self.steps for label in self.sample_arrays.keys()}

        self.loop_len = self.steps * self.samples_per_step
        self.lock = threading.Lock()
        self.mixed = np.zeros(self.loop_len, dtype=np.float32)
        self._rebuild_mixed()

        self.play_pos = 0
        self.stream = None
        self.playing = False

    def _load_all_samples(self):
        for label, path in self.samples_map.items():
            try:
                arr, src_sr = sf.read(path, dtype="float32")
                arr = mono_and_dtype(arr)
                if src_sr != self.sr:
                    arr = simple_resample(arr, src_sr, self.sr)
                self.sample_arrays[label] = arr
                if len(arr) > self.max_sample_len:
                    self.max_sample_len = len(arr)
            except Exception as e:
                self.sample_arrays[label] = np.zeros(1, dtype=np.float32)
                print(f"Warning: failed to load {path}: {e}")

    def set_pattern(self, pattern):
        with self.lock:
            self.pattern = {label: pattern.get(label, [0] * self.steps) for label in self.sample_arrays.keys()}
            self._rebuild_mixed()

    def toggle_step(self, label, step_idx):
        with self.lock:
            if label in self.pattern:
                self.pattern[label][step_idx] ^= 1
                self._rebuild_mixed()

    def add_empty_track(self, label, path):
        """Add a new track (label->path). Loads sample and appends zeros to pattern for new track."""
        with self.lock:
            self.samples_map[label] = path
            try:
                arr, src_sr = sf.read(path, dtype="float32")
                arr = mono_and_dtype(arr)
                if src_sr != self.sr:
                    arr = simple_resample(arr, src_sr, self.sr)
                self.sample_arrays[label] = arr
            except Exception as e:
                self.sample_arrays[label] = np.zeros(1, dtype=np.float32)
                print(f"Warning: failed to load {path}: {e}")

            # ensure pattern entry exists
            self.pattern[label] = [0] * self.steps
            self._rebuild_mixed()

    def _rebuild_mixed(self):
        self.loop_len = self.steps * self.samples_per_step
        mixed = np.zeros(self.loop_len, dtype=np.float32)

        for label, seq in self.pattern.items():
            arr = self.sample_arrays.get(label)
            if arr is None or arr.size == 0:
                continue
            arr_len = len(arr)
            for step_idx, on in enumerate(seq):
                if not on:
                    continue
                start = step_idx * self.samples_per_step
                end = start + arr_len
                if end <= self.loop_len:
                    mixed[start:end] += arr
                else:
                    first_len = max(0, self.loop_len - start)
                    if first_len > 0:
                        mixed[start:self.loop_len] += arr[:first_len]
                    rem = arr_len - first_len
                    if rem > 0:
                        mixed[0:rem] += arr[first_len:first_len + rem]

        mx = np.max(np.abs(mixed)) if mixed.size else 0.0
        if mx > 1.0:
            mixed = mixed / mx
        self.mixed = mixed

    def start(self):
        if self.stream is not None:
            self.playing = True
            return
        self.play_pos = 0
        self.playing = True
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                blocksize=self.blocksize,
                channels=1,
                dtype="float32",
                callback=self._callback
            )
            self.stream.start()
        except Exception as e:
            print("Error opening audio stream:", e)
            self.stream = None
            self.playing = False

    def stop(self):
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def seek_start(self):
        """Move playhead to start of loop (thread-safe)."""
        with self.lock:
            self.play_pos = 0

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            buf = self.mixed
            loop_len = len(buf)
            pos = self.play_pos

            if not self.playing or loop_len == 0:
                outdata.fill(0.0)
                self.play_pos = (pos + frames) % loop_len if loop_len else 0
                return

            out = np.zeros((frames,), dtype=np.float32)

            remaining = frames
            idx = 0
            while remaining > 0 and loop_len > 0:
                take = min(remaining, loop_len - pos)
                out[idx:idx + take] = buf[pos:pos + take]
                idx += take
                remaining -= take
                pos = (pos + take) % loop_len

            self.play_pos = pos % loop_len if loop_len else 0
            outdata[:, 0] = out


# ----------------------------
# CURSES UI + integration (adds w and a)
# ----------------------------
def sequencer_curses(stdscr, mixer: Mixer, base_folder_for_add):
    steps = mixer.steps
    bpm = mixer.bpm
    track_names = list(mixer.sample_arrays.keys())
    num_tracks = len(track_names)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    cur_row = 0
    cur_col = 0

    instructions = "Arrows: move  x:add  -:remove  SPACE:start/stop  w:rewind  a:add sample  q:quit"

    def draw_ui(play_pos_samples):
        nonlocal track_names
        stdscr.erase()
        stdscr.addstr(0, 0, f"Beemrus Sequencer (4/4)  Steps: {steps}  BPM: {bpm}")
        stdscr.addstr(1, 0, instructions)
        header = "       "
        for c in range(steps):
            header += f"{(c+1):>2}" if (c % 4) == 0 else "  "
        stdscr.addstr(3, 0, header)

        start_row = 5
        play_step = int(play_pos_samples // mixer.samples_per_step) % steps if mixer.samples_per_step > 0 else 0

        track_names = list(mixer.sample_arrays.keys())
        for t_idx, name in enumerate(track_names):
            row_y = start_row + t_idx
            display_name = (name[:8]).ljust(8)
            try:
                stdscr.addstr(row_y, 0, display_name + " |")
            except curses.error:
                pass

            for c in range(steps):
                col_x = 11 + c * 2
                ch = "x" if mixer.pattern.get(name, [0]*steps)[c] else "-"
                attr = curses.A_NORMAL
                if c == play_step:
                    attr |= curses.A_REVERSE
                if t_idx == cur_row and c == cur_col:
                    attr = curses.A_STANDOUT | curses.A_BOLD | curses.A_REVERSE
                try:
                    stdscr.addstr(row_y, col_x, ch, attr)
                except curses.error:
                    pass
            try:
                stdscr.addstr(row_y, 11 + steps * 2, " |")
            except curses.error:
                pass

        status_line = start_row + len(track_names) + 1
        play_status = "PLAYING" if mixer.playing else "STOPPED"
        stdscr.addstr(status_line, 0, f"Cursor: ({cur_row+1},{cur_col+1})   {play_status}")
        stdscr.refresh()

    last_draw = 0.0
    draw_ui(mixer.play_pos)

    try:
        while True:
            key = stdscr.getch()
            if key != -1:
                if key in (curses.KEY_UP, ord('k')):
                    cur_row = max(0, cur_row - 1)
                elif key in (curses.KEY_DOWN, ord('j')):
                    cur_row = min(len(mixer.sample_arrays) - 1, cur_row + 1)
                elif key in (curses.KEY_LEFT, ord('h')):
                    cur_col = (cur_col - 1) % steps
                elif key in (curses.KEY_RIGHT, ord('l')):
                    cur_col = (cur_col + 1) % steps
                elif key == ord('x'):
                    label = list(mixer.sample_arrays.keys())[cur_row]
                    mixer.toggle_step(label, cur_col)
                elif key == ord('-'):
                    label = list(mixer.sample_arrays.keys())[cur_row]
                    mixer.toggle_step(label, cur_col)
                elif key == ord(' '):
                    if mixer.playing:
                        mixer.stop()
                    else:
                        mixer.playing = True
                        mixer.start()
                elif key == ord('w'):
                    # restart from beginning
                    mixer.seek_start()
                elif key == ord('a'):
                    # add sample flow:
                    # stop playback, restore terminal, run selection, then re-create mixer preserving pattern
                    mixer.stop()
                    curses.endwin()
                    try:
                        add_sample_flow(mixer, base_folder_for_add)
                    finally:
                        # re-enter curses wrapper by returning control to caller; caller will re-run wrapper
                        return
                elif key == ord('q'):
                    mixer.stop()
                    break
                draw_ui(mixer.play_pos)

            now = time.perf_counter()
            if now - last_draw >= 0.02:
                draw_ui(mixer.play_pos)
                last_draw = now

            time.sleep(0.005)
    finally:
        mixer.stop()


# ----------------------------
# Add-sample flow (runs outside curses)
# ----------------------------
def add_sample_flow(old_mixer: Mixer, base_folder_for_add):
    """
    Called when user pressed 'a'. This function runs in normal terminal mode (curses.endwin called).
    It asks user whether to pick from same folder or new Yazi folder, performs selection,
    then rebuilds a new Mixer that includes the new sample while preserving pattern.
    Finally it re-enters the curses UI by calling curses.wrapper again in main control flow.
    """
    print("\n--- Add Sample ---")
    same = input("Do you want to grab a sample from the same folder? (y/N): ").strip().lower() == "y"

    if same and base_folder_for_add:
        folder = base_folder_for_add
        print(f"Using same folder: {folder}")
    else:
        folder = choose_sample_folder_cli()
        if not folder:
            print("No folder selected. Returning to sequencer.")
            return

    # choose a sample from folder
    chosen_path = pick_sample_from_folder_cli(folder)
    if not chosen_path:
        print("No sample selected. Returning to sequencer.")
        return

    # ask for a label for the new sample, ensure unique key
    while True:
        label = input("What should we call this sample (e.g., kick, snare, hat)? ").strip().lower()
        if not label:
            print("Please provide a non-empty label.")
            continue
        if label in old_mixer.sample_arrays:
            label = label + "_1"
            print(f"Label exists; using {label}")
        break

    # Build new samples_map merging old entries plus new one
    new_samples_map = old_mixer.samples_map.copy()
    new_samples_map[label] = chosen_path

    # Preserve old pattern mapping (old_mixer.pattern)
    old_pattern = old_mixer.pattern.copy()
    steps = old_mixer.steps
    bpm = old_mixer.bpm

    # Instantiate new mixer with updated samples_map
    new_mixer = Mixer(new_samples_map, steps, bpm)
    # Build a new pattern dict to pass to new_mixer — ensure every label exists
    new_pattern = {}
    for lab in new_mixer.sample_arrays.keys():
        if lab in old_pattern:
            new_pattern[lab] = old_pattern[lab]
        else:
            new_pattern[lab] = [0] * steps

    new_mixer.set_pattern(new_pattern)

    print(f"Added sample '{label}' from {chosen_path}. Returning to sequencer...")
    time.sleep(0.4)

    # Re-enter curses UI with new mixer — caller (main) uses wrapper to re-run UI
    global REENTER_MIXER
    REENTER_MIXER = (new_mixer, base_folder_for_add if same else folder)


# ----------------------------
# Glue: selection flow + start
# ----------------------------
def open_ascii_sequencer(samples_map):
    print("Let's get grooving.")
    print("PLEASE NOTE: Sequences can only be in 4/4 time right now")

    while True:
        try:
            steps = int(input("Enter sequence length (8–32): ").strip())
            if 8 <= steps <= 32 and steps % 4 == 0:
                break
            print("Must be between 8 and 32 and divisible by 4 (to stay in 4/4).")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        bpm_input = input("Enter tempo in BPM (default 120): ").strip()
        if bpm_input == "":
            bpm = 120
            break
        try:
            bpm = int(bpm_input)
            if 40 <= bpm <= 300:
                break
            print("Tempo must be between 40 and 300 BPM.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nSequence length: {steps} steps — Tempo: {bpm} BPM — Time signature: 4/4")
    print("Starting sequencer UI. Press 'q' inside the UI to quit back to CLI.")
    time.sleep(0.5)

    # initial mixer
    mixer = Mixer(samples_map, steps, bpm)
    mixer.set_pattern({label: [0]*steps for label in mixer.sample_arrays.keys()})

    # we keep base_folder_for_add to know same-folder option - infer from samples_map first item
    base_folder_for_add = None
    if samples_map:
        # take folder of first sample as "same folder"
        first_path = next(iter(samples_map.values()))
        base_folder_for_add = os.path.dirname(first_path)

    global REENTER_MIXER
    REENTER_MIXER = None

    # loop that allows re-entering curses when add_sample_flow requests it
    while True:
        try:
            curses.wrapper(lambda stdscr: sequencer_curses(stdscr, mixer, base_folder_for_add))
        except Exception as e:
            print("Sequencer UI exited with error:", e)
            mixer.stop()
            return

        # if add_sample_flow set REENTER_MIXER, grab it and continue; else break (user quit)
        if REENTER_MIXER:
            mixer, base_folder_for_add = REENTER_MIXER
            REENTER_MIXER = None
            # continue loop -> re-enter curses with new mixer and preserved pattern
            continue
        else:
            # normal exit
            break

    # ensure mixer stopped
    mixer.stop()


def choose_samples():
    folder = choose_sample_folder_cli()
    if not folder:
        return {}

    files = list_audio_files(folder)
    if not files:
        print("No audio files found in that folder.")
        return {}

    print(f"\nFound {len(files)} samples in {folder}:")
    for f in files:
        print("    " + f)

    selected_samples = {}
    first_time = True

    while True:
        prompt_word = "first" if first_time else "next"
        print(f"\nOkay! What sample do you want {prompt_word}?")
        keyword = input("Type a keyword to filter filenames (e.g., kick, snare, hat). Leave blank to list all: ").strip()

        matches = files[:] if keyword == "" else filter_samples(files, keyword)
        if not matches:
            print(f"No files found containing '{keyword}'.")
        else:
            print(f"\nFound {len(matches)} match(es):")
            for i, fname in enumerate(matches, start=1):
                print(f"  [{i}] {fname}")

            choice = input("Enter the number of the sample to select (or press Enter to cancel): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                chosen = matches[int(choice) - 1]
                kind = input("What should we call this sample (e.g., kick, snare, hat)? ").strip().lower()
                if not kind:
                    print("No label given — sample not saved.")
                else:
                    full_path = os.path.join(folder, chosen)
                    # ensure unique label
                    lab = kind
                    i = 1
                    while lab in selected_samples:
                        lab = f"{kind}_{i}"
                        i += 1
                    selected_samples[lab] = full_path
                    print(f"Added '{lab}': {chosen}")
                    first_time = False
            else:
                print("Selection cancelled or invalid number.")

        again = input("\nDo you want to select any other samples? (y/N): ").strip().lower()
        if again != "y":
            break

    return selected_samples


def main():
    print("Welcome to Beemrus! :3")
    print("     We'll need to start with picking your samples.")

    samples = choose_samples()
    if not samples:
        print("No samples selected. Exiting Beemrus.")
        return

    print("\nSamples selected:")
    for k, v in samples.items():
        print(f"   {k}: {v}")

    open_ascii_sequencer(samples)


if __name__ == "__main__":
    main()
