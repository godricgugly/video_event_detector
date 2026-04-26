import tkinter as tk
from tkinter import filedialog, ttk
import threading
import math

from src.app.pipeline import run_detection
from src.video.roi import select_roi


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Event Detector")

        self.ref_path = tk.StringVar()
        self.main_path = tk.StringVar()

        # --- Similarity Threshold Slider ---
        tk.Label(root, text="Similarity Threshold", font=("Arial", 12, "bold")).pack()
        self.threshold = tk.DoubleVar(value=0.25)
        threshold_slider = tk.Scale(
            root,
            from_=0.1,
            to=0.7,
            orient="horizontal",
            resolution=0.05,
            variable=self.threshold,
            length=300,
        )
        threshold_slider.pack()

        # --- Skip frames selection (2-10) ---
        tk.Label(root, text="", height=1).pack()
        tk.Label(root, text="Skip Frames", font=("Arial", 12, "bold")).pack()
        self.skip_frames = tk.IntVar(value=4)
        skip_frame_frame = tk.Frame(root)
        skip_frame_frame.pack()
        for n in range(2, 11):
            tk.Radiobutton(
                skip_frame_frame, text=str(n), variable=self.skip_frames, value=n
            ).pack(side="left")

        # --- Event Duration Slider ---
        tk.Label(root, text="", height=1).pack()
        tk.Label(root, text="Event Duration (how many frames it'll blend together)", font=("Arial", 12, "bold")).pack()
        self.duration_sec = tk.DoubleVar(value=0.1)
        duration_slider = tk.Scale(
            root,
            from_=0.1,
            to=0.7,
            orient="horizontal",
            resolution=0.1,
            variable=self.duration_sec,
            length=300,
        )
        duration_slider.pack()

        # --- Cooldown Slider (logarithmic) ---
        tk.Label(root, text="", height=1).pack()
        tk.Label(root, text="Cooldown (after it detects an event)", font=("Arial", 12, "bold")).pack()
        self.cooldown_raw = tk.DoubleVar(value=0.0)  # raw slider 0→1
        cooldown_slider = tk.Scale(
            root,
            from_=0,
            to=1,
            orient="horizontal",
            resolution=0.01,
            variable=self.cooldown_raw,
            length=300,
            showvalue=False,
            command=self.update_cooldown_label
        )
        cooldown_slider.pack()
        self.cooldown_label = tk.Label(root, text="Cooldown: 7 s")
        self.cooldown_label.pack()
        self.cooldown_sec = 7  # default value

        # --- Model complexity selection (0-2) ---
        tk.Label(root, text="", height=1).pack()
        tk.Label(root, text="Model Complexity", font=("Arial", 12, "bold")).pack()
        self.model_complexity = tk.IntVar(value=0)
        model_frame = tk.Frame(root)
        model_frame.pack()

        complexity_labels = ["Fast!", "Balanced", "Extra precise"]
        for n, label in enumerate(complexity_labels):
            tk.Radiobutton(
                model_frame,
                text=label,
                variable=self.model_complexity,
                value=n
            ).pack(side="left")

        # --- ROI selection toggle ---
        tk.Label(root, text="", height=1).pack()
        self.roi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            root,
            text="Analyze entire frame (skip ROI selection)\n*Not recommended",
            variable=self.roi_var
        ).pack()

        # --- File selectors ---
        tk.Label(root, text="", height=1).pack()
        tk.Button(root, text="Select Reference Video", command=self.select_ref).pack()
        tk.Label(root, textvariable=self.ref_path).pack()
        tk.Button(root, text="Select Session Video", command=self.select_main).pack()
        tk.Label(root, textvariable=self.main_path).pack()

        # --- Progress bar ---
        self.progress = ttk.Progressbar(root, length=300)
        self.progress.pack(pady=10)

        # --- Run button ---
        self.run_button = tk.Button(root, text="Analyse", command=self.run)
        self.run_button.pack()

        # --- Output ---
        self.output = tk.Text(root, height=10)
        self.output.pack()

        # --- Thread control ---
        self._thread = None
        self._stop_event = threading.Event()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------- Cooldown mapping -------------------
    def update_cooldown_label(self, val):
        # val from slider 0→1, map exponentially 3 → 60
        x = float(val)
        self.cooldown_sec = round(3 * ((60 / 3) ** x))
        self.cooldown_label.config(text=f"Cooldown: {self.cooldown_sec} s")

    # ------------------- File selection -------------------
    def select_ref(self):
        path = filedialog.askopenfilename()
        self.ref_path.set(path)

    def select_main(self):
        path = filedialog.askopenfilename()
        self.main_path.set(path)

    # ------------------- Progress bar -------------------
    def safe_update_progress(self, progress):
        self.root.after(0, self._update_progress_ui, progress)

    def _update_progress_ui(self, progress):
        progress = max(0, min(1, progress))
        self.progress["value"] = progress * 100

    # ------------------- Run -------------------
    def run(self):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Analyzing video...\n")
        self.run_button.config(state="disabled", text="Analyzing...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_detection, daemon=True)
        self._thread.start()

    # ------------------- Background detection -------------------
    def _run_detection(self):
        try:
            if not self.ref_path.get():
                raise ValueError("Reference video not selected")
            if not self.main_path.get():
                raise ValueError("Session video not selected")

            roi = None
            if not self.roi_var.get():
                roi = select_roi(self.main_path.get())

            results = run_detection(
                reference_video_path=self.ref_path.get(),
                main_video_path=self.main_path.get(),
                similarity_threshold=self.threshold.get(),
                skip_frames=self.skip_frames.get(),
                duration_sec=self.duration_sec.get(),
                cooldown_sec=self.cooldown_sec,
                model_complexity=self.model_complexity.get(),
                roi=roi,
                progress_callback=self.safe_update_progress,
                stop_event=self._stop_event,
            )

            if not self._stop_event.is_set():
                self.root.after(0, self.show_results, results)

        except ValueError as e:
            msg = str(e)
            if self.ref_path.get() and "Reference" in msg:
                msg = "Reference video error:\n" + msg
            elif self.main_path.get() and "Session" in msg:
                msg = "Session video error:\n" + msg
            self.root.after(0, self.show_error, msg)

        except Exception as e:
            if not self._stop_event.is_set():
                self.root.after(0, self.show_error, f"Unexpected error:\n{str(e)}")

    # ------------------- Show output -------------------
    def show_results(self, results):
        self.output.delete("1.0", tk.END)
        if results:
            # Add a header before listing timestamps
            output_text = "Events start at (hh:mm:ss):\n" + "\n".join(results)
        else:
            output_text = "No events detected."
        self.output.insert(tk.END, output_text)
        self.run_button.config(state="normal", text="Analyse")

    def show_error(self, message):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"ERROR:\n{message}")
        self.run_button.config(state="normal", text="Analyse")

    # ------------------- Window close -------------------
    def on_close(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()