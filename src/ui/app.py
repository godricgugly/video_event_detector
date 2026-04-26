import tkinter as tk
from tkinter import filedialog, ttk
import threading

from src.app.pipeline import run_detection
from src.video.roi import select_roi


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Event Detector")

        self.ref_path = tk.StringVar()
        self.main_path = tk.StringVar()
        self.threshold = tk.DoubleVar(value=0.25)

        # --- File selectors
        tk.Button(root, text="Select Reference Video", command=self.select_ref).pack()
        tk.Label(root, textvariable=self.ref_path).pack()

        tk.Button(root, text="Select Session Video", command=self.select_main).pack()
        tk.Label(root, textvariable=self.main_path).pack()

        # --- Threshold
        tk.Label(root, text="Similarity Threshold").pack()
        tk.Entry(root, textvariable=self.threshold).pack()

        # --- Progress bar
        self.progress = ttk.Progressbar(root, length=300)
        self.progress.pack(pady=10)

        # --- Run button
        self.run_button = tk.Button(root, text="Analyse", command=self.run)
        self.run_button.pack()

        # --- Output
        self.output = tk.Text(root, height=10)
        self.output.pack()

        # --- Thread control
        self._thread = None
        self._stop_event = threading.Event()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
        # Clear old output immediately
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Analyzing video...\n")

        # Disable button to prevent multiple runs
        self.run_button.config(state="disabled", text="Analyzing...")

        # Clear stop event for new run
        self._stop_event.clear()

        # Start background thread
        self._thread = threading.Thread(target=self._run_detection, daemon=True)
        self._thread.start()

    # ------------------- Background detection -------------------
    def _run_detection(self):
        try:
            if not self.ref_path.get():
                raise ValueError("Reference video not selected")
            if not self.main_path.get():
                raise ValueError("Session video not selected")

            roi = select_roi(self.main_path.get())

            # Pass stop_event to pipeline for clean early exit
            results = run_detection(
                reference_video_path=self.ref_path.get(),
                main_video_path=self.main_path.get(),
                similarity_threshold=self.threshold.get(),
                roi=roi,
                progress_callback=self.safe_update_progress,
                stop_event=self._stop_event,
            )

            if not self._stop_event.is_set():
                self.root.after(0, self.show_results, results)

        except ValueError as e:
            msg = str(e)
            # Enrich error context
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
        self.output.insert(tk.END, "\n".join(results))
        self.run_button.config(state="normal", text="Analyse")

    def show_error(self, message):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"ERROR:\n{message}")
        self.run_button.config(state="normal", text="Analyse")

    # ------------------- Window close -------------------
    def on_close(self):
        # Signal thread to stop
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            # Wait a short moment for thread to exit cleanly
            self._thread.join(timeout=2)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()