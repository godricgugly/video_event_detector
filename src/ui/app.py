import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import cv2
from PIL import Image, ImageTk

from src.app.pipeline import run_detection
from src.pose.detector import PoseDetector
from src.pose.normalization import normalize_landmarks
from src.reference.reference_builder import build_reference_pose


def get_frame(video_path, frame_index=2):
    """Retrieve a specific frame from a video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read reference frame")

    return frame


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Trick Finder!")

        # ------------ initial opening config ------------
        w, h = 1100, 900
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # ---------------- scrollable root ----------------
        outer_frame = tk.Frame(root)
        outer_frame.pack(fill="both", expand=True)

        self.canvas_scroll = tk.Canvas(outer_frame)
        scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=self.canvas_scroll.yview)

        self.scrollable_frame = tk.Frame(self.canvas_scroll)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )

        self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.configure(yscrollcommand=scrollbar.set)

        self.canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        self.canvas_scroll.bind_all("<MouseWheel>", self._on_mousewheel)   # Windows / macOS
        self.canvas_scroll.bind_all("<Button-4>", self._on_mousewheel_linux)  # Linux scroll up
        self.canvas_scroll.bind_all("<Button-5>", self._on_mousewheel_linux)  # Linux scroll down

        # ---------------- file paths ----------------
        self.ref_path = tk.StringVar()
        self.main_path = tk.StringVar()
        self.ref_path_full = None
        self.main_path_full = None

        # ---------------- busy state ----------------
        self._busy = False

        # ---------------- ROI state ----------------
        self.selected_roi = None
        self.display_scale = 1.0
        self.roi_start = None
        self.roi_rect = None

        # ---------------- cooldown ----------------
        self.cooldown_sec = 7

        # ---------------- layout ----------------
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Arial", 12, "bold"))
        notebook = ttk.Notebook(self.scrollable_frame)
        notebook.pack(fill="both", expand=True)

        # Home tab
        home_tab = tk.Frame(notebook)
        notebook.add(home_tab, text="Home")
        main_container = tk.Frame(home_tab)
        main_container.pack()

        # How-to tab
        howto_tab = tk.Frame(notebook)
        notebook.add(howto_tab, text="How to use")
        howto_text = tk.Text(howto_tab, wrap="word")
        howto_text.pack(fill="both", expand=True)
        howto_content = """
While filming your session:

1.  Make sure the camera is static.

2.  When you first get on the line (around the location where you'll be freestyling), make your pose and hold it for 3-4s (ej. look at the camera and cross your arms).

3.  Every time something happens that you want to locate after, make your pose again and hold it for 2-3s (the more similar you make the pose, the easiar it'll be for the program to find it!)

4.  Take the clip of your reference pose with a video editor of your choice (should be easy to locate, since you did it at the beginning of your session, right? ;) ).


Then to use the program:

1. Adjust (or leave in the default setting):
- Similarity Threshold
- Skip Frames
- Event Duration
- Cooldown

2. Select a reference video
→ This defines the pose you want to detect

3. (Optional) Draw a region of interest (ROI)
→ Focus detection on a specific area

4. Click "Build Reference Pose"

5. Select your session video

6. Click "Analyse"

7. Go chill... or, look obsesively at that progress bar, up to you!

Tips:
- Lower threshold = more detections (but more false positives)
- ROI improves speed and accuracy
- If soemthing doesn't work, or you clicked a lot of times in the reference pose → try closing and opening again!
- If you forgot to do a reference pose during your session, you can look for something you're likely to do after something that you'd want to access has happened (ej. Sitting on the line to rest up?... you know yourself on the line, you decide ;) )
"""
        howto_text.insert("1.0", howto_content)
        howto_text.config(state="disabled")

        # Left controls
        left_frame = tk.Frame(main_container)
        left_frame.pack(side="left")
        self._build_controls(left_frame)

        # Right ROI and info
        self.right_frame = tk.Frame(main_container)
        self.right_frame.pack(side="right", padx=10, fill="y")

        # Header (always visible)
        self.header_label = tk.Label(
            self.right_frame,
            text="TrickFinder proudly brought to you by Liam!\n\nCheck out the code at:\nhttps://github.com/godricgugly/video_event_detector",
            justify="center",
            wraplength=500,
            font=("Arial", 12, "bold")
        )
        self.header_label.pack(pady=(10, 100))

        # ROI container (canvas + build button)
        self.roi_container = tk.Frame(self.right_frame)
        self.roi_container.pack(side="top", pady=5)
        self._build_roi_panel(self.roi_container)

        # Footer (always visible)
        self.footer_label = tk.Label(
            self.right_frame,
            text="If you want to send support,\nyou can tip me here ;)\n\nhttps://buymeacoffee.com/liam_watford",
            justify="center",
            wraplength=500,
            font=("Arial", 12, "bold")
        )
        self.footer_label.pack(side="bottom", pady=50)

        # ---------------- threading ----------------
        self._thread = None
        self._stop_event = threading.Event()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- mouse scroll ----------------
    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas_scroll.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas_scroll.yview_scroll(1, "units")

    # ---------------- build controls ----------------
    def _build_controls(self, parent):
        tk.Label(parent, text="Similarity Threshold", font=("Arial", 12, "bold")).pack()
        self.threshold = tk.DoubleVar(value=0.25)
        tk.Scale(parent, from_=0.1, to=0.7, resolution=0.05,
                 orient="horizontal", variable=self.threshold, length=300).pack()

        tk.Label(parent).pack()
        tk.Label(parent, text="Skip Frames", font=("Arial", 12, "bold")).pack()
        self.skip_frames = tk.IntVar(value=4)
        f = tk.Frame(parent)
        f.pack()
        for n in range(2, 11):
            tk.Radiobutton(f, text=str(n), variable=self.skip_frames, value=n).pack(side="left")

        tk.Label(parent).pack()
        tk.Label(parent, text="Event Duration", font=("Arial", 12, "bold")).pack()
        self.duration_sec = tk.DoubleVar(value=0.1)
        tk.Scale(parent, from_=0.1, to=0.7, resolution=0.1,
                 orient="horizontal", variable=self.duration_sec, length=300).pack()

        tk.Label(parent).pack()
        tk.Label(parent, text="Cooldown", font=("Arial", 12, "bold")).pack()
        self.cooldown_raw = tk.DoubleVar(value=0.0)
        tk.Scale(parent, from_=0, to=1, resolution=0.01,
                 orient="horizontal", variable=self.cooldown_raw,
                 length=300, showvalue=False,
                 command=self.update_cooldown_label).pack()
        self.cooldown_label = tk.Label(parent, text=f"Cooldown: {self.cooldown_sec} s")
        self.cooldown_label.pack()

        tk.Label(parent).pack()
        tk.Label(parent, text="Model Complexity", font=("Arial", 12, "bold")).pack()
        self.model_complexity = tk.IntVar(value=0)
        mf = tk.Frame(parent)
        mf.pack()
        for i, label in enumerate(["Fast!", "Balanced", "Extra precise"]):
            tk.Radiobutton(mf, text=label, variable=self.model_complexity, value=i).pack(side="left")

        tk.Label(parent).pack()
        self.roi_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            parent,
            text="Analyze entire frame (skip ROI selection)\n*Not recommended",
            variable=self.roi_var,
            command=self.toggle_roi_panel
        ).pack()

        tk.Label(parent).pack()
        tk.Button(parent, text="Select Reference Video", command=self.select_ref).pack()
        tk.Label(parent, textvariable=self.ref_path).pack()

        tk.Button(parent, text="Select Session Video", command=self.select_main).pack()
        tk.Label(parent, textvariable=self.main_path).pack()

        self.progress = ttk.Progressbar(parent, length=300)
        self.progress.pack(pady=10)

        self.run_button = tk.Button(parent, text="Analyse", command=self.run)
        self.run_button.pack()

        # -------- output with scrollbar --------
        output_frame = tk.Frame(parent)
        output_frame.pack(fill="both", expand=True)

        self.output = tk.Text(output_frame, height=10, wrap="word")
        output_scroll = tk.Scrollbar(output_frame, command=self.output.yview)
        self.output.configure(yscrollcommand=output_scroll.set)
        self.output.pack(side="left", fill="both", expand=True)
        output_scroll.pack(side="right", fill="y")

    # ---------------- build ROI panel ----------------
    def _build_roi_panel(self, parent):
        self.roi_label = tk.Label(parent, text="Region of interest selection", font=("Arial", 11, "bold"))
        self.roi_label.pack()
        self.canvas = tk.Canvas(parent, bg="black")
        self.canvas.pack()
        self.build_button = tk.Button(parent, text="Build Reference Pose", command=self.build_reference)
        self.build_button.pack(pady=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.toggle_roi_panel()

    # ---------------- busy lock ----------------
    def set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.build_button.config(state=state)
        self.run_button.config(state=state)

    def toggle_roi_panel(self):
        if self.roi_var.get():
            self.roi_container.pack_forget()
        else:
            self.roi_container.pack(side="top", pady=5)

    def update_cooldown_label(self, val):
        x = float(val)
        self.cooldown_sec = round(3 * ((60 / 3) ** x))
        self.cooldown_label.config(text=f"Cooldown: {self.cooldown_sec} s")

    def select_ref(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.ref_path_full = path
            self.ref_path.set(os.path.basename(path))

            self.selected_roi = None
            frame = get_frame(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            max_w, max_h = 500, 400
            self.display_scale = min(max_w / w, max_h / h, 1.0)

            disp_w = int(w * self.display_scale)
            disp_h = int(h * self.display_scale)

            img = Image.fromarray(frame).resize((disp_w, disp_h))
            self.tk_img = ImageTk.PhotoImage(img)

            self.canvas.config(width=disp_w, height=disp_h)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def select_main(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.main_path_full = path
            self.main_path.set(os.path.basename(path))

    def on_mouse_down(self, event):
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)

    def on_mouse_drag(self, event):
        if not self.roi_start:
            return
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
        x0, y0 = self.roi_start
        self.roi_rect = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="red", width=2)

    def on_mouse_up(self, event):
        if not self.roi_start:
            return
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y

        scale = self.display_scale
        w, h = abs(x1 - x0), abs(y1 - y0)

        if w < 5 or h < 5:
            self.roi_start = None
            if self.roi_rect:
                self.canvas.delete(self.roi_rect)
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Please select a region of interest.")
            return

        self.selected_roi = (
            int(min(x0, x1) / scale),
            int(min(y0, y1) / scale),
            int(w / scale),
            int(h / scale),
        )


    def build_reference(self):
        if self._busy:
            return

        self.set_busy(True)
        self.output.delete("1.0", tk.END)

        if not self.ref_path_full:
            self.output.insert(tk.END, "Please select a reference video.")
            self.set_busy(False)
            return

        self.output.insert(tk.END, "Building reference pose...\n")

        def worker():
            cap = cv2.VideoCapture(self.ref_path_full)
            detector = PoseDetector(model_complexity=self.model_complexity.get())
            landmarks = []

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if self.selected_roi is not None:
                        x, y, w, h = self.selected_roi
                        frame = frame[y:y+h, x:x+w]

                    lm = detector.process(frame)
                    if lm is not None:
                        landmarks.append(normalize_landmarks(lm))

                if not landmarks:
                    raise ValueError("Could not build reference pose.")

                self.reference_pose = build_reference_pose(landmarks)
                self.root.after(0, self._on_reference_success)

            except Exception as e:
                self.root.after(0, self._on_reference_error, str(e))

            finally:
                cap.release()
                detector.close()
                self.root.after(0, lambda: self.set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _on_reference_success(self):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Reference pose built, ready to analyse!")
        if self.roi_rect:
            self.canvas.itemconfig(self.roi_rect, outline="green")

    def _on_reference_error(self, msg):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"ERROR:\n{msg}")
        if self.roi_rect:
            self.canvas.itemconfig(self.roi_rect, outline="red")

    def run(self):
        if self._busy:
            return

        self.set_busy(True)
        self.output.delete("1.0", tk.END)

        if not self.ref_path_full:
            self.output.insert(tk.END, "Please select a reference video.")
            self.set_busy(False)
            return

        if not self.main_path_full:
            self.output.insert(tk.END, "Please select a session video.")
            self.set_busy(False)
            return

        self.output.insert(tk.END, "Analyzing video...\n")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_detection, daemon=True)
        self._thread.start()

    def _run_detection(self):
        try:
            roi = None if self.roi_var.get() else self.selected_roi

            results = run_detection(
                reference_video_path=self.ref_path_full,
                main_video_path=self.main_path_full,
                similarity_threshold=self.threshold.get(),
                skip_frames=self.skip_frames.get(),
                duration_sec=self.duration_sec.get(),
                cooldown_sec=self.cooldown_sec,
                model_complexity=self.model_complexity.get(),
                roi=roi,
                progress_callback=self.safe_update_progress,
                stop_event=self._stop_event,
            )

            self.root.after(0, self.show_results, results)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def safe_update_progress(self, progress):
        self.root.after(0, lambda: self.progress.configure(value=progress * 100))

    def show_results(self, results):
        self.set_busy(False)
        self.output.delete("1.0", tk.END)
        if results:
            self.output.insert(tk.END, "Events:\n" + "\n".join(results))
            self.output.insert(tk.END, "\n\nThanks for using the Trickfinder, I hope you landed awesome stuff! =D")
        else:
            self.output.insert(tk.END, "No events detected.")

    def show_error(self, msg):
        self.set_busy(False)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"ERROR:\n{msg}")

    def on_close(self):
        self._stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()