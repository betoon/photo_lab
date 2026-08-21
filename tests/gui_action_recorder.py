"""Small manual GUI action recorder helper.

Use this when a Tkinter app needs button-by-button verification. It does not
control the app by itself; it gives you a repeatable checklist log while you
click through the target application.
"""

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk


LOG_DIR = Path(__file__).resolve().parents[1] / "test_logs"
LOG_DIR.mkdir(exist_ok=True)


class GuiActionRecorder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GUI Action Recorder")
        self.geometry("760x520")
        self.log_path = LOG_DIR / f"gui_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self.steps = []
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Action / Button / Menu item").pack(anchor="w")
        self.action_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.action_var).pack(fill="x", pady=(3, 8))
        ttk.Label(top, text="Expected result").pack(anchor="w")
        self.expected_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.expected_var).pack(fill="x", pady=(3, 8))
        ttk.Button(top, text="Record Step", command=self.record_step).pack(side="left")
        ttk.Button(top, text="Mark Last Step Passed", command=lambda: self.mark_last("PASS")).pack(side="left", padx=6)
        ttk.Button(top, text="Mark Last Step Failed", command=lambda: self.mark_last("FAIL")).pack(side="left")
        ttk.Button(top, text="Save Log", command=self.save_log).pack(side="right")

        self.output = tk.Text(self, wrap="word", padx=8, pady=8)
        self.output.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def record_step(self):
        action = self.action_var.get().strip() or "(unnamed action)"
        expected = self.expected_var.get().strip() or "(expected result not entered)"
        self.steps.append({"action": action, "expected": expected, "result": "PENDING"})
        self.render()
        self.action_var.set("")
        self.expected_var.set("")

    def mark_last(self, result):
        if self.steps:
            self.steps[-1]["result"] = result
            self.render()

    def render(self):
        self.output.delete("1.0", "end")
        for index, step in enumerate(self.steps, start=1):
            self.output.insert("end", f"{index}. [{step['result']}] {step['action']}\n   Expected: {step['expected']}\n\n")

    def save_log(self):
        lines = ["# GUI Action Recorder Log", ""]
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"{index}. **{step['result']}** {step['action']}")
            lines.append(f"   Expected: {step['expected']}")
            lines.append("")
        self.log_path.write_text("\n".join(lines), encoding="utf-8")
        self.title(f"GUI Action Recorder - saved {self.log_path.name}")


if __name__ == "__main__":
    GuiActionRecorder().mainloop()
