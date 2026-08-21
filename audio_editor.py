import os
import sys
import tempfile
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pydub import AudioSegment
from pydub.effects import normalize
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import logging

log = logging.getLogger(__name__)


class AudioEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panorama Audio Editor")
        self.resize(1280, 720)

        self.audio = None
        self.audio_path = None
        self.is_playing = False
        self.selection_start = None
        self.selection_end = None
        self.temp_file = None
        self.progress_line = None
        self.playback_offset = 0.0
        self.undo_audio = None
        self.undo_message = None
        self.waveform_color = "#ffd400"
        self.background_color = "#050505"
        self.grid_color = "#7a6b00"

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.setup_ui()
        self.apply_app_style()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("Open Audio")
        self.btn_play = QPushButton("Play")
        self.btn_stop = QPushButton("Stop")
        self.btn_cut = QPushButton("Cut Selection")
        self.btn_preview_selection = QPushButton("Play Selection")
        self.btn_export_selection = QPushButton("Export Selection")
        self.btn_fade_selection = QPushButton("Fade Selection")
        self.btn_normalize = QPushButton("Normalize Volume")
        self.btn_undo = QPushButton("Undo Last Edit")
        self.btn_clear = QPushButton("Clear Marks")
        self.btn_export = QPushButton("Export")

        self.btn_open.clicked.connect(self.load_audio)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_cut.clicked.connect(self.cut_selection)
        self.btn_preview_selection.clicked.connect(self.play_selected_section)
        self.btn_export_selection.clicked.connect(self.export_selected_section)
        self.btn_fade_selection.clicked.connect(self.fade_selected_section)
        self.btn_normalize.clicked.connect(self.normalize_audio)
        self.btn_undo.clicked.connect(self.undo_last_edit)
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_export.clicked.connect(self.export_audio)

        self.btn_cut.setStyleSheet("background-color: #b91c1c; color: white; font-weight: bold;")

        for button in (
            self.btn_open,
            self.btn_play,
            self.btn_stop,
            self.btn_cut,
            self.btn_preview_selection,
            self.btn_export_selection,
            self.btn_fade_selection,
            self.btn_normalize,
            self.btn_undo,
            self.btn_clear,
            self.btn_export,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Black / Yellow", "Light", "Blue"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        toolbar.addWidget(QLabel("Theme:"))
        toolbar.addWidget(self.theme_combo)
        layout.addLayout(toolbar)

        fade_layout = QHBoxLayout()
        fade_layout.addWidget(QLabel("Fade In:"))
        self.fade_in_slider = QSlider(Qt.Orientation.Horizontal)
        self.fade_in_slider.setRange(0, 5000)
        self.fade_in_slider.setValue(1000)
        fade_layout.addWidget(self.fade_in_slider)
        self.fade_in_label = QLabel("1000 ms")
        fade_layout.addWidget(self.fade_in_label)

        fade_layout.addSpacing(30)
        fade_layout.addWidget(QLabel("Fade Out:"))
        self.fade_out_slider = QSlider(Qt.Orientation.Horizontal)
        self.fade_out_slider.setRange(0, 5000)
        self.fade_out_slider.setValue(1000)
        fade_layout.addWidget(self.fade_out_slider)
        self.fade_out_label = QLabel("1000 ms")
        fade_layout.addWidget(self.fade_out_label)
        layout.addLayout(fade_layout)

        self.file_label = QLabel("No audio loaded")
        self.selection_label = QLabel("Selection: click once for the start cut line, click again for the end cut line")
        layout.addWidget(self.file_label)
        layout.addWidget(self.selection_label)

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.configure_plot_style()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.fade_in_slider.valueChanged.connect(self.update_fade_labels)
        self.fade_out_slider.valueChanged.connect(self.update_fade_labels)

    def apply_app_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111111; color: #f8fafc; }
            QLabel { color: #f8fafc; }
            QPushButton { padding: 7px 12px; background: #262626; color: #f8fafc; border: 1px solid #525252; border-radius: 4px; }
            QPushButton:hover { background: #3f3f46; }
            QComboBox { padding: 5px; background: #262626; color: #f8fafc; border: 1px solid #525252; }
            QStatusBar { background: #050505; color: #ffd400; }
            """
        )

    def change_theme(self, theme):
        if theme == "Light":
            self.background_color = "#ffffff"
            self.waveform_color = "#1d4ed8"
            self.grid_color = "#94a3b8"
        elif theme == "Blue":
            self.background_color = "#071426"
            self.waveform_color = "#38bdf8"
            self.grid_color = "#1e3a8a"
        else:
            self.background_color = "#050505"
            self.waveform_color = "#ffd400"
            self.grid_color = "#7a6b00"
        self.configure_plot_style()
        if self.audio:
            self.draw_waveform()

    def configure_plot_style(self):
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        self.ax.tick_params(colors=self.waveform_color)
        self.ax.xaxis.label.set_color(self.waveform_color)
        self.ax.yaxis.label.set_color(self.waveform_color)
        self.ax.title.set_color(self.waveform_color)
        for spine in self.ax.spines.values():
            spine.set_color(self.waveform_color)
        self.fig.tight_layout()

    def update_fade_labels(self):
        self.fade_in_label.setText(f"{self.fade_in_slider.value()} ms")
        self.fade_out_label.setText(f"{self.fade_out_slider.value()} ms")

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        if not file_path:
            return

        try:
            self.stop_playback()
            self.audio = AudioSegment.from_file(file_path)
            self.audio_path = file_path
            self.selection_start = None
            self.selection_end = None
            self.undo_audio = None
            self.undo_message = None
            self.draw_waveform()
            self.refresh_file_label()
            self.update_selection_label()
            self.statusBar.showMessage(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.statusBar.showMessage(f"Error: {e}")

    def waveform_samples(self):
        samples = np.array(self.audio.get_array_of_samples())
        if self.audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples = samples.astype(np.float32)
        max_abs = np.max(np.abs(samples)) if len(samples) else 1.0
        if max_abs > 0:
            samples = samples / max_abs
        max_points = 120000
        if len(samples) > max_points:
            step = int(np.ceil(len(samples) / max_points))
            samples = samples[::step]
        return samples

    def draw_waveform(self):
        self.ax.clear()
        self.configure_plot_style()

        samples = self.waveform_samples()
        duration = len(self.audio) / 1000.0
        time_axis = np.linspace(0, duration, len(samples))

        self.ax.plot(time_axis, samples, color=self.waveform_color, linewidth=0.8)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Level")
        self.ax.set_title("Audio Waveform")
        self.ax.grid(True, color=self.grid_color, alpha=0.28)
        self.ax.set_xlim(0, max(duration, 0.1))
        self.ax.set_ylim(-1.05, 1.05)

        if self.selection_start is not None:
            if self.selection_end is not None:
                start, end = self.selected_range()
                self.ax.axvspan(start, end, color="#ffd400", alpha=0.22)
                self.ax.axvline(start, color="#ffd400", linestyle="--", linewidth=2)
                self.ax.axvline(end, color="#ffd400", linestyle="--", linewidth=2)
            else:
                self.ax.axvline(self.selection_start, color="#ffd400", linestyle="--", linewidth=2)

        self.update_selection_label()
        self.canvas.draw()

    def selected_range(self):
        start = min(self.selection_start, self.selection_end)
        end = max(self.selection_start, self.selection_end)
        return start, end

    def has_selection(self):
        return self.audio is not None and self.selection_start is not None and self.selection_end is not None

    def selected_audio_segment(self):
        if not self.has_selection():
            return None, None, None
        start, end = self.selected_range()
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        return self.audio[start_ms:end_ms], start_ms, end_ms

    def remember_undo(self, message):
        if self.audio is not None:
            self.undo_audio = self.audio[:]
            self.undo_message = message

    def refresh_file_label(self):
        if not self.audio:
            self.file_label.setText("No audio loaded")
            return
        name = os.path.basename(self.audio_path) if self.audio_path else "Edited audio"
        self.file_label.setText(f"Loaded: {name} | Duration: {self.format_time(len(self.audio) / 1000.0)}")

    def format_time(self, seconds):
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        return f"{minutes}:{secs:05.2f}"

    def update_selection_label(self):
        if self.audio is None:
            self.selection_label.setText("Selection: no audio loaded")
        elif self.selection_start is None:
            self.selection_label.setText("Selection: click once for start, click again for end")
        elif self.selection_end is None:
            self.selection_label.setText(f"Start cut line: {self.format_time(self.selection_start)} | click again to set the end line")
        else:
            start, end = self.selected_range()
            self.selection_label.setText(
                f"Selected section: {self.format_time(start)} to {self.format_time(end)} | Length: {self.format_time(end - start)}"
            )

    def toggle_play(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self, audio_segment=None, offset_sec=0.0):
        if not self.audio and audio_segment is None:
            return

        self.stop_playback()
        playback_audio = audio_segment if audio_segment is not None else self.audio
        self.playback_offset = offset_sec
        self.is_playing = True
        self.btn_play.setText("Pause")

        fd, self.temp_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        playback_audio.export(self.temp_file, format="wav")

        def play_thread():
            try:
                pygame.mixer.music.load(self.temp_file)
                pygame.mixer.music.play()
                start_time = time.time()
                while self.is_playing and pygame.mixer.music.get_busy():
                    self.update_progress_line(self.playback_offset + time.time() - start_time)
                    time.sleep(0.05)
            except Exception as e:
                log.warning("Playback error: %s", e)
            finally:
                self.is_playing = False
                self.playback_offset = 0.0
                self.btn_play.setText("Play")
                self.remove_progress_line()
                self.cleanup_temp_file()

        threading.Thread(target=play_thread, daemon=True).start()

    def play_selected_section(self):
        segment, start_ms, end_ms = self.selected_audio_segment()
        if segment is None or end_ms <= start_ms:
            self.statusBar.showMessage("Select a section first")
            return
        self.start_playback(segment, start_ms / 1000.0)
        self.statusBar.showMessage(f"Playing selected section: {self.format_time((end_ms - start_ms) / 1000.0)}")

    def update_progress_line(self, pos_sec):
        try:
            if self.progress_line is None:
                self.progress_line = self.ax.axvline(x=pos_sec, color="#ef4444", linewidth=2)
            else:
                self.progress_line.set_xdata([pos_sec])
            self.canvas.draw_idle()
        except Exception:
            log.debug("update_progress_line: non-critical failure, continuing", exc_info=True)

    def remove_progress_line(self):
        if self.progress_line:
            try:
                self.progress_line.remove()
            except Exception:
                log.debug("remove_progress_line: non-critical failure, continuing", exc_info=True)
            self.progress_line = None
            try:
                self.canvas.draw_idle()
            except Exception:
                log.debug("remove_progress_line: non-critical failure, continuing", exc_info=True)

    def stop_playback(self):
        self.is_playing = False
        try:
            pygame.mixer.music.stop()
        except Exception:
            log.debug("stop_playback: non-critical failure, continuing", exc_info=True)
        self.btn_play.setText("Play")
        self.remove_progress_line()
        self.cleanup_temp_file()

    def cleanup_temp_file(self):
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except Exception:
                log.debug("cleanup_temp_file: non-critical failure, continuing", exc_info=True)
        self.temp_file = None

    def on_canvas_click(self, event):
        if not self.audio or event.inaxes != self.ax or event.xdata is None:
            return

        duration = len(self.audio) / 1000.0
        x = float(np.clip(event.xdata, 0.0, duration))
        if self.selection_start is None or self.selection_end is not None:
            self.selection_start = x
            self.selection_end = None
        else:
            self.selection_end = x
        self.draw_waveform()

    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
        if self.audio:
            self.draw_waveform()
        else:
            self.update_selection_label()
        self.statusBar.showMessage("Selection cleared")

    def cut_selection(self):
        segment, start_ms, end_ms = self.selected_audio_segment()
        if segment is None:
            self.statusBar.showMessage("Click twice on the waveform to select an area")
            return
        if end_ms <= start_ms:
            self.statusBar.showMessage("Selection is too short to cut")
            return

        self.remember_undo("cut")
        self.audio = self.audio[:start_ms] + self.audio[end_ms:]
        self.statusBar.showMessage(f"Cut {self.format_time((end_ms - start_ms) / 1000.0)} from the audio")
        self.selection_start = None
        self.selection_end = None
        self.draw_waveform()
        self.refresh_file_label()

    def fade_selected_section(self):
        segment, start_ms, end_ms = self.selected_audio_segment()
        if segment is None:
            self.statusBar.showMessage("Select a section to fade first")
            return
        if end_ms <= start_ms:
            self.statusBar.showMessage("Selection is too short to fade")
            return

        fade_in = min(self.fade_in_slider.value(), len(segment))
        fade_out = min(self.fade_out_slider.value(), len(segment))
        edited = segment.fade_in(fade_in).fade_out(fade_out)
        self.remember_undo("fade selection")
        self.audio = self.audio[:start_ms] + edited + self.audio[end_ms:]
        self.statusBar.showMessage(f"Faded selected section: {self.format_time(len(segment) / 1000.0)}")
        self.draw_waveform()

    def normalize_audio(self):
        if not self.audio:
            self.statusBar.showMessage("Open an audio file first")
            return
        self.remember_undo("normalize volume")
        self.audio = normalize(self.audio)
        self.draw_waveform()
        self.refresh_file_label()
        self.statusBar.showMessage("Normalized volume")

    def undo_last_edit(self):
        if self.undo_audio is None:
            self.statusBar.showMessage("Nothing to undo")
            return
        current_audio = self.audio
        self.audio = self.undo_audio
        self.undo_audio = current_audio
        label = self.undo_message or "edit"
        self.selection_start = None
        self.selection_end = None
        self.draw_waveform()
        self.refresh_file_label()
        self.statusBar.showMessage(f"Undid last {label}")

    def export_selected_section(self):
        segment, start_ms, end_ms = self.selected_audio_segment()
        if segment is None or end_ms <= start_ms:
            self.statusBar.showMessage("Select a section to export first")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Selected Audio", "selected_audio.wav", "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac);;OGG (*.ogg)"
        )
        if not save_path:
            return

        try:
            fmt = save_path.split(".")[-1].lower()
            segment.export(save_path, format=fmt)
            self.statusBar.showMessage(f"Saved selected section: {os.path.basename(save_path)}")
        except Exception as e:
            self.statusBar.showMessage(f"Save error: {e}")

    def export_audio(self):
        if not self.audio:
            self.statusBar.showMessage("Open an audio file first")
            return

        fade_in = self.fade_in_slider.value()
        fade_out = self.fade_out_slider.value()
        edited = self.audio
        if fade_in > 0:
            edited = edited.fade_in(fade_in)
        if fade_out > 0:
            edited = edited.fade_out(fade_out)

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac);;OGG (*.ogg)"
        )
        if not save_path:
            return

        try:
            fmt = save_path.split(".")[-1].lower()
            edited.export(save_path, format=fmt)
            self.statusBar.showMessage(f"Saved: {os.path.basename(save_path)}")
        except Exception as e:
            self.statusBar.showMessage(f"Save error: {e}")

    def closeEvent(self, event):
        self.stop_playback()
        pygame.mixer.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioEditor()
    window.show()
    sys.exit(app.exec())