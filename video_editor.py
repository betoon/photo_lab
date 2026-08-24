import os
import sys
import re
import time
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as cctk

# Define theme and colors
cctk.set_appearance_mode("Dark")
cctk.set_default_color_theme("blue")

# Theme colors
COLOR_BG = "#121212"
COLOR_CARD = "#1E1E1E"
COLOR_BORDER = "#2D2D2D"
COLOR_ACCENT = "#7C4DFF"      # Vibrant purple for import/primary actions
COLOR_SUCCESS = "#00E676"     # Vibrant green for export
COLOR_CANCEL = "#D32F2F"      # Red for cancel / close
COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#AAAAAA"

class VideoEditorApp(cctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("VeloCut Studio - Video Editor - Brian E. Toon, 2026")
        self.geometry("1060x780")
        self.minsize(1020, 740)
        self.configure(fg_color=COLOR_BG)
        
        # App state
        self.video_path = None
        self.cap = None
        self.fps = 0.0
        self.total_frames = 0
        self.duration = 0.0
        self.width = 0
        self.height = 0
        self.has_audio = False
        
        self.current_frame = 0
        self.playing = False
        self.after_id = None
        
        # Cached frame for lag-free visual slider updates
        self.last_raw_frame = None
        
        # Background audio state
        self.bg_audio_path = None
        
        # Selection markers (frame numbers)
        self.start_frame = None
        self.end_frame = None
        
        # Export state
        self.exporting = False
        self.export_process = None
        self.export_thread = None
        
        # Build UI layout
        self.create_layout()
        
        # Bind close window event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_layout(self):
        # Configure grid layout: 1 row, 2 columns (Left sidebar + Right main editor)
        self.grid_columnconfigure(0, weight=0, minsize=330)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ==========================================
        # LEFT COLUMN (Sidebar Controls)
        # ==========================================
        self.sidebar = cctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0, border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Title Header
        self.header_label = cctk.CTkLabel(self.sidebar, text="🎬 VeloCut Studio", font=cctk.CTkFont(size=24, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        self.header_label.pack(anchor="w", padx=10, pady=(10, 2))
        
        self.subtitle_label = cctk.CTkLabel(self.sidebar, text="Enhanced Video Editor & FX Tool", font=cctk.CTkFont(size=12), text_color=COLOR_TEXT_SECONDARY)
        self.subtitle_label.pack(anchor="w", padx=10, pady=(0, 15))
        
        # Action: Import Button
        self.import_btn = cctk.CTkButton(
            self.sidebar, 
            text="📁 Import Video File", 
            font=cctk.CTkFont(size=14, weight="bold"), 
            fg_color=COLOR_ACCENT, 
            hover_color="#6200EA",
            height=38,
            command=self.import_video
        )
        self.import_btn.pack(fill="x", padx=10, pady=5)
        
        # Collapsible Video Properties & Cut Markers (merged to save space)
        self.meta_frame = cctk.CTkFrame(self.sidebar, fg_color=COLOR_CARD, border_color=COLOR_BORDER, border_width=1)
        self.meta_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_filename = cctk.CTkLabel(self.meta_frame, text="File: No video loaded", font=cctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_PRIMARY, wraplength=290, justify="left")
        self.lbl_filename.pack(anchor="w", padx=12, pady=(8, 2))
        
        self.lbl_props = cctk.CTkLabel(self.meta_frame, text="Resolution: --  |  FPS: --  |  Audio: --", font=cctk.CTkFont(size=11), text_color=COLOR_TEXT_SECONDARY)
        self.lbl_props.pack(anchor="w", padx=12, pady=2)
        
        # Marker summary
        self.lbl_selected_len = cctk.CTkLabel(self.meta_frame, text="Cut: 0.0s (Full)", font=cctk.CTkFont(size=11, weight="bold"), text_color=COLOR_ACCENT)
        self.lbl_selected_len.pack(anchor="w", padx=12, pady=(2, 8))
        
        # Main Speed control (Fundamental setting)
        self.speed_frame = cctk.CTkFrame(self.sidebar, fg_color=COLOR_BG)
        self.speed_frame.pack(fill="x", padx=10, pady=2)
        
        self.lbl_speed = cctk.CTkLabel(self.speed_frame, text="Export Speed Multiplier:", font=cctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        self.lbl_speed.pack(anchor="w", padx=5, pady=1)
        
        self.speed_options = [
            "0.25x (Very Slow)",
            "0.5x (Slow)",
            "0.75x (Slightly Slow)",
            "1.0x (Normal)",
            "1.25x (Slightly Fast)",
            "1.5x (Fast)",
            "2.0x (Very Fast)",
            "4.0x (Hyper Fast)"
        ]
        self.speed_combo = cctk.CTkComboBox(self.speed_frame, values=self.speed_options, state="readonly", height=28)
        self.speed_combo.set("1.0x (Normal)")
        self.speed_combo.pack(fill="x", padx=5, pady=1)
        
        # Video Codec Selector (Essential for Action Cameras / HEVC workflows)
        self.codec_frame = cctk.CTkFrame(self.sidebar, fg_color=COLOR_BG)
        self.codec_frame.pack(fill="x", padx=10, pady=2)
        
        self.lbl_codec = cctk.CTkLabel(self.codec_frame, text="Export Video Codec:", font=cctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        self.lbl_codec.pack(anchor="w", padx=5, pady=1)
        
        self.codec_options = [
            "H.264 (Most Compatible)",
            "H.265/HEVC (High Efficiency)"
        ]
        self.codec_combo = cctk.CTkComboBox(self.codec_frame, values=self.codec_options, state="readonly", height=28)
        self.codec_combo.set("H.264 (Most Compatible)")
        self.codec_combo.pack(fill="x", padx=5, pady=1)
        
        # Crop Selector (Social Media Ratios)
        self.crop_frame = cctk.CTkFrame(self.sidebar, fg_color=COLOR_BG)
        self.crop_frame.pack(fill="x", padx=10, pady=2)
        
        self.lbl_crop = cctk.CTkLabel(self.crop_frame, text="Crop Aspect Ratio:", font=cctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        self.lbl_crop.pack(anchor="w", padx=5, pady=1)
        
        self.crop_options = [
            "Original (Landscape)",
            "9:16 (Vertical - TikTok/Reels)",
            "1:1 (Square - Instagram)"
        ]
        self.crop_combo = cctk.CTkComboBox(self.crop_frame, values=self.crop_options, state="readonly", height=28, command=self.on_filter_change)
        self.crop_combo.set("Original (Landscape)")
        self.crop_combo.pack(fill="x", padx=5, pady=1)
        
        # ==========================================
        # TABVIEW FOR ADVANCED FX
        # ==========================================
        self.tabs = cctk.CTkTabview(self.sidebar, height=270, fg_color=COLOR_CARD, segmented_button_selected_color=COLOR_ACCENT)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.tab_audio = self.tabs.add("Audio")
        self.tab_fades = self.tabs.add("Fades")
        self.tab_filters = self.tabs.add("Filters")
        self.tab_title = self.tabs.add("Title")
        
        # --- AUDIO TAB SETUP ---
        self.lbl_audio_mode = cctk.CTkLabel(self.tab_audio, text="Audio Track Mode:", font=cctk.CTkFont(size=12, weight="bold"))
        self.lbl_audio_mode.pack(anchor="w", padx=10, pady=(5, 2))
        
        self.audio_modes = ["Original Audio", "Mute Audio", "Replace Audio"]
        self.audio_combo = cctk.CTkComboBox(self.tab_audio, values=self.audio_modes, state="readonly", height=28, command=self.on_audio_mode_change)
        self.audio_combo.set("Original Audio")
        self.audio_combo.pack(fill="x", padx=10, pady=5)
        
        self.btn_select_audio = cctk.CTkButton(
            self.tab_audio, 
            text="📁 Select Audio File", 
            fg_color="transparent", 
            border_width=1, 
            border_color=COLOR_BORDER,
            hover_color="#2A2A2A",
            state="disabled",
            command=self.select_audio_file
        )
        self.btn_select_audio.pack(fill="x", padx=10, pady=5)
        
        self.lbl_audio_filename = cctk.CTkLabel(self.tab_audio, text="Not applicable", font=cctk.CTkFont(size=11), text_color=COLOR_TEXT_SECONDARY, wraplength=270)
        self.lbl_audio_filename.pack(anchor="w", padx=10, pady=2)
        
        self.chk_loop_audio = cctk.CTkCheckBox(self.tab_audio, text="Loop audio track if short", font=cctk.CTkFont(size=11), fg_color=COLOR_ACCENT, hover_color="#6200EA", state="disabled")
        self.chk_loop_audio.pack(anchor="w", padx=10, pady=5)
        
        # --- FADES TAB SETUP ---
        self.chk_fade_in = cctk.CTkCheckBox(self.tab_fades, text="Fade In (Start of Clip)", font=cctk.CTkFont(size=12), fg_color=COLOR_ACCENT, hover_color="#6200EA")
        self.chk_fade_in.pack(anchor="w", padx=10, pady=(10, 5))
        
        self.chk_fade_out = cctk.CTkCheckBox(self.tab_fades, text="Fade Out (End of Clip)", font=cctk.CTkFont(size=12), fg_color=COLOR_ACCENT, hover_color="#6200EA")
        self.chk_fade_out.pack(anchor="w", padx=10, pady=5)
        
        self.lbl_fade_duration = cctk.CTkLabel(self.tab_fades, text="Fade Duration: 1.0s", font=cctk.CTkFont(size=12, weight="bold"))
        self.lbl_fade_duration.pack(anchor="w", padx=10, pady=(10, 2))
        
        self.fade_duration_slider = cctk.CTkSlider(
            self.tab_fades, 
            from_=0.5, 
            to=3.0, 
            number_of_steps=25, 
            fg_color="#333333", 
            progress_color=COLOR_ACCENT, 
            button_color=COLOR_ACCENT,
            command=self.on_fade_duration_change
        )
        self.fade_duration_slider.set(1.0)
        self.fade_duration_slider.pack(fill="x", padx=10, pady=5)
        
        # --- FILTERS TAB SETUP ---
        # Creative Presets
        self.lbl_preset = cctk.CTkLabel(self.tab_filters, text="Creative Visual Preset:", font=cctk.CTkFont(size=12, weight="bold"))
        self.lbl_preset.pack(anchor="w", padx=10, pady=(5, 2))
        
        self.preset_options = [
            "None (Manual Sliders)",
            "Cinematic (Warm & Rich)",
            "Vintage (Warm Sepia)",
            "Vivid Pop (Bright & Saturated)",
            "Noir (Dramatic B&W)",
            "High Contrast B&W (Silver Halide)",
            "Old Time Film (Dust & Scratches)"
        ]
        self.preset_combo = cctk.CTkComboBox(self.tab_filters, values=self.preset_options, state="readonly", height=28, command=self.on_preset_change)
        self.preset_combo.set("None (Manual Sliders)")
        self.preset_combo.pack(fill="x", padx=10, pady=2)
        
        # Brightness Control
        self.lbl_brightness = cctk.CTkLabel(self.tab_filters, text="Brightness: +0.00", font=cctk.CTkFont(size=11, weight="bold"))
        self.lbl_brightness.pack(anchor="w", padx=10, pady=(8, 1))
        
        self.brightness_slider = cctk.CTkSlider(
            self.tab_filters, 
            from_=-0.5, 
            to=0.5, 
            fg_color="#333333", 
            progress_color=COLOR_ACCENT, 
            button_color=COLOR_ACCENT,
            command=self.on_filter_change
        )
        self.brightness_slider.set(0.0)
        self.brightness_slider.pack(fill="x", padx=10, pady=1)
        
        # Contrast Control
        self.lbl_contrast = cctk.CTkLabel(self.tab_filters, text="Contrast: 1.0x", font=cctk.CTkFont(size=11, weight="bold"))
        self.lbl_contrast.pack(anchor="w", padx=10, pady=(5, 1))
        
        self.contrast_slider = cctk.CTkSlider(
            self.tab_filters, 
            from_=0.5, 
            to=2.0, 
            fg_color="#333333", 
            progress_color=COLOR_ACCENT, 
            button_color=COLOR_ACCENT,
            command=self.on_filter_change
        )
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill="x", padx=10, pady=1)
        
        # Grayscale Control
        self.chk_grayscale = cctk.CTkCheckBox(self.tab_filters, text="Black & White (Grayscale)", font=cctk.CTkFont(size=11), fg_color=COLOR_ACCENT, hover_color="#6200EA", command=self.on_filter_change)
        self.chk_grayscale.pack(anchor="w", padx=10, pady=5)
        
        # Reset Filters Button
        self.btn_reset_filters = cctk.CTkButton(
            self.tab_filters, 
            text="Reset Filters", 
            height=20,
            fg_color="transparent", 
            border_width=1, 
            border_color=COLOR_BORDER,
            hover_color="#2A2A2A",
            command=self.reset_filters
        )
        self.btn_reset_filters.pack(fill="x", padx=10, pady=5)
        
        # --- TITLE CARD TAB SETUP ---
        self.chk_title_active = cctk.CTkCheckBox(self.tab_title, text="Enable Title Card Overlay", font=cctk.CTkFont(size=12, weight="bold"), fg_color=COLOR_ACCENT, hover_color="#6200EA", command=self.on_filter_change)
        self.chk_title_active.pack(anchor="w", padx=10, pady=(10, 10))
        
        self.lbl_title_text = cctk.CTkLabel(self.tab_title, text="Enter Title Text (Centered):", font=cctk.CTkFont(size=11, weight="bold"))
        self.lbl_title_text.pack(anchor="w", padx=10, pady=(2, 2))
        
        self.txt_title_text = cctk.CTkEntry(self.tab_title, placeholder_text="e.g. My GoPro Adventure", height=28)
        self.txt_title_text.insert(0, "My Adventure")
        self.txt_title_text.pack(fill="x", padx=10, pady=5)
        self.txt_title_text.bind("<KeyRelease>", self.on_filter_change)
        
        self.lbl_title_info = cctk.CTkLabel(self.tab_title, text="Fades out automatically after 3 seconds.", font=cctk.CTkFont(size=11), text_color=COLOR_TEXT_SECONDARY)
        self.lbl_title_info.pack(anchor="w", padx=10, pady=5)
        
        # ==========================================
        # EXPORT INTERACTION
        # ==========================================
        self.export_btn = cctk.CTkButton(
            self.sidebar, 
            text="🚀 Export Cut Video", 
            font=cctk.CTkFont(size=14, weight="bold"), 
            fg_color=COLOR_SUCCESS, 
            hover_color="#00C853",
            height=40,
            state="disabled",
            command=self.start_export
        )
        self.export_btn.pack(fill="x", padx=10, pady=5)
        
        # Export Progress overlay frame (starts invisible)
        self.progress_frame = cctk.CTkFrame(self.sidebar, fg_color=COLOR_CARD, border_color=COLOR_BORDER, border_width=1)
        
        self.lbl_progress = cctk.CTkLabel(self.progress_frame, text="Exporting: 0%", font=cctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        self.lbl_progress.pack(anchor="w", padx=15, pady=(10, 2))
        
        self.progress_bar = cctk.CTkProgressBar(self.progress_frame, fg_color="#333333", progress_color=COLOR_SUCCESS)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=15, pady=(5, 10))
        
        self.cancel_export_btn = cctk.CTkButton(
            self.progress_frame, 
            text="Cancel Export", 
            fg_color=COLOR_CANCEL, 
            hover_color="#B71C1C",
            height=28,
            command=self.cancel_export
        )
        self.cancel_export_btn.pack(fill="x", padx=15, pady=(0, 10))
        
        # ==========================================
        # RIGHT COLUMN (Main Preview and Timelines)
        # ==========================================
        self.main_panel = cctk.CTkFrame(self, fg_color=COLOR_BG)
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Preview Container
        self.preview_container = cctk.CTkFrame(self.main_panel, fg_color="#080808", border_color=COLOR_BORDER, border_width=1, corner_radius=8)
        self.preview_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Dynamic Video Canvas/Label
        self.preview_label = cctk.CTkLabel(self.preview_container, text="Click 'Import Video File' to start editing", font=cctk.CTkFont(size=14), text_color=COLOR_TEXT_SECONDARY)
        self.preview_label.pack(expand=True)
        
        # Timeline Frame (Timeline slider and labels)
        self.timeline_frame = cctk.CTkFrame(self.main_panel, fg_color=COLOR_BG)
        self.timeline_frame.pack(fill="x", padx=10, pady=5)
        
        self.timeline_slider = cctk.CTkSlider(
            self.timeline_frame, 
            from_=0, 
            to=100, 
            height=16,
            fg_color="#333333", 
            progress_color=COLOR_ACCENT, 
            button_color=COLOR_ACCENT,
            command=self.on_slider_scroll
        )
        self.timeline_slider.set(0)
        self.timeline_slider.pack(fill="x", padx=5, pady=2)
        
        # Timeline Labels
        self.lbl_time_current = cctk.CTkLabel(self.timeline_frame, text="00:00.000", font=cctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        self.lbl_time_current.pack(side="left", padx=5)
        
        self.lbl_time_total = cctk.CTkLabel(self.timeline_frame, text="00:00.000", font=cctk.CTkFont(size=12), text_color=COLOR_TEXT_SECONDARY)
        self.lbl_time_total.pack(side="right", padx=5)
        
        # Playback Controls Frame
        self.playback_frame = cctk.CTkFrame(self.main_panel, fg_color=COLOR_BG)
        self.playback_frame.pack(fill="x", padx=10, pady=5)
        
        # Centered inner frame for playback buttons
        self.btn_row_1 = cctk.CTkFrame(self.playback_frame, fg_color=COLOR_BG)
        self.btn_row_1.pack(anchor="center", pady=5)
        
        self.btn_rewind = cctk.CTkButton(self.btn_row_1, text="⏪ -5s", width=80, command=lambda: self.skip_seconds(-5))
        self.btn_rewind.pack(side="left", padx=5)
        
        self.btn_play = cctk.CTkButton(self.btn_row_1, text="▶ Play", width=100, font=cctk.CTkFont(weight="bold"), fg_color=COLOR_ACCENT, hover_color="#6200EA", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=5)
        
        self.btn_stop = cctk.CTkButton(self.btn_row_1, text="⏹ Stop", width=80, command=self.stop_playback)
        self.btn_stop.pack(side="left", padx=5)
        
        self.btn_forward = cctk.CTkButton(self.btn_row_1, text="⏩ +5s", width=80, command=lambda: self.skip_seconds(5))
        self.btn_forward.pack(side="left", padx=5)
        
        # Centered inner frame for Marker Setting and actions
        self.btn_row_2 = cctk.CTkFrame(self.playback_frame, fg_color=COLOR_BG)
        self.btn_row_2.pack(anchor="center", pady=5)
        
        self.btn_set_start = cctk.CTkButton(
            self.btn_row_2, 
            text="[ Set Start", 
            width=100, 
            fg_color="#333333", 
            text_color=COLOR_TEXT_PRIMARY, 
            hover_color="#444444", 
            command=self.set_start_marker
        )
        self.btn_set_start.pack(side="left", padx=5)
        
        self.btn_set_end = cctk.CTkButton(
            self.btn_row_2, 
            text="] Set End", 
            width=100, 
            fg_color="#333333", 
            text_color=COLOR_TEXT_PRIMARY, 
            hover_color="#444444", 
            command=self.set_end_marker
        )
        self.btn_set_end.pack(side="left", padx=5)
        
        self.btn_jump_start = cctk.CTkButton(self.btn_row_2, text="Go to Start", width=90, fg_color="transparent", border_width=1, border_color=COLOR_BORDER, hover_color="#222222", command=self.jump_to_start)
        self.btn_jump_start.pack(side="left", padx=5)
        
        self.btn_jump_end = cctk.CTkButton(self.btn_row_2, text="Go to End", width=90, fg_color="transparent", border_width=1, border_color=COLOR_BORDER, hover_color="#222222", command=self.jump_to_end)
        self.btn_jump_end.pack(side="left", padx=5)
        
        self.btn_clear_markers = cctk.CTkButton(self.btn_row_2, text="Clear", width=70, fg_color="transparent", border_width=1, border_color=COLOR_BORDER, hover_color="#222222", command=self.clear_markers)
        self.btn_clear_markers.pack(side="left", padx=5)
        
        # Extra Settings Panel: Loop Checkbox
        self.extra_settings_frame = cctk.CTkFrame(self.main_panel, fg_color=COLOR_BG)
        self.extra_settings_frame.pack(fill="x", padx=10, pady=5)
        
        self.chk_loop = cctk.CTkCheckBox(self.extra_settings_frame, text="Loop selection during playback", font=cctk.CTkFont(size=12), border_width=2, fg_color=COLOR_ACCENT, hover_color="#6200EA")
        self.chk_loop.select() # Default to loop selection enabled
        self.chk_loop.pack(anchor="center")
        
    # ==========================================
    # LOGIC: Video Loading & Display
    # ==========================================
    def import_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.insv *.lrv *.m4v *.webm *.mpeg *.mpg *.3gp *.wmv"),
                ("Insta360 Camera Files", "*.insv *.lrv"),
                ("GoPro Video Files", "*.mp4 *.lrv"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.load_video(file_path)
        
    def load_video(self, file_path):
        # Stop existing playback
        self.stop_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the selected video file.")
            self.video_path = None
            return
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Avoid zero or invalid FPS
        if self.fps <= 0:
            self.fps = 30.0
            
        self.duration = self.total_frames / self.fps
        
        # Check for audio streams via ffprobe
        self.has_audio = self.check_has_audio(file_path)
        
        # Reset markers
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        
        # Reset filter controls & variables
        self.preset_combo.set("None (Manual Sliders)")
        self.on_preset_change("None (Manual Sliders)")
        self.reset_filters()
        self.crop_combo.set("Original (Landscape)")
        self.chk_title_active.deselect()
        self.last_raw_frame = None
        
        # Reset background audio variables
        self.bg_audio_path = None
        self.audio_combo.set("Original Audio")
        self.on_audio_mode_change("Original Audio")
        
        # Update sidebar info labels
        filename = os.path.basename(file_path)
        self.lbl_filename.configure(text=f"File: {filename}")
        
        audio_status = "Yes" if self.has_audio else "No"
        self.lbl_props.configure(text=f"{self.width}x{self.height}  |  {self.fps:.2f} fps  |  Audio: {audio_status}")
        
        # Reset timeline slider configuration
        self.timeline_slider.configure(from_=0, to=self.total_frames - 1)
        self.set_slider_value_without_callback(0)
        self.current_frame = 0
        
        # Update time and marker labels
        self.update_time_labels()
        self.update_marker_display()
        
        # Enable Export Button
        self.export_btn.configure(state="normal")
        
        # Read and display first frame
        self.seek_to_frame(0)
        
    def check_has_audio(self, file_path):
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "default=nw=1:nk=1",
                file_path
            ]
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True,
                startupinfo=startupinfo
            )
            return "audio" in result.stdout.lower()
        except Exception:
            return False
            
    def seek_to_frame(self, frame_no):
        if self.cap is None:
            return
            
        frame_no = max(0, min(frame_no, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_no
            self.last_raw_frame = frame.copy() # Cache raw frame for filters
            self.show_frame(frame)
            self.update_time_labels()
            
    def show_frame(self, frame):
        if frame is None:
            return
            
        # Convert OpenCV BGR to Pillow RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply Creative Presets or Manual Filters
        preset = self.preset_combo.get()
        
        if preset == "Cinematic (Warm & Rich)":
            frame_float = frame_rgb.astype(float)
            frame_float = frame_float * 1.3 - 12.75 # Boost contrast, lower brightness
            frame_float[:, :, 0] *= 1.05  # R boost (warmth)
            frame_float[:, :, 2] *= 0.95  # B reduction
            frame_rgb = np.clip(frame_float, 0, 255).astype(np.uint8)
            
        elif preset == "Vintage (Warm Sepia)":
            frame_float = frame_rgb.astype(float)
            frame_float = frame_float * 0.9 + 12.75 # Lower contrast, lift blacks
            r, g, b = frame_float[:,:,0], frame_float[:,:,1], frame_float[:,:,2]
            # Vectorized Sepia
            tr = 0.393 * r + 0.769 * g + 0.189 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.272 * r + 0.534 * g + 0.131 * b
            # 50% Blend
            frame_float[:,:,0] = 0.5 * tr + 0.5 * r
            frame_float[:,:,1] = 0.5 * tg + 0.5 * g
            frame_float[:,:,2] = 0.5 * tb + 0.5 * b
            frame_rgb = np.clip(frame_float, 0, 255).astype(np.uint8)
            
        elif preset == "Vivid Pop (Bright & Saturated)":
            frame_float = frame_rgb.astype(float)
            frame_float = frame_float * 1.1 + 5.1 # Boost contrast & brightness
            frame_clipped = np.clip(frame_float, 0, 255).astype(np.uint8)
            # Boost Saturation in HSV
            hsv = cv2.cvtColor(frame_clipped, cv2.COLOR_RGB2HSV).astype(float)
            hsv[:,:,1] *= 1.4
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            frame_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        elif preset == "Noir (Dramatic B&W)":
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            gray_float = gray.astype(float) * 1.4 - 25.5 # High Contrast Monochrome
            gray_uint = np.clip(gray_float, 0, 255).astype(np.uint8)
            frame_rgb = cv2.cvtColor(gray_uint, cv2.COLOR_GRAY2RGB)
            
        elif preset == "High Contrast B&W (Silver Halide)":
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            # Super deep blacks (1.6 contrast multiplier, -0.12 brightness offset)
            gray_float = gray.astype(float) * 1.6 - 30.6
            gray_uint = np.clip(gray_float, 0, 255).astype(np.uint8)
            frame_rgb = cv2.cvtColor(gray_uint, cv2.COLOR_GRAY2RGB)
            
        elif preset == "Old Time Film (Dust & Scratches)":
            frame_float = frame_rgb.astype(float)
            # 1. Subtle Projector brightness flicker (fluctuates randomly on each render)
            flicker = np.random.uniform(-0.04, 0.04)
            frame_float = frame_float * 0.9 + (0.05 + flicker) * 255.0
            
            # 2. Sepia matrix coloring (60% blend)
            r, g, b = frame_float[:,:,0], frame_float[:,:,1], frame_float[:,:,2]
            tr = 0.393 * r + 0.769 * g + 0.189 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.272 * r + 0.534 * g + 0.131 * b
            frame_float[:,:,0] = 0.6 * tr + 0.4 * r
            frame_float[:,:,1] = 0.6 * tg + 0.4 * g
            frame_float[:,:,2] = 0.6 * tb + 0.4 * b
            frame_rgb = np.clip(frame_float, 0, 255).astype(np.uint8)
            
        else: # Manual Sliders
            b_val = self.brightness_slider.get()
            c_val = self.contrast_slider.get()
            if c_val != 1.0 or b_val != 0.0:
                frame_float = frame_rgb.astype(float)
                frame_float = frame_float * c_val + (b_val * 255.0)
                frame_rgb = np.clip(frame_float, 0, 255).astype(np.uint8)
                
            if self.chk_grayscale.get():
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Calculate scaling to fit 640x360 maintaining aspect ratio
        h, w, _ = frame_rgb.shape
        target_w, target_h = 640, 360
        
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Apply Social Media Crop Overlays (Semiparent letterbox margins)
        crop_mode = self.crop_combo.get()
        if "9:16" in crop_mode:
            crop_w = int(new_h * (9.0 / 16.0))
            x_start = max(0, (new_w - crop_w) // 2)
            x_end = min(new_w, x_start + crop_w)
            
            overlay = frame_resized.copy()
            overlay[:, :x_start] = (overlay[:, :x_start] * 0.4).astype(np.uint8)
            overlay[:, x_end:] = (overlay[:, x_end:] * 0.4).astype(np.uint8)
            cv2.line(overlay, (x_start, 0), (x_start, new_h), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(overlay, (x_end, 0), (x_end, new_h), (255, 255, 255), 1, cv2.LINE_AA)
            frame_resized = overlay
            
        elif "1:1" in crop_mode:
            crop_w = new_h
            x_start = max(0, (new_w - crop_w) // 2)
            x_end = min(new_w, x_start + crop_w)
            
            overlay = frame_resized.copy()
            overlay[:, :x_start] = (overlay[:, :x_start] * 0.4).astype(np.uint8)
            overlay[:, x_end:] = (overlay[:, x_end:] * 0.4).astype(np.uint8)
            cv2.line(overlay, (x_start, 0), (x_start, new_h), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(overlay, (x_end, 0), (x_end, new_h), (255, 255, 255), 1, cv2.LINE_AA)
            frame_resized = overlay
            
        # Draw dynamic vintage noise (Dust & Scratches) on the final resized preview frame
        if preset == "Old Time Film (Dust & Scratches)":
            # Generate 2 to 5 random dust spots
            num_dust = np.random.randint(2, 6)
            for _ in range(num_dust):
                x_d = np.random.randint(0, new_w)
                y_d = np.random.randint(0, new_h)
                r_size = np.random.randint(1, 3)
                cv2.circle(frame_resized, (x_d, y_d), r_size, (15, 15, 15), -1)
                
            # Generate 1 thin, vertical scratch line (35% probability per frame)
            if np.random.rand() < 0.35:
                x_s = np.random.randint(0, new_w)
                x_end = x_s + np.random.randint(-2, 2)
                cv2.line(frame_resized, (x_s, 0), (x_end, new_h), (35, 35, 35), 1)
            
        # Apply Title Intro Overlay
        if self.chk_title_active.get():
            t_sec = self.current_frame / self.fps
            if t_sec < 3.0:
                text = self.txt_title_text.get().strip()
                if text:
                    # Fade out between 2.0s and 3.0s
                    alpha = 1.0
                    if t_sec > 2.0:
                        alpha = max(0.0, 1.0 - (t_sec - 2.0))
                        
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = min(1.2, max(0.6, new_w / 500.0))
                    thickness = 2
                    
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = (new_w - text_size[0]) // 2
                    text_y = (new_h + text_size[1]) // 2
                    
                    text_x = max(10, text_x)
                    text_y = max(text_size[1] + 10, text_y)
                    
                    text_layer = frame_resized.copy()
                    cv2.putText(text_layer, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                    cv2.putText(text_layer, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    
                    frame_resized = cv2.addWeighted(text_layer, alpha, frame_resized, 1.0 - alpha, 0)
        
        # Convert to PIL and display (using CTkImage for DPI-scale warning prevention)
        img = Image.fromarray(frame_resized)
        ctk_img = cctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
        
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img  # Keep reference
        
    # ==========================================
    # LOGIC: Playback Controls
    # ==========================================
    def toggle_play(self):
        if self.cap is None:
            return
            
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()
            
    def start_playback(self):
        if self.playing:
            return
            
        self.playing = True
        self.btn_play.configure(text="⏸ Pause")
        self.play_loop()
        
    def pause_playback(self):
        self.playing = False
        self.btn_play.configure(text="▶ Play")
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
            
    def stop_playback(self):
        self.pause_playback()
        if self.cap is not None:
            start_f = self.start_frame if self.start_frame is not None else 0
            self.seek_to_frame(start_f)
            self.set_slider_value_without_callback(start_f)
            
    def play_loop(self):
        if not self.playing or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.last_raw_frame = frame.copy() # Cache raw frame
            self.show_frame(frame)
            self.set_slider_value_without_callback(self.current_frame)
            self.update_time_labels()
            
            # Check loop selection boundary
            loop_enabled = self.chk_loop.get()
            
            if self.end_frame is not None and self.current_frame >= self.end_frame:
                if loop_enabled:
                    start_f = self.start_frame if self.start_frame is not None else 0
                    self.seek_to_frame(start_f)
                else:
                    self.pause_playback()
                    return
            
            if self.current_frame >= self.total_frames - 1:
                if loop_enabled:
                    start_f = self.start_frame if self.start_frame is not None else 0
                    self.seek_to_frame(start_f)
                else:
                    self.pause_playback()
                    return
            
            delay = int(1000 / self.fps)
            delay = max(1, delay)
            
            self.after_id = self.after(delay, self.play_loop)
        else:
            if self.chk_loop.get():
                start_f = self.start_frame if self.start_frame is not None else 0
                self.seek_to_frame(start_f)
                delay = int(1000 / self.fps)
                self.after_id = self.after(max(1, delay), self.play_loop)
            else:
                self.pause_playback()
                
    def skip_seconds(self, seconds):
        if self.cap is None:
            return
            
        frame_offset = int(seconds * self.fps)
        new_frame = self.current_frame + frame_offset
        new_frame = max(0, min(new_frame, self.total_frames - 1))
        
        self.seek_to_frame(new_frame)
        self.set_slider_value_without_callback(new_frame)
        
    def on_slider_scroll(self, value):
        if self.cap is None:
            return
            
        if self.playing:
            self.pause_playback()
            
        frame_no = int(value)
        self.seek_to_frame(frame_no)
        
    def set_slider_value_without_callback(self, value):
        self.timeline_slider.configure(command=None)
        self.timeline_slider.set(value)
        self.timeline_slider.configure(command=self.on_slider_scroll)
        
    # ==========================================
    # LOGIC: Start/End Markers
    # ==========================================
    def set_start_marker(self):
        if self.cap is None:
            return
        
        if self.end_frame is not None and self.current_frame >= self.end_frame:
            messagebox.showwarning("Warning", "Start marker must be set before the End marker.")
            return
            
        self.start_frame = self.current_frame
        self.update_marker_display()
        
    def set_end_marker(self):
        if self.cap is None:
            return
        
        if self.start_frame is not None and self.current_frame <= self.start_frame:
            messagebox.showwarning("Warning", "End marker must be set after the Start marker.")
            return
            
        self.end_frame = self.current_frame
        self.update_marker_display()
        
    def jump_to_start(self):
        if self.cap is None or self.start_frame is None:
            return
        self.seek_to_frame(self.start_frame)
        self.set_slider_value_without_callback(self.start_frame)
        
    def jump_to_end(self):
        if self.cap is None or self.end_frame is None:
            return
        self.seek_to_frame(self.end_frame)
        self.set_slider_value_without_callback(self.end_frame)
        
    def clear_markers(self):
        if self.cap is None:
            return
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        self.update_marker_display()
        
    def update_marker_display(self):
        if self.cap is None:
            self.lbl_selected_len.configure(text="Cut: 0.0s (Full)")
            return
            
        start_sec = self.start_frame / self.fps
        end_sec = self.end_frame / self.fps
        diff_sec = end_sec - start_sec
        
        self.lbl_selected_len.configure(
            text=f"Cut: {self.format_seconds(start_sec)} to {self.format_seconds(end_sec)} ({diff_sec:.1f}s)"
        )
        
    def update_time_labels(self):
        if self.cap is None:
            self.lbl_time_current.configure(text="00:00.000")
            self.lbl_time_total.configure(text="00:00.000")
            return
            
        curr_sec = self.current_frame / self.fps
        self.lbl_time_current.configure(text=self.format_seconds(curr_sec))
        self.lbl_time_total.configure(text=self.format_seconds(self.duration))
        
    def format_seconds(self, total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        ms = int((total_seconds - int(total_seconds)) * 1000)
        
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        else:
            return f"{m:02d}:{s:02d}.{ms:03d}"
            
    # ==========================================
    # LOGIC: Tabs Interactive Callbacks
    # ==========================================
    def on_audio_mode_change(self, choice):
        if choice == "Replace Audio":
            self.btn_select_audio.configure(state="normal")
            self.chk_loop_audio.configure(state="normal")
            if not self.bg_audio_path:
                self.lbl_audio_filename.configure(text="No audio file selected", text_color=COLOR_CANCEL)
            else:
                self.lbl_audio_filename.configure(text=f"Selected: {os.path.basename(self.bg_audio_path)}", text_color=COLOR_SUCCESS)
        else:
            self.btn_select_audio.configure(state="disabled")
            self.chk_loop_audio.configure(state="disabled")
            self.lbl_audio_filename.configure(text="Not applicable", text_color=COLOR_TEXT_SECONDARY)
            
    def select_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Background Audio File",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.aac *.ogg *.flac *.wma")]
        )
        if file_path:
            self.bg_audio_path = file_path
            filename = os.path.basename(file_path)
            self.lbl_audio_filename.configure(text=f"Selected: {filename}", text_color=COLOR_SUCCESS)
            
    def on_fade_duration_change(self, value):
        self.lbl_fade_duration.configure(text=f"Fade Duration: {float(value):.1f}s")
        
    def on_preset_change(self, choice):
        if choice == "None (Manual Sliders)":
            self.brightness_slider.configure(state="normal")
            self.contrast_slider.configure(state="normal")
            self.chk_grayscale.configure(state="normal")
        else:
            self.brightness_slider.configure(state="disabled")
            self.contrast_slider.configure(state="disabled")
            self.chk_grayscale.configure(state="disabled")
        self.on_filter_change()
        
    def on_filter_change(self, *args):
        b_val = self.brightness_slider.get()
        c_val = self.contrast_slider.get()
        self.lbl_brightness.configure(text=f"Brightness: {b_val:+.2f}")
        self.lbl_contrast.configure(text=f"Contrast: {c_val:.1f}x")
        
        if self.last_raw_frame is not None and not self.playing:
            self.show_frame(self.last_raw_frame)
            
    def reset_filters(self):
        self.brightness_slider.set(0.0)
        self.contrast_slider.set(1.0)
        self.chk_grayscale.deselect()
        self.on_filter_change()
        
    # ==========================================
    # LOGIC: Video Exporting (FFmpeg Integration)
    # ==========================================
    def parse_speed_value(self):
        speed_str = self.speed_combo.get()
        match = re.match(r"([\d\.]+)x", speed_str)
        if match:
            return float(match.group(1))
        return 1.0
        
    def start_export(self):
        if self.cap is None or self.video_path is None:
            return
            
        if self.start_frame is None or self.end_frame is None:
            messagebox.showwarning("Warning", "Please set valid Start and End markers first.")
            return
            
        if self.start_frame >= self.end_frame:
            messagebox.showwarning("Warning", "Start marker must be before the End marker.")
            return
            
        audio_mode = self.audio_combo.get()
        if audio_mode == "Replace Audio":
            if not self.bg_audio_path or not os.path.exists(self.bg_audio_path):
                messagebox.showwarning("Warning", "Please select a valid background audio file first.")
                self.tabs.set("Audio")
                return
            
        # Parse extension and recommend standard MP4 for raw action formats (.insv, .lrv)
        ext = os.path.splitext(self.video_path)[1].lower()
        if ext in [".insv", ".lrv"]:
            default_ext = ".mp4"
        else:
            default_ext = ext if ext else ".mp4"
            
        save_path = filedialog.asksaveasfilename(
            title="Save Exported Video",
            defaultextension=default_ext,
            filetypes=[("Video Files", f"*{default_ext}"), ("MP4 Video", "*.mp4"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
            
        # Pause playing during export
        self.pause_playback()
        
        # UI State: Disable interactive controls, show progress bar
        self.toggle_ui_state(exporting=True)
        
        start_sec = self.start_frame / self.fps
        end_sec = self.end_frame / self.fps
        speed = self.parse_speed_value()
        
        # Start export thread
        self.export_thread = threading.Thread(
            target=self.run_ffmpeg_export,
            args=(self.video_path, save_path, start_sec, end_sec, speed),
            daemon=True
        )
        self.export_thread.start()
        
    def cancel_export(self):
        if self.export_process and self.export_process.poll() is None:
            if messagebox.askyesno("Cancel Export", "Are you sure you want to cancel the export process?"):
                try:
                    self.export_process.terminate()
                    self.export_process.wait()
                except Exception:
                    pass
                self.on_export_complete(False, "Export cancelled by user.")
                
    def toggle_ui_state(self, exporting):
        self.exporting = exporting
        btn_state = "disabled" if exporting else "normal"
        combo_state = "disabled" if exporting else "readonly"
        
        self.import_btn.configure(state=btn_state)
        self.export_btn.configure(state=btn_state)
        self.speed_combo.configure(state=combo_state)
        self.codec_combo.configure(state=combo_state)
        self.crop_combo.configure(state=combo_state)
        
        self.audio_combo.configure(state=combo_state)
        if not exporting and self.audio_combo.get() == "Replace Audio":
            self.btn_select_audio.configure(state="normal")
            self.chk_loop_audio.configure(state="normal")
        else:
            self.btn_select_audio.configure(state="disabled")
            self.chk_loop_audio.configure(state="disabled")
            
        self.fade_duration_slider.configure(state=btn_state)
        self.chk_fade_in.configure(state=btn_state)
        self.chk_fade_out.configure(state=btn_state)
        
        self.preset_combo.configure(state=combo_state)
        if not exporting and self.preset_combo.get() == "None (Manual Sliders)":
            self.brightness_slider.configure(state="normal")
            self.contrast_slider.configure(state="normal")
            self.chk_grayscale.configure(state="normal")
        else:
            self.brightness_slider.configure(state="disabled")
            self.contrast_slider.configure(state="disabled")
            self.chk_grayscale.configure(state="disabled")
            
        self.btn_reset_filters.configure(state=btn_state)
        self.chk_title_active.configure(state=btn_state)
        self.txt_title_text.configure(state=btn_state)
        
        self.btn_rewind.configure(state=btn_state)
        self.btn_play.configure(state=btn_state)
        self.btn_stop.configure(state=btn_state)
        self.btn_forward.configure(state=btn_state)
        self.btn_set_start.configure(state=btn_state)
        self.btn_set_end.configure(state=btn_state)
        self.btn_jump_start.configure(state=btn_state)
        self.btn_jump_end.configure(state=btn_state)
        self.btn_clear_markers.configure(state=btn_state)
        self.timeline_slider.configure(state=btn_state)
        self.chk_loop.configure(state=btn_state)
        
        if exporting:
            self.progress_frame.pack(fill="x", padx=10, pady=10)
            self.lbl_progress.configure(text="Starting Export...")
            self.progress_bar.set(0.0)
        else:
            self.progress_frame.pack_forget()
            
    def set_export_progress(self, progress_val):
        self.after(0, lambda: self.progress_bar.set(progress_val))
        self.after(0, lambda: self.lbl_progress.configure(text=f"Exporting: {int(progress_val * 100)}%"))
        
    def sanitize_text(self, text):
        return text.replace("\\", "").replace("'", "").replace("\"", "").replace(":", "")

    def run_ffmpeg_export(self, input_path, output_path, start_time, end_time, speed):
        audio_mode = self.audio_combo.get()
        fade_in = self.chk_fade_in.get()
        fade_out = self.chk_fade_out.get()
        fade_duration = self.fade_duration_slider.get()
        
        brightness = self.brightness_slider.get()
        contrast = self.contrast_slider.get()
        grayscale = self.chk_grayscale.get()
        
        preset = self.preset_combo.get()
        crop_mode = self.crop_combo.get()
        
        title_active = self.chk_title_active.get()
        title_text = self.sanitize_text(self.txt_title_text.get().strip())
        
        # Read chosen video codec
        codec_selection = self.codec_combo.get()
        video_codec = "libx264"
        if "H.265" in codec_selection:
            video_codec = "libx265"
        
        use_audio = False
        if audio_mode == "Original Audio" and self.has_audio:
            use_audio = True
        elif audio_mode == "Replace Audio" and self.bg_audio_path:
            use_audio = True

        duration = end_time - start_time
        out_duration = duration / speed

        # 1. Base command & output overwrite
        cmd = ["ffmpeg", "-y"]
        
        # 2. Seek start time before the input (faster seeking)
        cmd += ["-ss", f"{start_time:.3f}"]
        
        # 3. Add duration limit
        cmd += ["-t", f"{duration:.3f}"]
        
        # 4. Input file
        cmd += ["-i", input_path]
        
        # 5. Audio replacement file
        if audio_mode == "Replace Audio" and self.bg_audio_path:
            if self.chk_loop_audio.get():
                cmd += ["-stream_loop", "-1"]
            cmd += ["-ss", "0", "-t", f"{duration:.3f}", "-i", self.bg_audio_path]
            
        # ==========================================
        # BUILD FILTER GRAPH CHAIN
        # ==========================================
        video_filters = []
        
        # 1. Aspect Ratio Cropping (first in chain)
        if "9:16" in crop_mode:
            video_filters.append("crop=ih*9/16:ih")
        elif "1:1" in crop_mode:
            video_filters.append("crop=ih:ih")
        
        # 2. Video Speed (PTS scale)
        if speed != 1.0:
            video_filters.append(f"setpts={1.0/speed:.4f}*PTS")
            
        # 3. Creative Presets or Manual Visual Filters
        if preset == "Cinematic (Warm & Rich)":
            video_filters.append("eq=contrast=1.3:brightness=-0.05,colorchannelmixer=rr=1.05:gg=1.0:bb=0.95")
        elif preset == "Vintage (Warm Sepia)":
            video_filters.append("eq=contrast=0.9:brightness=0.05,colorchannelmixer=0.696:0.385:0.095:0:0.175:0.843:0.084:0:0.136:0.267:0.566")
        elif preset == "Vivid Pop (Bright & Saturated)":
            video_filters.append("hue=s=1.4,eq=contrast=1.1:brightness=0.02")
        elif preset == "Noir (Dramatic B&W)":
            video_filters.append("hue=s=0,eq=contrast=1.4:brightness=-0.1")
        elif preset == "High Contrast B&W (Silver Halide)":
            video_filters.append("hue=s=0,eq=contrast=1.6:brightness=-0.12")
        elif preset == "Old Time Film (Dust & Scratches)":
            # Grain, projector flicker at 8Hz, sepia filter, and modulo-time jumping vertical scratches
            video_filters.append("noise=alls=12:allf=t+u")
            video_filters.append("eq=brightness='0.03+0.02*sin(2*PI*t*8)':contrast=0.9")
            video_filters.append("colorchannelmixer=0.696:0.385:0.095:0:0.175:0.843:0.084:0:0.136:0.267:0.566")
            
            # Escape commas in drawbox equations inside FFmpeg filter graphs
            scratch_filter = r"drawbox=x='if(lt(mod(t\,0.25)\,0.04)\,abs(sin(t*99))*w\,-20)':w=1:h=ih:color=black@0.3"
            video_filters.append(scratch_filter)
        else: # Manual
            if contrast != 1.0 or brightness != 0.0:
                video_filters.append(f"eq=contrast={contrast:.2f}:brightness={brightness:.2f}")
            if grayscale:
                video_filters.append("hue=s=0")
                
        # 4. Title Card Overlay (renders centered on final frame size for first 3 seconds)
        if title_active and title_text:
            alpha_math = "if(lt(t,2),1,if(lt(t,3),1-(t-2),0))"
            drawtext_str = (
                f"drawtext=text='{title_text}':fontcolor=white:fontsize=48:"
                f"bordercolor=black:borderw=2:x=(w-tw)/2:y=(h-th)/2:alpha='{alpha_math}'"
            )
            video_filters.append(drawtext_str)
            
        # 5. Video Transitions (Fades)
        if fade_in:
            actual_fade = min(fade_duration, out_duration / 2.0)
            video_filters.append(f"fade=t=in:st=0:d={actual_fade:.2f}")
        if fade_out:
            actual_fade = min(fade_duration, out_duration / 2.0)
            fade_start = max(0.0, out_duration - actual_fade)
            video_filters.append(f"fade=t=out:st={fade_start:.2f}:d={actual_fade:.2f}")

        # Audio Filters
        audio_filters = []
        if use_audio:
            # Audio Speed (atempo)
            if speed != 1.0:
                factors = []
                temp_speed = speed
                if temp_speed > 2.0:
                    while temp_speed > 2.0:
                        factors.append("2.0")
                        temp_speed /= 2.0
                    factors.append(f"{temp_speed:.4f}")
                elif temp_speed < 0.5:
                    while temp_speed < 0.5:
                        factors.append("0.5")
                        temp_speed /= 0.5
                    factors.append(f"{temp_speed:.4f}")
                else:
                    factors.append(f"{temp_speed:.4f}")
                
                atempo_str = ",".join([f"atempo={f}" for f in factors])
                audio_filters.append(atempo_str)
                
            # Audio Fades
            if fade_in:
                actual_fade = min(fade_duration, out_duration / 2.0)
                audio_filters.append(f"afade=t=in:st=0:d={actual_fade:.2f}")
            if fade_out:
                actual_fade = min(fade_duration, out_duration / 2.0)
                fade_start = max(0.0, out_duration - actual_fade)
                audio_filters.append(f"afade=t=out:st={fade_start:.2f}:d={actual_fade:.2f}")

        # Stitch filters using filter_complex
        if len(video_filters) > 0 or (use_audio and len(audio_filters) > 0):
            filter_parts = []
            
            # Video stream graph
            vf_chain = ",".join(video_filters)
            if vf_chain:
                filter_parts.append(f"[0:v]{vf_chain}[v]")
                v_map = "[v]"
            else:
                filter_parts.append("[0:v]null[v]")
                v_map = "[v]"
                
            # Audio stream graph
            if use_audio:
                a_in = "[0:a]" if audio_mode == "Original Audio" else "[1:a]"
                af_chain = ",".join(audio_filters)
                if af_chain:
                    filter_parts.append(f"{a_in}{af_chain}[a]")
                    a_map = "[a]"
                else:
                    filter_parts.append(f"{a_in}anull[a]")
                    a_map = "[a]"
            
            cmd += ["-filter_complex", ";".join(filter_parts)]
            cmd += ["-map", v_map]
            if use_audio:
                cmd += ["-map", a_map]
        else:
            # Simple maps without complex filters
            cmd += ["-map", "0:v"]
            if use_audio:
                a_in = "0:a" if audio_mode == "Original Audio" else "1:a"
                cmd += ["-map", a_in]
                
        # Set codecs (using dynamically chosen video_codec, H.264 or H.265/HEVC)
        cmd += ["-c:v", video_codec]
        if use_audio:
            cmd += ["-c:a", "aac"]
        else:
            cmd += ["-an"]
            
        cmd += [output_path]
        
        print("Executing FFmpeg command:", " ".join(cmd))
        
        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            self.export_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                startupinfo=startupinfo
            )
            
            expected_out_duration = out_duration
            if expected_out_duration <= 0:
                expected_out_duration = 1.0
                
            # Parse progress logs from FFmpeg stderr
            time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d+)")
            
            while True:
                line = self.export_process.stderr.readline()
                if not line:
                    break
                
                # Check for progress time marker
                match = time_pattern.search(line)
                if match:
                    h, m, s, ms_str = match.groups()
                    h, m, s = int(h), int(m), int(s)
                    ms = float(f"0.{ms_str}")
                    curr_out_seconds = h * 3600 + m * 60 + s + ms
                    
                    progress_percent = min(0.99, curr_out_seconds / expected_out_duration)
                    self.set_export_progress(progress_percent)
            
            self.export_process.wait()
            ret_code = self.export_process.returncode
            
            if ret_code == 0:
                self.set_export_progress(1.0)
                time.sleep(0.5) # Let user see 100% complete
                self.on_export_complete(True, "Video exported successfully!")
            else:
                if self.exporting: # If not intentionally cancelled by user
                    self.on_export_complete(False, f"FFmpeg error. Return code: {ret_code}")
                    
        except Exception as e:
            self.on_export_complete(False, f"Export failed: {str(e)}")
            
    def on_export_complete(self, success, message):
        # Return UI controls to active state
        self.after(0, lambda: self.toggle_ui_state(exporting=False))
        
        # Display completion box
        if success:
            self.after(0, lambda: messagebox.showinfo("Export Successful", message))
        else:
            self.after(0, lambda: messagebox.showerror("Export Failed", message))
            
    def on_closing(self):
        # Stop playback loop
        self.playing = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            
        # Kill export process if it's active
        if self.export_process and self.export_process.poll() is None:
            try:
                self.export_process.terminate()
                self.export_process.wait()
            except Exception:
                pass
                
        # Release capture
        if self.cap is not None:
            self.cap.release()
            
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = VideoEditorApp()
    # PhotoLab / CLI: python video_editor.py path/to/clip.mp4
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.startswith("--"):
            # optional: --video PATH
            if arg in ("--video", "-i") and len(sys.argv) > 2:
                arg = sys.argv[2]
            else:
                arg = None
        if arg and os.path.isfile(arg):
            app.after(250, lambda p=arg: app.load_video(p))
    app.mainloop()
