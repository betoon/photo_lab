"""Measured display and camera profiling workflows backed by ArgyllCMS."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PyQt6.QtCore import QProcess, QSettings, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)


ARGYLL_TOOLS = ("dispcal", "dispwin", "scanin", "colprof")


def _exe_name(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def find_argyll_dir(preferred: str = "") -> str:
    """Return a directory containing the core Argyll tools, or an empty string."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        from external_paths import resolve_path
        configured = resolve_path("argyllcms_dir")
        if configured:
            candidates.append(configured)
    except Exception:
        pass
    env = os.environ.get("ARGYLLCMS_BIN", "")
    if env:
        candidates.append(env)
    for name in ("dispcal", "dispcal.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(os.path.dirname(found))
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\Argyll_V3.5.0\bin",
            r"C:\Program Files\ArgyllCMS\bin",
            os.path.join(os.path.expanduser("~"), "Argyll", "bin"),
        ])
    for folder in candidates:
        folder = os.path.abspath(os.path.expanduser(folder))
        if os.path.isfile(os.path.join(folder, _exe_name("dispcal"))):
            return folder
    return ""


def argyll_tool(folder: str, name: str) -> str:
    path = os.path.join(folder or "", _exe_name(name))
    return path if os.path.isfile(path) else ""


def build_dispcal_args(output_base: str, display: int = 1, whitepoint: str = "D65",
                       luminance: int = 120, gamma: str = "2.2",
                       quality: str = "Medium") -> list[str]:
    wp = {"D65": "6500", "D50": "5000", "Native": ""}.get(whitepoint, "6500")
    q = {"Low": "l", "Medium": "m", "High": "h"}.get(quality, "m")
    args = ["-v", "-d", str(max(1, int(display))), "-q", q]
    if wp:
        args += ["-t", wp]
    if luminance > 0:
        args += ["-b", str(int(luminance))]
    if gamma and gamma != "Native":
        args += ["-g", "s" if gamma == "sRGB" else str(gamma)]
    args += ["-o", output_base + ".icc", "-O", os.path.basename(output_base), output_base]
    return args


def build_scanin_args(image: str, chart: str, reference: str, output_base: str) -> list[str]:
    return ["-v", image, chart, reference, output_base]


def build_colprof_args(output_base: str, description: str) -> list[str]:
    return ["-v", "-D", description, "-qm", "-as", "-O", output_base + ".icc", output_base]


def validate_icc(path: str) -> tuple[bool, str]:
    if not path or not os.path.isfile(path):
        return False, "Profile file was not found."
    try:
        from PIL import ImageCms
        profile = ImageCms.getOpenProfile(path)
        desc = ImageCms.getProfileDescription(profile).strip() or os.path.basename(path)
        return True, desc
    except Exception as exc:
        return False, f"Not a readable ICC profile: {exc}"


def _field_row(edit: QLineEdit, button_text: str, callback):
    box = QWidget(); row = QHBoxLayout(box); row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1); button = QPushButton(button_text); button.clicked.connect(callback); row.addWidget(button)
    return box


class ColorCalibrationDialog(QDialog):
    """Photographer-facing wrapper around Argyll display and camera profiling."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Calibration Studio")
        self.resize(760, 650)
        self.settings = QSettings("PhotoLab", "PhotoLab")
        self.process = None
        self._camera_colprof = None

        root = QVBoxLayout(self)
        title = QLabel("Color Calibration Studio")
        title.setStyleSheet("font-size:20px; font-weight:700; color:#eee;")
        root.addWidget(title)
        intro = QLabel("Measured display and camera profiling powered by ArgyllCMS. "
                       "Accurate results require a supported instrument or a photographed reference chart.")
        intro.setWordWrap(True); intro.setStyleSheet("color:#aaa;"); root.addWidget(intro)

        engine = QGroupBox("ArgyllCMS engine")
        form = QFormLayout(engine)
        self.argyll_edit = QLineEdit()
        self.argyll_edit.setText(find_argyll_dir())
        form.addRow("bin folder:", _field_row(self.argyll_edit, "Browse…", self._browse_argyll))
        self.engine_status = QLabel(); form.addRow("Status:", self.engine_status)
        root.addWidget(engine)

        self.tabs = QTabWidget(); self.tabs.addTab(self._monitor_tab(), "Display")
        self.tabs.addTab(self._camera_tab(), "Camera chart")
        self.tabs.addTab(self._profile_tab(), "ICC profiles")
        root.addWidget(self.tabs, 1)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Calibration progress and verification details appear here.")
        root.addWidget(self.log, 1)
        close = QPushButton("Close"); close.clicked.connect(self.reject)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(close); root.addLayout(row)
        self.argyll_edit.textChanged.connect(self._refresh_engine)
        self._refresh_engine()

    def _monitor_tab(self):
        tab=QWidget(); form=QFormLayout(tab)
        self.display_spin=QSpinBox(); self.display_spin.setRange(1,16); form.addRow("Display number:",self.display_spin)
        self.white_combo=QComboBox(); self.white_combo.addItems(["D65","D50","Native"]); form.addRow("White point:",self.white_combo)
        self.lum_spin=QSpinBox(); self.lum_spin.setRange(60,300); self.lum_spin.setValue(120); self.lum_spin.setSuffix(" cd/m²"); form.addRow("Brightness:",self.lum_spin)
        self.gamma_combo=QComboBox(); self.gamma_combo.addItems(["2.2","2.4","sRGB","Native"]); form.addRow("Response:",self.gamma_combo)
        self.quality_combo=QComboBox(); self.quality_combo.addItems(["Medium","High","Low"]); form.addRow("Quality:",self.quality_combo)
        self.monitor_output=QLineEdit(os.path.join(os.path.expanduser("~"), "Documents", "PhotoLab Profiles", "PhotoLab_Display_D65"))
        form.addRow("Output base:",_field_row(self.monitor_output,"Choose…",self._browse_monitor_output))
        note=QLabel("Warm the display for about 30 minutes, disable Night Light/HDR and competing profile loaders, "
                    "connect the colorimeter, then follow Argyll's measured prompts in the calibration window.")
        note.setWordWrap(True); note.setStyleSheet("color:#bbb;"); form.addRow(note)
        self.start_monitor=QPushButton("Start guided display calibration…"); self.start_monitor.clicked.connect(self._start_monitor)
        form.addRow(self.start_monitor)
        return tab

    def _camera_tab(self):
        tab=QWidget(); form=QFormLayout(tab)
        self.chart_image=QLineEdit(); form.addRow("Chart photograph:",_field_row(self.chart_image,"Browse…",lambda:self._pick_file(self.chart_image,"Chart image","Images (*.tif *.tiff *.jpg *.jpeg *.png)")))
        self.chart_layout=QLineEdit(); form.addRow("Recognition file:",_field_row(self.chart_layout,"Browse…",lambda:self._pick_file(self.chart_layout,"Argyll chart layout","Chart layout (*.cht)")))
        self.chart_reference=QLineEdit(); form.addRow("Reference values:",_field_row(self.chart_reference,"Browse…",lambda:self._pick_file(self.chart_reference,"Chart reference","Reference (*.cie *.ti2 *.txt)")))
        self.camera_desc=QLineEdit("PhotoLab Camera Profile"); form.addRow("Profile name:",self.camera_desc)
        self.camera_output=QLineEdit(os.path.join(os.path.expanduser("~"), "Documents", "PhotoLab Profiles", "PhotoLab_Camera"))
        form.addRow("Output base:",_field_row(self.camera_output,"Choose…",self._browse_camera_output))
        note=QLabel("Use an evenly lit, glare-free chart photograph with automatic corrections disabled. "
                    "A TIFF made from a neutral RAW conversion is preferred. Argyll scanin locates the patches; "
                    "colprof creates a matrix/shaper input ICC profile and reports fit errors.")
        note.setWordWrap(True); note.setStyleSheet("color:#bbb;"); form.addRow(note)
        self.start_camera=QPushButton("Build camera ICC profile"); self.start_camera.clicked.connect(self._start_camera)
        form.addRow(self.start_camera)
        return tab

    def _profile_tab(self):
        tab=QWidget(); form=QFormLayout(tab)
        self.icc_edit=QLineEdit(); form.addRow("ICC profile:",_field_row(self.icc_edit,"Browse…",self._browse_icc))
        self.icc_status=QLabel("Choose an ICC/ICM profile to inspect."); self.icc_status.setWordWrap(True); form.addRow("Validation:",self.icc_status)
        row=QWidget(); lay=QHBoxLayout(row); lay.setContentsMargins(0,0,0,0)
        self.validate_btn=QPushButton("Validate"); self.validate_btn.clicked.connect(self._validate_profile); lay.addWidget(self.validate_btn)
        self.install_btn=QPushButton("Install for display…"); self.install_btn.clicked.connect(self._install_profile); lay.addWidget(self.install_btn); lay.addStretch(1)
        form.addRow(row)
        note=QLabel("Installing changes the operating system's default profile for the selected display and loads its calibration curves. "
                    "PhotoLab asks for confirmation before making that change.")
        note.setWordWrap(True); note.setStyleSheet("color:#bbb;"); form.addRow(note)
        return tab

    def _browse_argyll(self):
        p=QFileDialog.getExistingDirectory(self,"Choose ArgyllCMS bin folder",self.argyll_edit.text() or os.path.expanduser("~"))
        if p: self.argyll_edit.setText(p)
    def _pick_file(self, edit, title, filt):
        p,_=QFileDialog.getOpenFileName(self,title,edit.text() or os.path.expanduser("~"),filt)
        if p: edit.setText(p)
    def _browse_icc(self): self._pick_file(self.icc_edit,"Choose ICC profile","ICC profiles (*.icc *.icm)")
    def _browse_monitor_output(self):
        p,_=QFileDialog.getSaveFileName(self,"Display profile base",self.monitor_output.text(),"ICC profile (*.icc)")
        if p: self.monitor_output.setText(os.path.splitext(p)[0])
    def _browse_camera_output(self):
        p,_=QFileDialog.getSaveFileName(self,"Camera profile base",self.camera_output.text(),"ICC profile (*.icc)")
        if p: self.camera_output.setText(os.path.splitext(p)[0])

    def _refresh_engine(self):
        folder=find_argyll_dir(self.argyll_edit.text().strip())
        if folder and folder != self.argyll_edit.text(): self.argyll_edit.blockSignals(True); self.argyll_edit.setText(folder); self.argyll_edit.blockSignals(False)
        missing=[n for n in ARGYLL_TOOLS if not argyll_tool(folder,n)] if folder else list(ARGYLL_TOOLS)
        ok=not missing
        self.engine_status.setText("Ready - all required tools found" if ok else "Not ready - missing: "+", ".join(missing))
        self.engine_status.setStyleSheet("color:#43c96b;" if ok else "color:#e05252;")
        self.start_monitor.setEnabled(bool(argyll_tool(folder,"dispcal")))
        self.start_camera.setEnabled(bool(argyll_tool(folder,"scanin") and argyll_tool(folder,"colprof")))
        self.install_btn.setEnabled(bool(argyll_tool(folder,"dispwin")))

    def _start_monitor(self):
        exe=argyll_tool(self.argyll_edit.text(),"dispcal"); base=os.path.splitext(self.monitor_output.text().strip())[0]
        if not exe or not base: return
        Path(base).parent.mkdir(parents=True,exist_ok=True)
        args=build_dispcal_args(base,self.display_spin.value(),self.white_combo.currentText(),self.lum_spin.value(),self.gamma_combo.currentText(),self.quality_combo.currentText())
        self.log.appendPlainText("Starting measured display calibration:\n"+subprocess.list2cmdline([exe]+args))
        try:
            if os.name == "nt": subprocess.Popen([exe]+args,creationflags=subprocess.CREATE_NEW_CONSOLE)
            else: subprocess.Popen([exe]+args,start_new_session=True)
            QMessageBox.information(self,"Calibration started","ArgyllCMS opened its guided calibration process. Follow its prompts and return here when it finishes. The ICC profile will be written beside the selected output base.")
        except Exception as exc: QMessageBox.warning(self,"Could not start calibration",str(exc))

    def _start_camera(self):
        fields=[self.chart_image.text(),self.chart_layout.text(),self.chart_reference.text()]
        if not all(os.path.isfile(x) for x in fields):
            QMessageBox.information(self,"Camera profile","Choose a valid chart photograph, .cht recognition file, and reference-values file."); return
        base=os.path.splitext(self.camera_output.text().strip())[0]; Path(base).parent.mkdir(parents=True,exist_ok=True)
        scan=argyll_tool(self.argyll_edit.text(),"scanin")
        self._camera_colprof=(base,self.camera_desc.text().strip() or "PhotoLab Camera Profile")
        self._run_process(scan,build_scanin_args(*fields,base),self._scanin_finished)

    def _run_process(self, program, args, finished):
        if self.process and self.process.state()!=QProcess.ProcessState.NotRunning:
            QMessageBox.information(self,"Calibration","Another calibration task is still running."); return
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(lambda:self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self.process.finished.connect(finished); self.log.appendPlainText("\n"+subprocess.list2cmdline([program]+args)); self.process.start(program,args)

    def _scanin_finished(self,code,_status):
        if code!=0: QMessageBox.warning(self,"Chart recognition failed","Argyll scanin did not complete. Review the log and chart framing/reference files."); return
        base,desc=self._camera_colprof; self._run_process(argyll_tool(self.argyll_edit.text(),"colprof"),build_colprof_args(base,desc),self._colprof_finished)
    def _colprof_finished(self,code,_status):
        base,_=self._camera_colprof; profile=base+".icc"
        if code==0 and os.path.isfile(profile):
            self.icc_edit.setText(profile); self._validate_profile(); QMessageBox.information(self,"Camera profile created",f"Created:\n{profile}\n\nReview the Delta E fit information in the log before using the profile.")
        else: QMessageBox.warning(self,"Profile creation failed","Argyll colprof did not create the ICC profile. Review the log.")

    def _validate_profile(self):
        ok,msg=validate_icc(self.icc_edit.text().strip()); self.icc_status.setText(("Valid ICC profile: " if ok else "Invalid: ")+msg); self.icc_status.setStyleSheet("color:#43c96b;" if ok else "color:#e05252;"); return ok
    def _install_profile(self):
        if not self._validate_profile(): return
        if QMessageBox.question(self,"Install display profile","Install this as the operating system profile for display " + str(self.display_spin.value()) + " and load its calibration curves?") != QMessageBox.StandardButton.Yes: return
        exe=argyll_tool(self.argyll_edit.text(),"dispwin"); args=["-d",str(self.display_spin.value()),"-I",self.icc_edit.text().strip()]
        self._run_process(exe,args,self._install_finished)
    def _install_finished(self,code,_status):
        QMessageBox.information(self,"Display profile","Profile installed and calibration loaded." if code==0 else "Installation failed. Review the log; elevated operating-system permission may be required.")
