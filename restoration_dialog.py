"""Guided non-AI Restoration Studio and optional AI Restoration Lab."""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox,QComboBox,QDialog,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QMessageBox,QProgressDialog,QPushButton,QScrollArea,QSlider,QSpinBox,QTabWidget,QVBoxLayout,QWidget,QDialogButtonBox)
from distraction_dialog import ClickImage
from external_paths import resolve_path
from restoration import (ai_confidence_map,blend_ai_result,detect_crease_scratch_mask,
    load_model_pack,restore_photo,run_ai_provider)

def _write(path,image):
    ok,data=cv2.imencode(Path(path).suffix or ".png",image)
    if not ok:raise ValueError("Could not encode model input")
    data.tofile(str(path))
def _read(path):
    data=np.fromfile(str(path),np.uint8);image=cv2.imdecode(data,cv2.IMREAD_UNCHANGED)
    if image is None:raise ValueError("Model output is not a readable image")
    if image.ndim==2:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    return image

class RestorationStudioDialog(QDialog):
    def __init__(self,parent,image,source_path=""):
        super().__init__(parent);self.setWindowTitle("Restore & Colorize — Restoration Studio");self.resize(1320,850);self.image=image.copy();self.source_path=source_path;self.mask=np.zeros(image.shape[:2],np.uint8);self.result_image=image.copy();self.ai_results=[];self.ai_reports=[];self._temp=tempfile.TemporaryDirectory(prefix="photolab_restore_")
        root=QHBoxLayout(self);panel=QScrollArea();panel.setWidgetResizable(True);panel.setMaximumWidth(430);body=QWidget();pv=QVBoxLayout(body);panel.setWidget(body);root.addWidget(panel);self.tabs=QTabWidget();self.tabs.setObjectName("RestorationWorkspaceTabs");self.tabs.setAccessibleName("Restoration workspaces");pv.addWidget(self.tabs);self._restore_tab();self._ai_tab();self.status=QLabel("All restoration is previewed non-destructively. Apply creates a new file.");self.status.setWordWrap(True);pv.addWidget(self.status);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Apply);buttons.rejected.connect(self.reject);buttons.accepted.connect(self._accept);pv.addWidget(buttons)
        right=QVBoxLayout();row=QHBoxLayout();row.addWidget(QLabel("View"));self.view=QComboBox();self.view.addItems(["Restored Preview","Original","Repair Mask","AI Confidence"]);self.view.currentIndexChanged.connect(self.refresh);row.addWidget(self.view);row.addStretch();right.addLayout(row);self.canvas=ClickImage();self.canvas.clicked.connect(self._clicked);right.addWidget(self.canvas,1);root.addLayout(right,1);self.refresh()
    def _restore_tab(self):
        tab=QWidget();f=QFormLayout(tab);self.controls={}
        for key,label,lo,hi,default in (("sensitivity","Crease sensitivity",1,100,55),("max_width","Largest crease width",2,40,12),("repair_radius","Content-fill radius",1,20,4),("stain","Stain suppression",0,100,0),("texture","Paper texture suppression",0,100,0),("fade","Fading correction",0,100,0),("silvering","Silvering correction",0,100,0),("deblur_radius","Defocus radius",0,12,0),("snr","Deblur SNR",5,100,30),("denoise","Grain-aware denoise",0,100,0),("sharpen","Detail recovery",0,100,0)):
            s=QSpinBox();s.setRange(lo,hi);s.setValue(default);s.valueChanged.connect(self.refresh);f.addRow(label,s);self.controls[key]=s
        self.join_tears=QCheckBox("Join tear edges before filling");self.join_tears.setChecked(True);self.join_tears.toggled.connect(self.refresh);f.addRow(self.join_tears);detect=QPushButton("Detect Creases and Scratches");detect.clicked.connect(self._detect);f.addRow(detect);self.brush_mode=QComboBox();self.brush_mode.addItems(["Paint repair mask","Erase repair mask"]);f.addRow("Repair brush",self.brush_mode);self.brush=QSpinBox();self.brush.setRange(2,300);self.brush.setValue(24);f.addRow("Brush size",self.brush);clear=QPushButton("Clear Repair Mask");clear.clicked.connect(self._clear_mask);f.addRow(clear);self.tabs.addTab(tab,"Restoration Studio")
    def _ai_tab(self):
        tab=QWidget();v=QVBoxLayout(tab);self.pack_status=QLabel();self.pack_status.setWordWrap(True);v.addWidget(self.pack_status);self.provider=QComboBox();self.provider.currentIndexChanged.connect(self._provider_changed);self.capability=QComboBox();form=QFormLayout();form.addRow("Provider",self.provider);form.addRow("Operation",self.capability);self.fidelity=QSlider(Qt.Orientation.Horizontal);self.fidelity.setRange(0,100);self.fidelity.setValue(70);form.addRow("Identity / fidelity",self.fidelity);self.candidates=QSpinBox();self.candidates.setRange(1,4);self.candidates.setValue(2);form.addRow("Candidate results",self.candidates);self.blend=QSlider(Qt.Orientation.Horizontal);self.blend.setRange(0,100);self.blend.setValue(100);self.blend.valueChanged.connect(self.refresh);form.addRow("AI blend",self.blend);self.select_candidate=QComboBox();self.select_candidate.currentIndexChanged.connect(self.refresh);form.addRow("Result",self.select_candidate);self.mask_ai=QCheckBox("Apply AI only through repair mask");self.mask_ai.toggled.connect(self.refresh);form.addRow(self.mask_ai);v.addLayout(form);run=QPushButton("Run Local AI Model");run.clicked.connect(self._run_ai);v.addWidget(run);config=QPushButton("Open Configuration / INI Editor");config.clicked.connect(self._open_config);v.addWidget(config);warning=QLabel("AI output may invent textures, facial features, objects, and colors. Colorization is plausible—not historical evidence. Keep the original and inspect at 100%.");warning.setWordWrap(True);warning.setStyleSheet("background:#4a3213;color:#ffd98a;padding:9px;border-radius:6px");v.addWidget(warning);v.addStretch();self.tabs.addTab(tab,"AI Restoration Lab");self._load_pack()
    def settings(self):
        values={k:v.value() for k,v in self.controls.items()};values["join_tears"]=self.join_tears.isChecked();return values
    def _detect(self):
        self.mask=detect_crease_scratch_mask(self.image,self.controls["sensitivity"].value(),18,self.controls["max_width"].value());self.view.setCurrentText("Repair Mask");self.status.setText(f"Detected {cv2.countNonZero(self.mask):,} candidate damage pixels. Paint or erase before applying.");self.refresh()
    def _clicked(self,x,y):
        px=int(x*(self.mask.shape[1]-1));py=int(y*(self.mask.shape[0]-1));value=255 if self.brush_mode.currentIndex()==0 else 0;cv2.circle(self.mask,(px,py),self.brush.value(),value,-1,cv2.LINE_AA);self.view.setCurrentText("Repair Mask");self.refresh()
    def _clear_mask(self):self.mask[:]=0;self.refresh()
    def _base_result(self):return restore_photo(self.image,self.settings(),self.mask)
    def _current_ai(self):
        if not self.ai_results:return None
        return self.ai_results[max(0,min(self.select_candidate.currentIndex(),len(self.ai_results)-1))]
    def result(self):
        base=self._base_result();generated=self._current_ai()
        if generated is None:return base
        if self.capability.currentText()=="super_resolution" and self.blend.value()==100 and not self.mask_ai.isChecked():return generated
        return blend_ai_result(base,generated,self.blend.value(),self.mask if self.mask_ai.isChecked() else None)
    def refresh(self):
        if not hasattr(self,"view"):return
        mode=self.view.currentText()
        if mode=="Original":shown=self.image
        elif mode=="Repair Mask":shown=self.mask
        elif mode=="AI Confidence" and self._current_ai() is not None:shown=(ai_confidence_map(self._base_result(),self._current_ai())*255).astype(np.uint8)
        else:shown=self.result()
        self.canvas.show_image(shown)
    def _load_pack(self):
        self.provider.clear();folder=resolve_path("ai_restoration_model_pack")
        if not folder:self.pack_status.setText("No model pack configured. Restoration Studio remains fully available.");self.pack=None;return
        try:
            self.pack=load_model_pack(folder)
            for provider in self.pack["providers"]:self.provider.addItem(provider.get("name",provider.get("id","Provider")),provider)
            self.pack_status.setText(f"Model pack: {self.pack.get('name',Path(folder).name)}\n{len(self.pack['providers'])} local provider(s)")
            self._provider_changed()
        except Exception as exc:self.pack=None;self.pack_status.setText(f"Model pack unavailable: {exc}")
    def _provider_changed(self):
        previous=self.capability.currentText();self.capability.clear();provider=self.provider.currentData()
        if provider:
            self.capability.addItems(provider.get("capabilities",[]))
            if previous in provider.get("capabilities",[]):self.capability.setCurrentText(previous)
    def _run_ai(self):
        provider=self.provider.currentData();capability=self.capability.currentText()
        if not provider:QMessageBox.information(self,"AI Model Pack","Configure a valid external model-pack folder first.");return
        if capability not in provider.get("capabilities",[]):QMessageBox.warning(self,"Unsupported",f"This provider does not support {capability}.");return
        progress=QProgressDialog("Running local model…","Cancel",0,self.candidates.value(),self);progress.setWindowModality(Qt.WindowModality.WindowModal);input_path=Path(self._temp.name)/"input.png";_write(input_path,self._base_result());self.ai_results=[];self.ai_reports=[]
        try:
            for index in range(1,self.candidates.value()+1):
                if progress.wasCanceled():break
                output=Path(self._temp.name)/f"candidate_{index}.png";report=run_ai_provider(provider,input_path,output,capability,self.fidelity.value()/100,index);self.ai_results.append(_read(output));self.ai_reports.append(report);progress.setValue(index)
        except Exception as exc:QMessageBox.critical(self,"AI restoration failed",str(exc));return
        self.select_candidate.clear();self.select_candidate.addItems([f"Candidate {i+1}" for i in range(len(self.ai_results))]);self.status.setText(f"Generated {len(self.ai_results)} candidate(s) locally. Compare them and control the blend.");self.refresh()
    def _open_config(self):
        from configuration_dialog import ConfigurationDialog
        dialog=ConfigurationDialog(self);dialog.exec();self._load_pack()
    def _accept(self):self.result_image=self.result();self.accept()
