"""Guided non-AI Restoration Studio and optional AI Restoration Lab."""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtCore import Qt,QThread,pyqtSignal
from PyQt6.QtWidgets import (QApplication,QCheckBox,QComboBox,QDialog,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QMessageBox,QProgressDialog,QPushButton,QScrollArea,QSlider,QSpinBox,QTabWidget,QVBoxLayout,QWidget,QDialogButtonBox)
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

class LocalAIWorker(QThread):
    progress=pyqtSignal(int,int,str);candidate_ready=pyqtSignal(int,str,object);completed=pyqtSignal(int);failed=pyqtSignal(str);cancelled=pyqtSignal(int)
    def __init__(self,provider,input_path,output_dir,capability,fidelity,count):
        super().__init__();self.provider=provider;self.input_path=Path(input_path);self.output_dir=Path(output_dir);self.capability=capability;self.fidelity=fidelity;self.count=count;self._cancel=False
    def cancel(self):self._cancel=True
    def run(self):
        generated=0
        try:
            for index in range(1,self.count+1):
                if self._cancel:self.cancelled.emit(generated);return
                self.progress.emit(index-1,self.count,f"Generating candidate {index} of {self.count} locally…")
                output=self.output_dir/f"candidate_{index}.png";report=run_ai_provider(self.provider,self.input_path,output,self.capability,self.fidelity,index)
                generated=index;self.candidate_ready.emit(index,str(output),report);self.progress.emit(index,self.count,f"Candidate {index} of {self.count} ready")
            self.completed.emit(generated)
        except Exception as exc:self.failed.emit(str(exc))

class RestorationStudioDialog(QDialog):
    def __init__(self,parent,image,source_path=""):
        super().__init__(parent);self.setWindowTitle("Restore & Colorize — Restoration Studio");self.resize(1320,850);self.image=image.copy();self.source_path=source_path;self.mask=np.zeros(image.shape[:2],np.uint8);self.result_image=image.copy();self.ai_results=[];self.ai_reports=[];self._ai_worker=None;self._ai_progress=None;self._temp=tempfile.TemporaryDirectory(prefix="photolab_restore_")
        root=QHBoxLayout(self);panel=QScrollArea();panel.setWidgetResizable(True);panel.setMaximumWidth(430);body=QWidget();pv=QVBoxLayout(body);panel.setWidget(body);root.addWidget(panel);self.tabs=QTabWidget();self.tabs.setObjectName("RestorationWorkspaceTabs");self.tabs.setAccessibleName("Restoration workspaces");pv.addWidget(self.tabs);self._restore_tab();self._ai_tab();self.status=QLabel("Preview changes here, then choose Apply & Save Copy to create a new image file.");self.status.setWordWrap(True);pv.addWidget(self.status);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Save);buttons.rejected.connect(self._cancel_dialog);buttons.accepted.connect(self._accept);self.save_button=buttons.button(QDialogButtonBox.StandardButton.Save);self.save_button.setText("Apply && Save Copy…");self.save_button.setToolTip("Prepare the displayed restoration, then choose where to save a new image");pv.addWidget(buttons)
        right=QVBoxLayout();row=QHBoxLayout();row.addWidget(QLabel("View"));self.view=QComboBox();self.view.addItems(["Restored Preview","Original","Repair Mask","AI Confidence"]);self.view.currentIndexChanged.connect(self.refresh);row.addWidget(self.view);row.addStretch();right.addLayout(row);self.canvas=ClickImage();self.canvas.clicked.connect(self._clicked);right.addWidget(self.canvas,1);root.addLayout(right,1);self.refresh()
    def _restore_tab(self):
        tab=QWidget();f=QFormLayout(tab);self.controls={}
        for key,label,lo,hi,default in (("sensitivity","Crease sensitivity",1,100,55),("max_width","Largest crease width",2,40,12),("repair_radius","Content-fill radius",1,20,4),("stain","Stain suppression",0,100,0),("texture","Paper texture suppression",0,100,0),("fade","Fading correction",0,100,0),("silvering","Silvering correction",0,100,0),("deblur_radius","Defocus radius",0,12,0),("snr","Deblur SNR",5,100,30),("denoise","Grain-aware denoise",0,100,0),("sharpen","Detail recovery",0,100,0)):
            s=QSpinBox();s.setRange(lo,hi);s.setValue(default);s.valueChanged.connect(self.refresh);f.addRow(label,s);self.controls[key]=s
        self.join_tears=QCheckBox("Join tear edges before filling");self.join_tears.setChecked(True);self.join_tears.toggled.connect(self.refresh);f.addRow(self.join_tears);detect=QPushButton("Detect Creases and Scratches");detect.clicked.connect(self._detect);f.addRow(detect);self.brush_mode=QComboBox();self.brush_mode.addItems(["Paint repair mask","Erase repair mask"]);f.addRow("Repair brush",self.brush_mode);self.brush=QSpinBox();self.brush.setRange(2,300);self.brush.setValue(24);f.addRow("Brush size",self.brush);clear=QPushButton("Clear Repair Mask");clear.clicked.connect(self._clear_mask);f.addRow(clear);self.tabs.addTab(tab,"Restoration Studio")
    def _ai_tab(self):
        tab=QWidget();v=QVBoxLayout(tab);self.pack_status=QLabel();self.pack_status.setWordWrap(True);v.addWidget(self.pack_status);self.provider=QComboBox();self.provider.currentIndexChanged.connect(self._provider_changed);self.capability=QComboBox();form=QFormLayout();form.addRow("Provider",self.provider);form.addRow("Operation",self.capability);self.fidelity=QSlider(Qt.Orientation.Horizontal);self.fidelity.setRange(0,100);self.fidelity.setValue(70);form.addRow("Identity / fidelity",self.fidelity);self.candidates=QSpinBox();self.candidates.setRange(1,4);self.candidates.setValue(2);form.addRow("Candidate results",self.candidates);self.blend=QSlider(Qt.Orientation.Horizontal);self.blend.setRange(0,100);self.blend.setValue(100);self.blend.valueChanged.connect(self.refresh);form.addRow("AI blend",self.blend);self.select_candidate=QComboBox();self.select_candidate.currentIndexChanged.connect(self.refresh);form.addRow("Preview candidate",self.select_candidate);self.mask_ai=QCheckBox("Apply AI only through repair mask");self.mask_ai.toggled.connect(self.refresh);form.addRow(self.mask_ai);v.addLayout(form);self.run_ai_button=QPushButton("Run Free Local AI Model (Offline)");self.run_ai_button.clicked.connect(self._run_ai);v.addWidget(self.run_ai_button);self.export_candidate_button=QPushButton("Export Selected Candidate…");self.export_candidate_button.setEnabled(False);self.export_candidate_button.clicked.connect(self._export_candidate);v.addWidget(self.export_candidate_button);temporary=QLabel("Candidates appear in the preview and selector above. They remain temporary until you export one, or choose Apply & Save Copy to save the blended final result.");temporary.setWordWrap(True);temporary.setStyleSheet("color:#aebbd0;padding:5px");v.addWidget(temporary);config=QPushButton("Open Configuration / INI Editor");config.clicked.connect(self._open_config);v.addWidget(config);warning=QLabel("LOCAL / OFFLINE — no Google quota, API key, subscription, or usage fee. AI colors are plausible interpretations, not historical evidence.");warning.setWordWrap(True);warning.setStyleSheet("background:#173a2a;color:#b8f5cf;padding:9px;border-radius:6px");v.addWidget(warning);v.addStretch();self.tabs.addTab(tab,"AI Restoration Lab");self._load_pack()
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
        previous=self.capability.currentText();self.capability.clear();provider=self.provider.currentData();self.ai_results=[];self.ai_reports=[];self.select_candidate.clear();self.export_candidate_button.setEnabled(False)
        if provider:
            self.capability.addItems(provider.get("capabilities",[]))
            if previous in provider.get("capabilities",[]):self.capability.setCurrentText(previous)
    def _run_ai(self):
        provider=self.provider.currentData();capability=self.capability.currentText()
        if not provider:QMessageBox.information(self,"AI Model Pack","Configure a valid external model-pack folder first.");return
        if capability not in provider.get("capabilities",[]):QMessageBox.warning(self,"Unsupported",f"This provider does not support {capability}.");return
        if self._ai_worker is not None and self._ai_worker.isRunning():QMessageBox.information(self,"Local AI","A local model is already running.");return
        count=self.candidates.value();progress=QProgressDialog("Preparing image for the local model…","Cancel",0,count,self);progress.setWindowTitle("Local AI — Offline Processing");progress.setWindowModality(Qt.WindowModality.WindowModal);progress.setMinimumDuration(0);progress.setAutoClose(False);progress.setAutoReset(False);progress.show();QApplication.processEvents();input_path=Path(self._temp.name)/"input.png"
        try:_write(input_path,self._base_result())
        except Exception as exc:progress.close();QMessageBox.critical(self,"AI preparation failed",str(exc));return
        self.ai_results=[];self.ai_reports=[];self.select_candidate.clear();self.export_candidate_button.setEnabled(False);self.run_ai_button.setEnabled(False)
        worker=LocalAIWorker(provider,input_path,Path(self._temp.name),capability,self.fidelity.value()/100,count);self._ai_worker=worker;self._ai_progress=progress;progress.canceled.connect(worker.cancel);worker.progress.connect(self._ai_progress_changed);worker.candidate_ready.connect(self._ai_candidate_ready);worker.completed.connect(self._ai_completed);worker.cancelled.connect(self._ai_cancelled);worker.failed.connect(self._ai_failed);worker.finished.connect(self._ai_worker_finished);worker.start()
    def _ai_progress_changed(self,value,total,message):
        if self._ai_progress is not None:self._ai_progress.setMaximum(total);self._ai_progress.setLabelText(message);self._ai_progress.setValue(value)
    def _ai_candidate_ready(self,index,path,report):
        try:image=_read(path)
        except Exception as exc:self._ai_failed(str(exc));return
        self.ai_results.append(image);self.ai_reports.append(report);self.select_candidate.addItem(f"Candidate {index}");self.select_candidate.setCurrentIndex(self.select_candidate.count()-1);self.export_candidate_button.setEnabled(True);self.status.setText(f"Candidate {index} is ready and displayed. Use Original/Restored Preview to compare it.");self.view.setCurrentText("Restored Preview");self.refresh()
    def _close_ai_progress(self):
        if self._ai_progress is not None:self._ai_progress.close();self._ai_progress.deleteLater();self._ai_progress=None
        self.run_ai_button.setEnabled(True)
    def _ai_completed(self,count):self._close_ai_progress();self.status.setText(f"Generated {count} candidate(s) locally. Choose one above, then export it or use Apply & Save Copy for the final blend.");self.refresh()
    def _ai_cancelled(self,count):self._close_ai_progress();self.status.setText(f"Local AI stopped after {count} candidate(s). Completed candidates remain available for preview.")
    def _ai_failed(self,message):self._close_ai_progress();QMessageBox.critical(self,"Local AI restoration failed",message)
    def _ai_worker_finished(self):
        if self._ai_worker is not None:self._ai_worker.deleteLater();self._ai_worker=None
    def _export_candidate(self):
        image=self._current_ai()
        if image is None:QMessageBox.information(self,"Export Candidate","Generate and select a candidate first.");return
        source=Path(self.source_path) if self.source_path else Path("restored.png");default=source.with_name(f"{source.stem}_candidate_{self.select_candidate.currentIndex()+1}.png");path,_=QFileDialog.getSaveFileName(self,"Export AI Candidate",str(default),"PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)")
        if not path:return
        try:_write(path,image);self.status.setText(f"Candidate exported to {path}")
        except Exception as exc:QMessageBox.critical(self,"Candidate export failed",str(exc))
    def _open_config(self):
        from configuration_dialog import ConfigurationDialog
        dialog=ConfigurationDialog(self);dialog.exec();self._load_pack()
    def _cancel_dialog(self):
        if self._ai_worker is not None and self._ai_worker.isRunning():self._ai_worker.cancel();self.status.setText("Stopping after the current local candidate finishes…");return
        self.reject()
    def closeEvent(self,event):
        if self._ai_worker is not None and self._ai_worker.isRunning():self._ai_worker.cancel();self.status.setText("Stopping after the current local candidate finishes…");event.ignore();return
        super().closeEvent(event)
    def _accept(self):
        self.save_button.setEnabled(False);self.status.setText("Preparing the displayed restoration for Save Copy…");QApplication.processEvents()
        try:self.result_image=self.result()
        except Exception as exc:self.save_button.setEnabled(True);QMessageBox.critical(self,"Could not prepare restored image",str(exc));return
        self.accept()
