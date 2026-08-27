"""PhotoLab Remove Distractions workspace."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from distractions import (apply_distraction_operations, apply_reflection_adjustment,
    build_sensor_dust_map, detect_dust_spots, load_images, operations_mask,
    reflection_mask, separate_reflections, smart_object_mask)


def _pixmap(image):
    if image.ndim == 2:
        q = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class ClickImage(QLabel):
    clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 420); self.setStyleSheet("background:#10151d;border:1px solid #334155;border-radius:8px")
        self._image_size = (1,1); self._shown = None

    def show_image(self, image):
        self._image_size = (image.shape[1], image.shape[0]); self._shown = _pixmap(image); self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event); self._fit()

    def _fit(self):
        if self._shown: self.setPixmap(self._shown.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event):
        pix = self.pixmap()
        if pix:
            left=(self.width()-pix.width())/2; top=(self.height()-pix.height())/2
            x=(event.position().x()-left)/max(pix.width(),1); y=(event.position().y()-top)/max(pix.height(),1)
            if 0 <= x <= 1 and 0 <= y <= 1: self.clicked.emit(float(x),float(y))
        super().mousePressEvent(event)


class DistractionDialog(QDialog):
    """A deliberately guided workspace; recipe is only changed on Apply."""
    def __init__(self, parent, image, recipe, source_path=""):
        super().__init__(parent); self.setWindowTitle("Remove Distractions")
        self.resize(1220, 780); self.setMinimumSize(900, 620)
        self.image=image.copy(); self.recipe=copy.deepcopy(recipe); self.source_path=source_path
        self.operations=copy.deepcopy(getattr(recipe,"distraction_operations",[]) or [])
        self.undo=[]; self.pending=None; self.detected=[]; self.external_preview=None
        self.setStyleSheet("QDialog{background:#151b24;color:#e5edf8}QGroupBox{border:1px solid #334155;border-radius:7px;margin-top:10px;padding-top:8px;font-weight:600}QGroupBox::title{subcontrol-origin:margin;left:9px;padding:0 5px}QPushButton{background:#2563a8;color:white;border:0;border-radius:6px;padding:7px 10px}QPushButton:hover{background:#3480d1}QComboBox,QSpinBox{background:#202b39;color:#edf5ff;border:1px solid #405268;border-radius:4px;padding:4px}QTabWidget::pane{border:1px solid #334155}QTabBar::tab{background:#202b39;padding:8px 14px}QTabBar::tab:selected{background:#2563a8}")
        root=QHBoxLayout(self); controls=QScrollArea(); controls.setWidgetResizable(True); controls.setMaximumWidth(390)
        panel=QWidget(); pv=QVBoxLayout(panel); controls.setWidget(panel); root.addWidget(controls)
        self.tabs=QTabWidget(); pv.addWidget(self.tabs)
        self._manual_tab(); self._dust_tab(); self._reflection_tab(); self._smart_tab()
        self.status=QLabel("Choose a tool, then click the photograph."); self.status.setWordWrap(True); pv.addWidget(self.status)
        row=QHBoxLayout(); undo=QPushButton("Undo"); undo.clicked.connect(self._undo); row.addWidget(undo)
        clear=QPushButton("Clear All"); clear.clicked.connect(self._clear); row.addWidget(clear); pv.addLayout(row)
        buttons=QHBoxLayout(); cancel=QPushButton("Cancel"); cancel.clicked.connect(self.reject); buttons.addWidget(cancel)
        apply=QPushButton("Apply to Recipe"); apply.setStyleSheet("background:#18a678;color:white;border-radius:6px;padding:9px;font-weight:700"); apply.clicked.connect(self.accept); buttons.addWidget(apply); pv.addLayout(buttons)
        view=QVBoxLayout(); top=QHBoxLayout(); self.view_combo=QComboBox(); self.view_combo.addItems(["Corrected Preview","Original","Removal Mask","Reflection Mask"]); self.view_combo.currentIndexChanged.connect(self.refresh); top.addWidget(QLabel("View")); top.addWidget(self.view_combo); top.addStretch(); view.addLayout(top)
        self.canvas=ClickImage(); self.canvas.clicked.connect(self._clicked); view.addWidget(self.canvas,1); root.addLayout(view,1); self.refresh()

    def _manual_tab(self):
        tab=QWidget(); f=QFormLayout(tab); self.tool=QComboBox(); self.tool.addItems(["Heal","Clone","Content-Aware","Wire / Hair"]); f.addRow("Tool",self.tool)
        self.radius=QSpinBox(); self.radius.setRange(2,250); self.radius.setValue(18); self.radius.setSuffix(" px"); f.addRow("Brush size",self.radius)
        help=QLabel("Heal: click a spot. Clone: click a clean source, then its destination. Wire: click both ends. Content-Aware removes a painted-size area."); help.setWordWrap(True); f.addRow(help); self.tabs.addTab(tab,"1  Manual")

    def _dust_tab(self):
        tab=QWidget(); v=QVBoxLayout(tab); form=QFormLayout(); self.dust_sens=QSpinBox(); self.dust_sens.setRange(1,100); self.dust_sens.setValue(60); form.addRow("Sensitivity",self.dust_sens); self.dust_max=QSpinBox(); self.dust_max.setRange(3,80); self.dust_max.setValue(20); form.addRow("Largest spot",self.dust_max); v.addLayout(form)
        b=QPushButton("Detect on This Image"); b.clicked.connect(self._detect_dust); v.addWidget(b)
        b=QPushButton("Build Reusable Folder Dust Map…"); b.clicked.connect(self._folder_dust); v.addWidget(b)
        b=QPushButton("Save Dust Map…"); b.clicked.connect(self._save_dust); v.addWidget(b)
        b=QPushButton("Load Dust Map…"); b.clicked.connect(self._load_dust); v.addWidget(b); v.addStretch(); self.tabs.addTab(tab,"2  Dust")

    def _reflection_tab(self):
        tab=QWidget(); f=QFormLayout(tab); self.ref_enable=QCheckBox("Enable editable reflection layer"); self.ref_enable.setChecked(bool(getattr(self.recipe,"reflection_enabled",False))); self.ref_enable.toggled.connect(self.refresh); f.addRow(self.ref_enable)
        self.ref_controls={}
        for label,key,lo,hi,default in [("Detection sensitivity","sensitivity",1,100,55),("Layer opacity","strength",0,100,50),("Highlights","highlights",-100,0,-35),("Saturation","saturation",-100,100,0),("Color neutralization","neutralize",0,100,20),("Local contrast","contrast",-50,100,10),("Mask softness","blur",0,30,8)]:
            s=QSpinBox(); s.setRange(lo,hi); s.setValue(int(getattr(self.recipe,"reflection_"+key,default))); s.valueChanged.connect(self.refresh); f.addRow(label,s); self.ref_controls[key]=s
        b=QPushButton("Separate Reflections from Several Images…"); b.clicked.connect(self._separate); f.addRow(b)
        note=QLabel("Single-image adjustment reduces glare; it cannot reconstruct detail hidden by a reflection. Multi-image separation works best when the camera is steady and reflections move."); note.setWordWrap(True); f.addRow(note); self.tabs.addTab(tab,"3  Reflections")

    def _smart_tab(self):
        tab=QWidget(); v=QVBoxLayout(tab); info=QLabel("Smart Object Removal uses a two-click rectangle and foreground analysis to select a larger distraction, then content-aware filling removes it. Review the mask before applying."); info.setWordWrap(True); v.addWidget(info)
        b=QPushButton("Start Smart Selection"); b.clicked.connect(lambda:self._set_pending("smart")); v.addWidget(b)
        b=QPushButton("Detect Straight Wires / Hairs"); b.clicked.connect(self._detect_lines); v.addWidget(b); v.addStretch(); self.tabs.addTab(tab,"4  Smart")

    def _snapshot(self): self.undo.append(copy.deepcopy(self.operations))
    def _radius_norm(self): return self.radius.value()/max(min(self.image.shape[:2]),1)
    def _set_pending(self,kind): self.pending=(kind,[]); self.status.setText("Click two opposite corners around the object.")

    def _clicked(self,x,y):
        tool=self.tool.currentText(); r=self._radius_norm()
        if self.pending and self.pending[0]=="smart":
            self.pending[1].append((x,y))
            if len(self.pending[1])==2:
                (x1,y1),(x2,y2)=self.pending[1]; rect=[min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)]
                mask=smart_object_mask(self.image,rect); self._snapshot()
                # Store simplified normalized contours for resolution independence.
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 4:
                        continue
                    simplified=cv2.approxPolyDP(c,max(1.0,cv2.arcLength(c,True)*.005),True).reshape(-1,2)
                    polygon=[(float(px)/(mask.shape[1]-1),float(py)/(mask.shape[0]-1)) for px,py in simplified]
                    self.operations.append({"type":"smart","polygon":polygon,"enabled":True})
                self.pending=None; self.status.setText("Smart selection added. Inspect Removal Mask, then undo if needed."); self.refresh()
            return
        if tool=="Clone":
            if not self.pending or self.pending[0]!="clone": self.pending=("clone",[(x,y)]); self.status.setText("Source chosen. Click the destination."); return
            sx,sy=self.pending[1][0]; self._snapshot(); self.operations.append({"type":"clone","source_x":sx,"source_y":sy,"x":x,"y":y,"radius":r,"enabled":True}); self.pending=None
        elif tool=="Wire / Hair":
            if not self.pending or self.pending[0]!="wire": self.pending=("wire",[(x,y)]); self.status.setText("Click the other end of the wire or hair."); return
            self._snapshot(); self.operations.append({"type":"wire","points":[self.pending[1][0],(x,y)],"radius":r/2,"enabled":True}); self.pending=None
        else:
            self._snapshot(); self.operations.append({"type":"heal" if tool=="Heal" else "inpaint","x":x,"y":y,"radius":r,"enabled":True})
        self.status.setText(f"{tool} correction added ({len(self.operations)} total)."); self.refresh()

    def _detect_dust(self):
        found=detect_dust_spots(self.image,self.dust_sens.value(),self.dust_max.value()); self._snapshot(); self.operations.extend(found); self.detected=found; self.status.setText(f"Found {len(found)} candidate spots. Use Removal Mask to inspect them."); self.refresh()

    def _folder_dust(self):
        paths,_=QFileDialog.getOpenFileNames(self,"Select matching photographs",str(Path(self.source_path).parent if self.source_path else Path.home()),"Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if not paths:return
        try: found=build_sensor_dust_map(load_images(paths),self.dust_sens.value(),.5,self.dust_max.value())
        except Exception as e: QMessageBox.warning(self,"Dust Map",str(e)); return
        self._snapshot(); self.operations.extend(found); self.detected=found; self.status.setText(f"Reusable dust map contains {len(found)} recurring spots."); self.refresh()

    def _save_dust(self):
        path,_=QFileDialog.getSaveFileName(self,"Save Dust Map","sensor_dust_map.json","PhotoLab Dust Map (*.json)")
        if path:
            Path(path).write_text(json.dumps({"format":"photolab-dust-map-1","operations":[o for o in self.operations if o.get("dust_map") or o in self.detected]},indent=2),encoding="utf-8")

    def _load_dust(self):
        path,_=QFileDialog.getOpenFileName(self,"Load Dust Map","","PhotoLab Dust Map (*.json)")
        if path:
            try: ops=json.loads(Path(path).read_text(encoding="utf-8"))["operations"]
            except Exception as e: QMessageBox.warning(self,"Dust Map",str(e)); return
            self._snapshot(); self.operations.extend(ops); self.refresh()

    def _detect_lines(self):
        gray=cv2.cvtColor(self.image if self.image.dtype==np.uint8 else np.clip(self.image/256,0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
        lines=cv2.HoughLinesP(cv2.Canny(gray,60,160),1,np.pi/180,80,minLineLength=max(30,min(gray.shape)//8),maxLineGap=12)
        if lines is None: self.status.setText("No strong straight wires were found."); return
        self._snapshot(); h,w=gray.shape
        for line in lines[:20]:
            x1,y1,x2,y2=line[0]; self.operations.append({"type":"wire","points":[(x1/(w-1),y1/(h-1)),(x2/(w-1),y2/(h-1))],"radius":max(1,self.radius.value()/4)/min(h,w),"enabled":True})
        self.status.setText(f"Added {min(20,len(lines))} candidate lines. Inspect the mask carefully."); self.refresh()

    def _separate(self):
        paths,_=QFileDialog.getOpenFileNames(self,"Select reflection variants",str(Path(self.source_path).parent if self.source_path else Path.home()),"Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if len(paths)<2:return
        try: base,layer,mask,confidence,diagnostics=separate_reflections(load_images(paths))
        except Exception as e: QMessageBox.warning(self,"Reflection Separation",str(e)); return
        folder=QFileDialog.getExistingDirectory(self,"Save separation package",str(Path(paths[0]).parent))
        if not folder:return
        stem=Path(paths[0]).stem; outputs={"base":base,"reflection":layer,"mask":mask,"confidence":confidence}
        for name,data in outputs.items(): cv2.imencode(".png",data)[1].tofile(str(Path(folder)/f"{stem}_{name}.png"))
        Path(folder,f"{stem}_reflection_report.json").write_text(json.dumps({"sources":paths,"alignment":diagnostics},indent=2),encoding="utf-8")
        self.external_preview=base; self.status.setText("Separation package saved. The clean-base estimate is shown; open it in PhotoLab to continue editing."); self.refresh()

    def _undo(self):
        if self.undo:self.operations=self.undo.pop(); self.pending=None; self.refresh()
    def _clear(self): self._snapshot(); self.operations=[]; self.detected=[]; self.refresh()

    def _reflection_values(self): return {k:v.value() for k,v in self.ref_controls.items()}
    def refresh(self):
        mode=self.view_combo.currentText() if hasattr(self,"view_combo") else "Corrected Preview"
        rv=self._reflection_values() if hasattr(self,"ref_controls") else {"sensitivity":55,"blur":8,"strength":50,"highlights":-35,"saturation":0,"neutralize":20,"contrast":10}
        rm=reflection_mask(self.image,rv["sensitivity"],rv["blur"])
        if mode=="Original": shown=self.image
        elif mode=="Removal Mask": shown=operations_mask(self.image.shape,self.operations)
        elif mode=="Reflection Mask": shown=rm
        else:
            shown=self.external_preview if self.external_preview is not None else self.image
            if self.ref_enable.isChecked(): shown=apply_reflection_adjustment(shown,rm,rv["strength"],rv["highlights"],rv["saturation"],rv["neutralize"],rv["contrast"])
            shown=apply_distraction_operations(shown,self.operations)
        self.canvas.show_image(shown)

    def accept(self):
        self.recipe.distraction_operations=copy.deepcopy(self.operations)
        self.recipe.reflection_enabled=self.ref_enable.isChecked()
        for key,value in self._reflection_values().items(): setattr(self.recipe,"reflection_"+key,float(value))
        super().accept()
