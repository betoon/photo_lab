"""map_view.py — optional GPS map for PhotoLab (Leaflet via local HTML).

No hard dependency on Qt WebEngine: writes a self-contained HTML file and
opens it in the system browser, and shows a simple in-app list of pins.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox,
)


def collect_gps_points(paths: List[str]) -> List[dict]:
    """[{path, name, lat, lon}, …] for images that have GPS EXIF."""
    from imaging import extract_gps
    out = []
    for p in paths:
        gps = extract_gps(p)
        if not gps:
            continue
        lat, lon = gps
        out.append({
            "path": p,
            "name": os.path.basename(p),
            "lat": float(lat),
            "lon": float(lon),
        })
    return out


def write_leaflet_html(points: List[dict], out_path: str) -> str:
    """Write a standalone Leaflet map (CDN) with markers."""
    if not points:
        raise ValueError("No GPS points")
    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)
    markers_js = json.dumps([
        {"lat": p["lat"], "lon": p["lon"], "name": p["name"]} for p in points
    ])
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>PhotoLab Map — {len(points)} photo(s)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ height: 100%; margin: 0; background: #121212; }}
  .info {{ position: absolute; z-index: 1000; top: 10px; left: 50px;
           background: #1a1a1a; color: #ddd; padding: 8px 12px;
           border-radius: 6px; font: 13px sans-serif; }}
</style>
</head><body>
<div class="info">PhotoLab · {len(points)} geotagged photo(s)</div>
<div id="map"></div>
<script>
  const pts = {markers_js};
  const map = L.map('map').setView([{center_lat}, {center_lon}], 12);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap'
  }}).addTo(map);
  const group = L.featureGroup();
  pts.forEach(p => {{
    const m = L.marker([p.lat, p.lon]).bindPopup(p.name);
    group.addLayer(m);
  }});
  group.addTo(map);
  if (pts.length > 1) map.fitBounds(group.getBounds().pad(0.2));
</script>
</body></html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


class MapDialog(QDialog):
    """List geotagged images + open interactive Leaflet map in the browser."""

    def __init__(self, paths: List[str], parent=None, on_open_path=None):
        super().__init__(parent)
        self.setWindowTitle("Map — GPS from EXIF")
        self.resize(480, 420)
        self._on_open = on_open_path
        self.points = collect_gps_points(paths)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"{len(self.points)} geotagged / {len(paths)} selected or in folder"
        ))
        self.list = QListWidget()
        for p in self.points:
            item = QListWidgetItem(
                f"{p['name']}  ({p['lat']:.5f}, {p['lon']:.5f})"
            )
            item.setData(Qt.ItemDataRole.UserRole, p["path"])
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(self._open_item)
        lay.addWidget(self.list)

        row = QHBoxLayout()
        open_btn = QPushButton("Open map in browser…")
        open_btn.clicked.connect(self._open_map)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(open_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        lay.addLayout(row)

        if not self.points:
            open_btn.setEnabled(False)

    def _open_item(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and self._on_open:
            self._on_open(path)

    def _open_map(self):
        if not self.points:
            QMessageBox.information(self, "Map", "No GPS data in EXIF.")
            return
        td = tempfile.gettempdir()
        path = os.path.join(td, "photolab_map.html")
        write_leaflet_html(self.points, path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
