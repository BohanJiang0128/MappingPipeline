import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


# --------------- Zoom + Pan Label for main photo ---------------

class ZoomPanLabel(QtWidgets.QLabel):
    """
    QLabel that supports:
    - loading an image from disk
    - zooming with scroll wheel
    - panning with left mouse drag

    It also emits zoom_delta(factor) so the viewer can sync avatar zoom.
    """
    zoom_delta = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background:#111; color:#ddd;")

        self._pixmap: QtGui.QPixmap | None = None
        self.zoom_factor: float = 1.0
        self.pan = QtCore.QPoint(0, 0)
        self._dragging = False
        self._last_pos: QtCore.QPoint | None = None

    def load_image(self, img_path: Path):
        """Load a new image and reset zoom/pan."""
        try:
            pil = Image.open(img_path).convert("RGB")
            data = pil.tobytes("raw", "RGB")
            qimg = QtGui.QImage(
                data, pil.width, pil.height, 3 * pil.width,
                QtGui.QImage.Format_RGB888
            )
            self._pixmap = QtGui.QPixmap.fromImage(qimg)
        except Exception:
            self._pixmap = None
        self.zoom_factor = 1.0
        self.pan = QtCore.QPoint(0, 0)
        self.update()

    def clear_image(self, text: str = ""):
        self._pixmap = None
        self.zoom_factor = 1.0
        self.pan = QtCore.QPoint(0, 0)
        self.setText(text)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        if self._pixmap is None:
            super().paintEvent(event)
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        rect = self.rect()
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        if img_w <= 0 or img_h <= 0:
            super().paintEvent(event)
            return

        lw = max(rect.width(), 1)
        lh = max(rect.height(), 1)
        base_scale = min(lw / img_w, lh / img_h)
        scale = base_scale * self.zoom_factor

        scaled_w = max(1, int(img_w * scale))
        scaled_h = max(1, int(img_h * scale))
        scaled = self._pixmap.scaled(
            scaled_w, scaled_h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        center = rect.center()
        top_left = QtCore.QPoint(
            center.x() - scaled.width() // 2 + self.pan.x(),
            center.y() - scaled.height() // 2 + self.pan.y(),
        )
        painter.drawPixmap(top_left, scaled)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if self._pixmap is None:
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        factor = 1.0
        if delta > 0:
            factor = 1.1
            self.zoom_factor *= 1.1
        elif delta < 0:
            factor = 0.9
            self.zoom_factor *= 0.9

        self.zoom_factor = max(0.2, min(self.zoom_factor, 5.0))
        self.update()

        if factor != 1.0:
            self.zoom_delta.emit(factor)

    def apply_external_zoom(self, factor: float):
        """Zoom triggered from the avatar side."""
        self.zoom_factor *= factor
        self.zoom_factor = max(0.2, min(self.zoom_factor, 5.0))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton and self._pixmap is not None:
            self._dragging = True
            self._last_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._dragging and self._pixmap is not None and self._last_pos is not None:
            pos = event.position().toPoint()
            delta = pos - self._last_pos
            self.pan += delta
            self._last_pos = pos
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self._last_pos = None
        super().mouseReleaseEvent(event)


# --------------- Image Panel (right, main photo) ---------------

class ImagePanel(QtWidgets.QWidget):
    image_changed = QtCore.Signal(Path)
    photo_zoom_delta = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images: list[Path] = []
        self.index = -1
        self.global_total = 0

        self.img_label = ZoomPanLabel()
        self.img_label.zoom_delta.connect(self._on_zoom_delta)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("color:#888;")

        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.img_label, 1)
        layout.addWidget(self.info_label)
        layout.addLayout(btn_row)

        self.update_state()

    @QtCore.Slot(float)
    def _on_zoom_delta(self, factor: float):
        self.photo_zoom_delta.emit(factor)

    def current_image(self) -> Path | None:
        if 0 <= self.index < len(self.images):
            return self.images[self.index]
        return None

    def set_images(self, images: list[Path], global_total: int | None = None, *, select_first: bool = True):
        """
        If select_first=False, we keep index=-1 (no selected photo),
        while still showing the montage list and enabling Prev/Next to start selection.
        """
        self.images = images
        if images and select_first:
            self.index = 0
        else:
            self.index = -1
        self.global_total = global_total if global_total is not None else len(images)
        self.update_state()
        if self.index != -1 and self.images:
            self.image_changed.emit(self.images[self.index])

    def set_current_image(self, img_path: Path):
        if not self.images:
            return
        try:
            idx = self.images.index(img_path)
        except ValueError:
            return
        self.index = idx
        self.update_state()
        self.image_changed.emit(self.images[self.index])

    def clear(self, msg="No images."):
        self.images = []
        self.index = -1
        self.global_total = 0
        self.img_label.clear_image(msg)
        self.info_label.setText("")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def prev_image(self):
        if not self.images:
            return
        if self.index == -1:
            self.index = len(self.images) - 1
        else:
            self.index = (self.index - 1) % len(self.images)
        self.update_state()
        self.image_changed.emit(self.images[self.index])

    def next_image(self):
        if not self.images:
            return
        if self.index == -1:
            self.index = 0
        else:
            self.index = (self.index + 1) % len(self.images)
        self.update_state()
        self.image_changed.emit(self.images[self.index])

    def update_state(self, empty_message="Select a vertex to see images."):
        if not self.images:
            self.img_label.clear_image(empty_message)
            self.info_label.setText("")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

        subset_total = len(self.images)
        Y = self.global_total if self.global_total else subset_total

        if self.index == -1:
            self.img_label.clear_image("No photo selected.\n(Click a montage photo or press Next/Prev)")
            self.info_label.setText(f"0/{subset_total} photos  •  subset of {Y} total")
            return

        img_path = self.images[self.index]
        try:
            self.img_label.load_image(img_path)
        except Exception as e:
            self.img_label.clear_image(f"Failed to load image:\n{img_path.name}\n{e}")

        self.info_label.setText(
            f"{self.index+1}/{subset_total} photos  •  subset of {Y} total  •  {img_path.name}"
        )


# --------------- Montage Panel ---------------

class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class MontagePanel(QtWidgets.QScrollArea):
    image_clicked = QtCore.Signal(Path)

    def __init__(self, parent=None, thumb_size: int = 140):
        super().__init__(parent)
        self.thumb_size = thumb_size

        self.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(container)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(4)
        self.setWidget(container)

        self.current_images: list[Path] = []

    def clear_montage(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.current_images = []

    def set_images(self, images: list[Path]):
        self.clear_montage()
        self.current_images = images

        if not self.current_images:
            label = QtWidgets.QLabel("No images.")
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.grid.addWidget(label, 0, 0)
            return

        cols = 2  # vertical-ish montage: fewer columns, taller layout
        for idx, img_path in enumerate(self.current_images):
            row = idx // cols
            col = idx % cols

            lbl = ClickableLabel()
            lbl.setFixedSize(self.thumb_size, self.thumb_size)
            lbl.setFrameShape(QtWidgets.QFrame.Box)
            lbl.setLineWidth(1)
            lbl.setStyleSheet("background-color: #111;")

            try:
                pil = Image.open(img_path).convert("RGB")
                data = pil.tobytes("raw", "RGB")
                qimg = QtGui.QImage(
                    data, pil.width, pil.height, 3 * pil.width,
                    QtGui.QImage.Format_RGB888
                )
                pix = QtGui.QPixmap.fromImage(qimg)
                scaled = pix.scaled(
                    self.thumb_size, self.thumb_size,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                lbl.setPixmap(scaled)
            except Exception:
                lbl.setText(img_path.name[:12])

            lbl.clicked.connect(lambda p=img_path: self.image_clicked.emit(p))
            self.grid.addWidget(lbl, row, col)


# --------------- Main Viewer ---------------

class PatientViewer(QtWidgets.QWidget):
    """
    Layout:
      [ Montage | Avatar | Photo ]
      Stretch ratios: 2 | 2 | 3

    Startup:
      - Patient dropdown with "ALL patients" first.
      - Default selection: first individual patient (if exists), not ALL.

    Camera:
      self.plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
    """

    def __init__(self, data_dir: Path, assets_dir: Path, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.assets_dir = assets_dir

        self.avatar_paths: dict[tuple[str, int], Path] = {}
        self.current_gender: str = "m"
        self.current_bmi_scale: int = 3
        self._collect_avatar_files()

        self.mesh: pv.PolyData | None = None
        self.base_colors: np.ndarray | None = None
        self.vertex_to_images: dict[int, list[Path]] = {}
        self.image_vertex_count: dict[Path, int] = {}

        self.patient_rgb_cache: dict[Path, dict | None] = {}
        self.image_colors_cache: dict[Path, np.ndarray] = {}
        self.patient_aggregate_colors: dict[Path, np.ndarray] = {}

        self.highlight_actor = None
        self.mesh_actor = None

        self.current_vertex: int | None = None
        self.current_patient_idx: int = 0    # index in dropdown (0 = ALL), but we won't start with 0
        self.texture_mode_on: bool = False

        self.npts_ref: int | None = None

        self.all_images: list[Path] = []
        self.images_by_patient: dict[Path, list[Path]] = {}

        # ----------------- UI -----------------
        outer = QtWidgets.QVBoxLayout(self)

        hdr = QtWidgets.QLabel(f"Data directory: {data_dir}")
        hdr.setStyleSheet("font-weight:600;")
        outer.addWidget(hdr)

        ctl_row = QtWidgets.QHBoxLayout()

        self.patient_combo = QtWidgets.QComboBox()
        self.patient_combo.setMinimumWidth(260)

        self.texture_btn = QtWidgets.QPushButton("Texture: OFF")
        self.clear_sel_btn = QtWidgets.QPushButton("Clear selection")

        self.texture_btn.clicked.connect(self._toggle_texture_mode)
        self.clear_sel_btn.clicked.connect(self._clear_selection_clicked)
        self.patient_combo.currentIndexChanged.connect(self._on_patient_combo_changed)

        self.gender_btn = QtWidgets.QPushButton("")
        self.gender_btn.clicked.connect(self._toggle_gender)

        self.bmi_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bmi_slider.setRange(1, 5)
        self.bmi_slider.setTickInterval(1)
        self.bmi_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.bmi_slider.valueChanged.connect(self._on_bmi_changed)
        self.bmi_label = QtWidgets.QLabel("BMI scale: 3")
        self.bmi_label.setMinimumWidth(110)

        ctl_row.addWidget(QtWidgets.QLabel("Patient:"))
        ctl_row.addWidget(self.patient_combo)
        ctl_row.addSpacing(12)
        ctl_row.addWidget(self.texture_btn)
        ctl_row.addWidget(self.clear_sel_btn)
        ctl_row.addStretch(1)
        ctl_row.addWidget(self.gender_btn)
        ctl_row.addWidget(self.bmi_label)
        ctl_row.addWidget(self.bmi_slider)

        outer.addLayout(ctl_row)

        # --- three columns ---
        row = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.montage_panel = MontagePanel()
        self.montage_panel.image_clicked.connect(self._on_montage_image_clicked)

        self.plotter = QtInteractor(self, auto_update=True)
        self.plotter.set_background("black")
        self.plotter.enable_anti_aliasing()
        self.plotter.enable_trackball_style()

        self.image_panel = ImagePanel()
        self.image_panel.image_changed.connect(self._on_image_changed)
        self.image_panel.photo_zoom_delta.connect(self._on_photo_zoom_delta)

        row.addWidget(self.montage_panel)
        row.addWidget(self.plotter.interactor)
        row.addWidget(self.image_panel)

        # 2 | 2 | 3
        row.setStretchFactor(0, 2)
        row.setStretchFactor(1, 2)
        row.setStretchFactor(2, 3)

        outer.addWidget(row, 1)

        # ----------------- Data loading & setup -----------------
        self._load_mesh_and_vertex_index()
        self._load_vertex_to_images_only()
        self._populate_patient_combo()

        self._setup_scene()
        self._setup_lighting()
        self._setup_point_picking()
        self._setup_mesh_zoom_sync()

        self._update_gender_button_text()
        self.bmi_slider.setValue(self.current_bmi_scale)
        self.bmi_label.setText(f"BMI scale: {self.current_bmi_scale}")

        # Default start: first individual patient if exists (combo index 1)
        if self.patient_combo.count() > 1:
            self.patient_combo.setCurrentIndex(1)
        else:
            self.patient_combo.setCurrentIndex(0)

        self.current_vertex = None
        self._show_all_images_for_current_filter()

    # ---------- patient dropdown ----------

    def _populate_patient_combo(self):
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItem("ALL patients")
        for pdir in self.patient_dirs:
            self.patient_combo.addItem(pdir.name)
        self.patient_combo.blockSignals(False)

    def _on_patient_combo_changed(self, idx: int):
        self.current_patient_idx = idx
        if self.current_vertex is None:
            self._show_all_images_for_current_filter()
        else:
            self._refresh_current_vertex_images()

    # ---------- avatar mesh family ----------

    def _collect_avatar_files(self):
        for g in ("m", "f"):
            for i in range(1, 6):
                p = self.assets_dir / f"{g}_{i}.obj"
                if p.is_file():
                    self.avatar_paths[(g, i)] = p
        if not self.avatar_paths:
            raise FileNotFoundError(
                f"No avatar OBJ files found. Expected m_1..5.obj / f_1..5.obj in {self.assets_dir}"
            )
        if ("m", 3) in self.avatar_paths:
            self.current_gender, self.current_bmi_scale = "m", 3
        else:
            (g, i), _ = next(iter(self.avatar_paths.items()))
            self.current_gender, self.current_bmi_scale = g, i

    def _current_avatar_path(self) -> Path:
        key = (self.current_gender, self.current_bmi_scale)
        if key not in self.avatar_paths:
            available = sorted(
                [(g, i) for (g, i) in self.avatar_paths.keys() if g == self.current_gender],
                key=lambda t: abs(t[1] - self.current_bmi_scale)
            )
            if available:
                key = available[0]
            else:
                key = next(iter(self.avatar_paths.keys()))
            self.current_gender, self.current_bmi_scale = key
        return self.avatar_paths[key]

    def _update_gender_button_text(self):
        self.gender_btn.setText(f"Gender: {self.current_gender.upper()}")

    def _reload_mesh_only(self):
        path = self._current_avatar_path()
        new_mesh = pv.read(str(path)).triangulate()
        if not isinstance(new_mesh, pv.PolyData):
            new_mesh = new_mesh.extract_surface().triangulate()

        if self.npts_ref is not None and new_mesh.n_points != self.npts_ref:
            raise RuntimeError(
                f"Avatar mesh {path} has {new_mesh.n_points} points "
                f"but reference topology has {self.npts_ref}."
            )

        self.mesh = new_mesh
        npts = self.mesh.n_points

        base = np.array([1.0, 0.95, 0.8], dtype=float)
        self.base_colors = np.tile(base, (npts, 1))
        self.mesh.point_data["RGB"] = self.base_colors.copy()

        if self.highlight_actor is not None:
            try:
                self.plotter.remove_actor(self.highlight_actor)
            except Exception:
                pass
            self.highlight_actor = None

        self._setup_scene()
        self._setup_lighting()
        self._update_colors_for_current_state()
        self.plotter.render()

    # ---------- helpers ----------

    @staticmethod
    def _resolve_image_path(img_dir: Path, stem: str) -> Path | None:
        exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
        for ext in exts:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def _compute_image_colors(npts: int,
                              base_colors: np.ndarray,
                              vertex_color_dict: dict) -> np.ndarray:
        colors = base_colors.copy()
        for key, val in vertex_color_dict.items():
            try:
                vidx = int(key)
            except ValueError:
                continue
            if vidx < 0 or vidx >= npts:
                continue
            arr = np.array(val)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                rgb = arr.astype(float)
            else:
                rgb = np.median(arr, axis=0).astype(float)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            colors[vidx, :] = rgb
        return colors

    # ---------- loading ----------

    def _load_mesh_and_vertex_index(self):
        path = self._current_avatar_path()
        self.mesh = pv.read(str(path)).triangulate()
        if not isinstance(self.mesh, pv.PolyData):
            self.mesh = self.mesh.extract_surface().triangulate()
        npts = self.mesh.n_points
        self.npts_ref = npts

        base = np.array([1.0, 0.95, 0.8], dtype=float)
        self.base_colors = np.tile(base, (npts, 1))
        self.mesh.point_data["RGB"] = self.base_colors.copy()

        self.patient_dirs: list[Path] = []
        for child in sorted(self.data_dir.iterdir()):
            if not child.is_dir():
                continue
            img_dir = child / cfg.UNMARKED_IMAGES_DIR
            parts_path = child / cfg.MAPPING_OUTPUT_DIR / "vertex_parts.json"
            if img_dir.is_dir() and parts_path.is_file():
                self.patient_dirs.append(child)

        self.one_based = False

    def _load_vertex_to_images_only(self):
        npts = self.mesh.n_points
        v2imgs: dict[int, list[Path]] = {}

        all_imgs_set: set[Path] = set()
        imgs_by_pdir: dict[Path, set[Path]] = {}

        for pdir in self.patient_dirs:
            img_dir = pdir / cfg.UNMARKED_IMAGES_DIR
            parts_path = pdir / cfg.MAPPING_OUTPUT_DIR / "vertex_parts.json"
            try:
                with open(parts_path, "r") as f:
                    corr = json.load(f)
            except Exception:
                continue

            for stem, verts in corr.items():
                img_path = self._resolve_image_path(img_dir, stem)
                if img_path is None:
                    continue

                all_imgs_set.add(img_path)
                imgs_by_pdir.setdefault(pdir, set()).add(img_path)

                count = len(verts)
                if img_path not in self.image_vertex_count:
                    self.image_vertex_count[img_path] = count
                else:
                    self.image_vertex_count[img_path] = min(
                        self.image_vertex_count[img_path], count
                    )

                for vi in verts:
                    vi_int = int(vi)
                    if 0 <= vi_int < npts:
                        v2imgs.setdefault(vi_int, []).append(img_path)

        self.vertex_to_images = v2imgs
        self.all_images = sorted(all_imgs_set, key=lambda p: (p.parent.name, p.name))
        self.images_by_patient = {
            pdir: sorted(imgs, key=lambda p: p.name)
            for pdir, imgs in imgs_by_pdir.items()
        }

    # ---------- filter / toggle ----------

    def _current_filter_all_images(self) -> list[Path]:
        # combo index 0 = ALL
        if self.current_patient_idx <= 0:
            return self.all_images
        pdir = self.patient_dirs[self.current_patient_idx - 1]
        return self.images_by_patient.get(pdir, [])

    def _show_all_images_for_current_filter(self):
        imgs = self._current_filter_all_images()
        total = len(imgs)
        if total:
            self.image_panel.set_images(imgs, global_total=total, select_first=False)
            self.montage_panel.set_images(imgs)
            self._update_colors_for_current_state()
            self._hud_text(f"No vertex selected • {total} images (no photo selected)")
        else:
            self.image_panel.clear("No images available.")
            self.montage_panel.set_images([])
            self._reset_to_base_colors()
            self._hud_text("No images available for this filter")

    def _toggle_texture_mode(self):
        self.texture_mode_on = not self.texture_mode_on
        self.texture_btn.setText(f"Texture: {'ON' if self.texture_mode_on else 'OFF'}")
        self._update_colors_for_current_state()

    def _clear_selection_clicked(self):
        self.current_vertex = None
        self._clear_highlight()
        self._show_all_images_for_current_filter()
        self._hud_text("Selection cleared (no photo selected)")
        self.plotter.render()

    def _toggle_gender(self):
        self.current_gender = "f" if self.current_gender == "m" else "m"
        self._update_gender_button_text()
        self._reload_mesh_only()

    def _on_bmi_changed(self, value: int):
        self.current_bmi_scale = int(value)
        self.bmi_label.setText(f"BMI scale: {self.current_bmi_scale}")
        self._reload_mesh_only()

    # ---------- visualization ----------

    def _setup_scene(self):
        self.plotter.clear()
        scalars = self.mesh.point_data.get("RGB", None)
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars=scalars,
            rgb=True if scalars is not None else False,
            smooth_shading=True,
            show_edges=False,
            opacity=1.0,
            specular=0.25,
            diffuse=0.7,
            ambient=0.15,
            pickable=True,
        )
        self.plotter.add_axes(interactive=True)

        # Requested initial camera
        self.plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
        self.plotter.render()

    def _setup_lighting(self):
        try:
            self.plotter.remove_all_lights()
        except Exception:
            pass

        light = pv.Light()
        if hasattr(light, "set_camera_light"):
            light.set_camera_light()
        else:
            try:
                light.light_type = "camera_light"
            except Exception:
                light.position = (0.0, 0.0, 1.0)
                light.focal_point = (0.0, 0.0, 0.0)

        try:
            light.position = (0.0, 0.4, 1.0)
        except Exception:
            pass

        light.intensity = 1.0
        self.plotter.add_light(light)

    def _setup_point_picking(self):
        def on_pick(point, *args):
            if point is None or not np.all(np.isfinite(point)):
                self.current_vertex = None
                self._clear_highlight()
                self._show_all_images_for_current_filter()
                self._hud_text("Selection cleared")
                self.plotter.render()
                return

            try:
                vi = int(self.mesh.find_closest_point(point))
            except Exception:
                return

            self.current_vertex = vi
            self._update_for_vertex(vi)
            self.plotter.render()

        self.plotter.enable_point_picking(
            callback=on_pick,
            show_message=False,
            left_clicking=True,
            tolerance=0.002,
        )

    def _setup_mesh_zoom_sync(self):
        iren = self.plotter.iren

        def wheel_forward(obj, ev):
            self._apply_photo_zoom(1.1)

        def wheel_backward(obj, ev):
            self._apply_photo_zoom(0.9)

        try:
            iren.add_observer("MouseWheelForwardEvent", wheel_forward)
            iren.add_observer("MouseWheelBackwardEvent", wheel_backward)
        except Exception:
            pass

    def _hud_text(self, text):
        try:
            self.plotter.add_text(
                text,
                font_size=10,
                color="white",
                position="upper_left",
                name="hud_text",
            )
        except Exception:
            pass

    def _clear_highlight(self):
        if self.highlight_actor is not None:
            try:
                self.plotter.remove_actor(self.highlight_actor)
            except Exception:
                pass
            self.highlight_actor = None

    def _reset_to_base_colors(self):
        if self.base_colors is not None:
            self.mesh.point_data["RGB"] = self.base_colors.copy()
            self.mesh.set_active_scalars("RGB")
            self.mesh.Modified()
            self.plotter.render()

    def _highlight_vertex(self, vi: int):
        if self.highlight_actor is not None:
            try:
                self.plotter.remove_actor(self.highlight_actor)
            except Exception:
                pass
            self.highlight_actor = None

        xyz = self.mesh.points[vi]
        radius = max(self.mesh.length / 200.0, 1e-3)
        sphere = pv.Sphere(radius=radius, center=xyz, theta_resolution=24, phi_resolution=24)
        self.highlight_actor = self.plotter.add_mesh(sphere, color="red", smooth_shading=True)

    def _get_patient_dir_for_image(self, img_path: Path) -> Path | None:
        try:
            return img_path.parent.parent
        except Exception:
            return None

    def _filtered_images_for_vertex(self, vi: int) -> list[Path]:
        imgs = self.vertex_to_images.get(vi, [])
        if self.current_patient_idx <= 0:
            return imgs
        target_pdir = self.patient_dirs[self.current_patient_idx - 1]
        return [p for p in imgs if self._get_patient_dir_for_image(p) == target_pdir]

    def _update_for_vertex(self, vi: int):
        self._highlight_vertex(vi)

        global_total = len(self._current_filter_all_images())

        imgs = self._filtered_images_for_vertex(vi)
        imgs_sorted = sorted(imgs, key=lambda p: self.image_vertex_count.get(p, 1_000_000))

        self._hud_text(f"Vertex {vi} • {len(imgs_sorted)} images (subset of {global_total})")

        if imgs_sorted:
            self.image_panel.set_images(imgs_sorted, global_total=global_total, select_first=True)
            self.montage_panel.set_images(imgs_sorted)
            self._update_colors_for_current_state()
        else:
            self.current_vertex = None
            self._show_all_images_for_current_filter()

    def _refresh_current_vertex_images(self):
        if self.current_vertex is None:
            self._show_all_images_for_current_filter()
        else:
            self._update_for_vertex(self.current_vertex)

    # ---------- texture/color logic ----------

    def _get_image_colors(self, img_path: Path) -> np.ndarray | None:
        if img_path in self.image_colors_cache:
            return self.image_colors_cache[img_path]

        pdir = self._get_patient_dir_for_image(img_path)
        if pdir is None:
            return None

        if pdir not in self.patient_rgb_cache:
            rgb_path = pdir / cfg.MAPPING_OUTPUT_DIR / "vertex_rgb.json"
            if rgb_path.is_file():
                try:
                    with open(rgb_path, "r") as f:
                        self.patient_rgb_cache[pdir] = json.load(f)
                except Exception:
                    self.patient_rgb_cache[pdir] = None
            else:
                self.patient_rgb_cache[pdir] = None

        rgb_data = self.patient_rgb_cache.get(pdir)
        if not rgb_data:
            return None

        stem = img_path.stem
        if stem not in rgb_data:
            return None

        vertex_color_dict = rgb_data[stem]
        npts = self.mesh.n_points
        colors = self._compute_image_colors(npts, self.base_colors, vertex_color_dict)
        self.image_colors_cache[img_path] = colors
        return colors

    def _get_patient_aggregate_colors(self, pdir: Path) -> np.ndarray | None:
        if pdir in self.patient_aggregate_colors:
            return self.patient_aggregate_colors[pdir]

        if pdir not in self.patient_rgb_cache:
            rgb_path = pdir / cfg.MAPPING_OUTPUT_DIR / "vertex_rgb.json"
            if rgb_path.is_file():
                try:
                    with open(rgb_path, "r") as f:
                        self.patient_rgb_cache[pdir] = json.load(f)
                except Exception:
                    self.patient_rgb_cache[pdir] = None
            else:
                self.patient_rgb_cache[pdir] = None

        rgb_data = self.patient_rgb_cache.get(pdir)
        if not rgb_data:
            return None

        npts = self.mesh.n_points
        accum: dict[int, list[np.ndarray]] = {}
        for _, vertex_color_dict in rgb_data.items():
            for key, val in vertex_color_dict.items():
                try:
                    vidx = int(key)
                except ValueError:
                    continue
                if 0 <= vidx < npts:
                    arr = np.asarray(val, dtype=float)
                    if arr.size == 0:
                        continue
                    if arr.ndim == 1:
                        arr = arr.reshape(1, 3)
                    accum.setdefault(vidx, []).append(arr)

        colors = self.base_colors.copy()
        for vidx, arr_list in accum.items():
            arr_all = np.concatenate(arr_list, axis=0)
            rgb = np.median(arr_all, axis=0)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            colors[vidx, :] = rgb

        self.patient_aggregate_colors[pdir] = colors
        return colors

    def _update_colors_for_current_state(self):
        img_path = self.image_panel.current_image()

        if self.texture_mode_on:
            if self.current_patient_idx <= 0:
                self._reset_to_base_colors()
                return
            pdir = self.patient_dirs[self.current_patient_idx - 1]
            colors = self._get_patient_aggregate_colors(pdir)
            if colors is None:
                self._reset_to_base_colors()
                return
        else:
            if img_path is None:
                self._reset_to_base_colors()
                return
            colors = self._get_image_colors(img_path)
            if colors is None:
                self._reset_to_base_colors()
                return

        self.mesh.point_data["RGB"] = colors
        self.mesh.set_active_scalars("RGB")
        self.mesh.Modified()
        self.plotter.render()

    # ---------- zoom sync helpers ----------

    def _apply_photo_zoom(self, factor: float):
        try:
            self.image_panel.img_label.apply_external_zoom(factor)
        except Exception:
            pass

    @QtCore.Slot(float)
    def _on_photo_zoom_delta(self, factor: float):
        cam = self.plotter.camera
        if cam is None:
            return
        try:
            cam.Zoom(factor)
        except Exception:
            return
        self.plotter.render()

    # ---------- slots ----------

    @QtCore.Slot(Path)
    def _on_image_changed(self, img_path: Path):
        self._update_colors_for_current_state()

    @QtCore.Slot(Path)
    def _on_montage_image_clicked(self, img_path: Path):
        self.image_panel.set_current_image(img_path)


# --------------- Main Window ---------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_dir: Path, assets_dir: Path):
        super().__init__()
        self.setWindowTitle("Mesh ↔ Image Correspondence Viewer")
        self.resize(1700, 900)

        viewer = PatientViewer(data_dir, assets_dir)
        self.setCentralWidget(viewer)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for DensePose vertex ↔ image correspondence")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Patient data directory (default: {cfg.DATA_DIR})")
    parser.add_argument("--assets-dir", type=str, default=None,
                        help=f"Assets directory containing avatar OBJ files (default: {cfg.ASSETS_DIR})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else cfg.DATA_DIR
    assets_dir = Path(args.assets_dir) if args.assets_dir else cfg.ASSETS_DIR

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(data_dir, assets_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
