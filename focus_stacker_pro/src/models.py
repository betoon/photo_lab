from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class AlignmentOptions:
    method: str = "ecc_affine"
    reference: str = "middle"
    crop_common: bool = True
    max_dimension: int = 2400
    ecc_iterations: int = 150
    ecc_epsilon: float = 1e-6
    multiscale: bool = True
    pyramid_scales: int = 3


@dataclass
class StackOptions:
    algorithm: str = "depth_map"
    focus_radius: int = 5
    smooth_radius: int = 7
    temperature: float = 8.0
    pyramid_levels: int = 5
    cleanup_radius: int = 5
    ghost_suppression: float = 0.0
    sharpen: float = 0.25
    denoise: float = 0.0
    normalize_exposure: bool = False
    normalize_color: bool = False


@dataclass
class PerformanceOptions:
    tile_size: int = 1024
    tiled_fusion: bool = True
    disk_cache: bool = True
    cache_directory: str = ""
    use_gpu: bool = True
    cpu_threads: int = 0
    recover_failed_frames: bool = True
    alignment_proxy_dimension: int = 1800


@dataclass
class OutputOptions:
    preset: str = "archival"
    bit_depth: int = 16
    grayscale: bool = False
    include_alpha: bool = False
    preserve_icc: bool = True
    preserve_dpi: bool = True
    resize_percent: float = 100.0
    export_aligned: bool = False
    export_masks: bool = False
    export_depth: bool = False
    export_confidence: bool = False
    bigtiff: bool = True


@dataclass
class MicroscopeOptions:
    enabled: bool = False
    illumination_normalization: bool = True
    background_sigma: float = 45.0
    hot_pixel_cleanup: bool = True
    hot_pixel_strength: float = 2.5
    contrast_boost: float = 0.0
    preserve_brightness: bool = True
    flat_field_path: str = ""
    dark_frame_path: str = ""
    focus_scale_mode: str = "smart"
    fine_radius: int = 2
    medium_radius: int = 5
    coarse_radius: int = 11
    minimum_structure: float = 0.08
    minimum_confidence: float = 0.12
    uncertain_mode: str = "average"
    patch_morphology: int = 0
    depth_preference: float = 0.0
    color_selective: bool = False
    target_color: list[int] = field(default_factory=lambda: [255, 0, 255])
    color_tolerance: float = 25.0
    color_space: str = "Lab"
    color_focus_mix: float = 0.5
    synthesize_intermediate: bool = False
    intermediate_count: int = 1
    scientific_mode: bool = True
    microns_per_pixel: float = 0.0
    scale_bar_microns: float = 100.0
    scale_bar_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    scale_bar_position: str = "bottom-right"
    scale_bar_enabled: bool = False


@dataclass
class Project:
    format_version: int = 3
    images: list[str] = field(default_factory=list)
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    stack: StackOptions = field(default_factory=StackOptions)
    microscope: MicroscopeOptions = field(default_factory=MicroscopeOptions)
    performance: PerformanceOptions = field(default_factory=PerformanceOptions)
    output: OutputOptions = field(default_factory=OutputOptions)
    output_path: str = ""
    disabled_images: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(format_version=data.get("format_version", 1), images=data.get("images", []),
                   alignment=AlignmentOptions(**data.get("alignment", {})),
                   stack=StackOptions(**data.get("stack", {})),
                   microscope=MicroscopeOptions(**data.get("microscope", {})),
                   performance=PerformanceOptions(**data.get("performance", {})),
                   output=OutputOptions(**data.get("output", {})),
                   output_path=data.get("output_path", ""),
                   disabled_images=data.get("disabled_images", []))
