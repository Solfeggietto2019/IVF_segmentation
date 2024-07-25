from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Any, Union
import json
import numpy as np

class LogicStatus:
    def __init__(self):
        self.reset()

    def reset(self):
        self.egg_class = 4
        self.pipette_class = 5
        self.pipette_detected_frame = -1
        self.cooldown_frames = 30
        self.fps = 30
        self.save_frame_delay = 3 * self.fps
        self.last_collision_frame = -self.cooldown_frames
        self.frame_saved = False
        self.paused = False
        self.show_options = False
        self.manually_egg_frame_saved = False

@dataclass
class Sequence:
    sequence_ID: int
    initial_frame: int
    final_frame: int
    sperms_in_sequence: list
    number_of_analyzed_sperms: int
    rank_A_id: int
    rank_B_id: int
    rank_C_id: int


@dataclass
class Sperm:
    id: int
    initial_frame: int
    final_frame: int
    positions: List[Tuple[int, float, float]]
    deleted: bool
    SiDScore: float
    confidence: int
    mean_brightness: List[int]
    motility_parameters: Dict[str, float]
    standard_motility_parameters: Dict[str, float]


@dataclass
class SelectedSperm:
    Id: str = None
    motility_parameters: Dict[str, float] = None
    morphological_parameters: Dict[str, float] = None
    standardize_morph_parameters: Dict[str, float] = None  # MANDAR
    mask: Union[List[Any], np.ndarray] = field(default_factory=list)  # MARK
    bbox: Union[List[Any], np.ndarray] = field(default_factory=list)
    frame: Any = None
    sid_score: int = None  # MANDAR
    initial_frame: int = None  # MANDAR
    b64_string_frame: str = None  # Mandar

    def to_serializable(self):
        return {
            "Id": self.Id,
            "motility_parameters": self.motility_parameters,
            "morphological_parameters": self.morphological_parameters,
            "standardize_morph_parameters": self.standardize_morph_parameters,
            "mask": (
                self.mask.tolist() if isinstance(self.mask, np.ndarray) else self.mask
            ),
            "bbox": (
                self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox
            ),
            "frame": self.frame,
            "sid_score": self.sid_score,
            "initial_frame": self.initial_frame,
            "b64_string_frame": self.b64_string_frame,
        }


@dataclass
class Egg:
    frame_number: int
    mask: Any
    egg_features: Any
    b64_string_frame: str = None  # Mandar
    score: float = None


@dataclass
class VersionControl:
    ApiVersion: str = "1.0"
    AnalyzedVideo: str = "video.mp4"
    Fecha: str = "2024-06-26"


@dataclass
class DataStructure:
    objectID: str
    VersionControl: VersionControl
    SiD: List[SelectedSperm]
    Aeris: List[Egg]
    Sofi: List

    def to_json(self):
        return json.dumps(asdict(self), indent=4)

    def to_dict(self):
        return asdict(self)
