from dataclasses import dataclass
from typing import List, Tuple, Dict

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


class SelectedSperm:
    def __init__(self):
        self.Id = ""
        self.Data = {}
        self.mask = []
        self.bbox = []
        self.frame = None