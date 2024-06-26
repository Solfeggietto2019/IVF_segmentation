from dataclasses import dataclass

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
    positions: list
    number_of_positions: list
    stats: dict


class SelectedSperm:
    def __init__(self):
        self.Id = ""
        self.Data = {}
        self.mask = []
        self.bbox = []
        self.frame = None