import json
from typing import List, Dict, Any, Union
from utils.dataclasses import Sequence, Sperm
from collections import defaultdict

class JSONReader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_json()

    def _load_json(self) -> Union[List[Dict[str, Any]], None]:
        """
        Load JSON data from the specified file.
        """
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None
        
    def extract_sperms_data(self) -> Union[Dict[str, List[Sperm]], None]:
        """
        Extract relevant data from the 'sperms' key in the JSON.
        """
        if not self.data:
            print("No data loaded")
            return None

        sperms_data = defaultdict(list)
        
        for idx, sperm_data in self.data.get('sperms', {}).items():
            positions = [(pos["frame"], pos["_position"]["x"], pos["_position"]["y"]) for pos in sperm_data["_positions"]]
            initial_frame = positions[0][0] if positions else None
            final_frame = positions[-1][0] if positions else None

            sperm = Sperm(
                id=sperm_data["_id"],
                initial_frame=initial_frame,
                final_frame=final_frame,
                positions=positions,
                deleted=sperm_data["deleted"],
                SiDScore=sperm_data["_SiDScore"],
                confidence=sperm_data["_confidence"],
                mean_brightness=sperm_data["_meanBrightness"],
                motility_parameters=sperm_data["_MotilityParameters"],
                standard_motility_parameters=sperm_data["_StandardMotilityParameters"]
            )
            sperms_data[idx].append(sperm)
        
        return sperms_data
    
    def extract_sperms_data_v2(self) -> Union[Dict[str, List[Sperm]], None]:
        """
        Extract relevant data from the 'sperms' key in the JSON.
        """
        if not self.data:
            print("No data loaded")
            return None
        
        analyzed_sequences = self.data["AnalyzedSequencesInVideo"]
        sequence_objects = [
        Sequence(
            sequence["AnalizedSequenceID"],
            sequence["InitialFrame"],
            sequence["FinalFrame"],
            sequence["SpermsInSequence"],
            sequence["NumberOfAnalyzedSperms"],
            sequence["RankAId"],
            sequence["RankBId"],
            sequence["RankCId"],
        )
        for sequence in analyzed_sequences
        ]

        sperms_data = defaultdict(list)
        for idx_sequence, sequence in enumerate(sequence_objects):
            sperms_data[str(idx_sequence)].extend(
                [
                    Sperm(
                        sperm["id"],
                        sperm["InitialFrame"],
                        sperm["FinalFrame"],
                        sperm["Positions"],
                        sperm["NumberOfPositions"],
                        {
                            "VSl": sperm["VSL"],
                            "VCL": sperm["VCL"],
                            "ROT": sperm["ROT"],
                            "LIN": sperm["LIN"],
                            "VAP": sperm["VAP"],
                            "ALH": sperm["ALH"],
                            "WOB": sperm["WOB"],
                            "STR": sperm["STR"],
                            "BCF": sperm["BCF"],
                            "MAD": sperm["MAD"],
                            "rank": sperm["Rank"],
                        },
                    )
                    for idx_sperm, sperm in enumerate(sequence.sperms_in_sequence)
                ]
            )
        return sperms_data

    def extract_sperms_data_v3(self) -> Union[List[Dict[str, Any]], None]:
        """
        Extract relevant data from the 'sperms' key in the JSON.
        """
        if not self.data:
            print("No data loaded")
            return None

        sperms_data = []
        for item in self.data:
            if "sperms" in item:
                for key, sperm in item["sperms"].items():
                    sperms_data.append({
                        "_id": sperm.get("_id"),
                        "_frames": sperm.get("_frames"),
                        "deleted": sperm.get("deleted"),
                        "_SiDScore": sperm.get("_SiDScore"),
                        "_positions": sperm.get("_positions"),
                        "_confidence": sperm.get("_confidence"),
                        "_meanBrightness": sperm.get("_meanBrightness"),
                        "_MotilityParameters": sperm.get("_MotilityParameters"),
                        "_StandardMotilityParameters": sperm.get("_StandardMotilityParameters"),
                    })
        return sperms_data
