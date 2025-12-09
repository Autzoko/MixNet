from dataclasses import dataclass
from typing import Optional, List

import json
from pathlib import Path

@dataclass
class UltrasoundSample:
    id: str
    image_path: str
    mask_path: Optional[str] = None
    label: Optional[int] = None
    domain: Optional[str] = None

def load_ultrasound_metadata(json_path: str) -> List[UltrasoundSample]:
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = UltrasoundSample(
            id=item.get('id'),
            image_path=item.get('image_path'),
            mask_path=item.get('mask_path', None),
            label=item.get('label', None),
            domain=item.get('domain', None)
        )
        samples.append(sample)

    return samples