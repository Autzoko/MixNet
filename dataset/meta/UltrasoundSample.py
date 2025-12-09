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
    json_dir = Path(json_path)

    if not json_dir.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_dir}")
    
    with open(json_dir, 'r') as f:
        data = json.load(f)

    data_root = json_dir.parent

    samples = []
    for item in data:
        image_path = item.get('image_path')
        if image_path:
            img_path = Path(image_path)
            if not img_path.is_absolute():
                image_path = str(data_root / image_path)

        mask_path = item.get('mask_path', None)
        if mask_path:
            msk_path = Path(mask_path)
            if not msk_path.is_absolute():
                mask_path = str(data_root / mask_path)

        sample = UltrasoundSample(
            id=item.get('id'),
            image_path=image_path,
            mask_path=mask_path,
            label=item.get('label', None),
            domain=item.get('domain', None)
        )
        samples.append(sample)

    return samples