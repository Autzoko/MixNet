import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trainer.hybridunet_trainer import hybridunet_trainer

if __name__ == '__main__':
    hybridunet_trainer()