# 1. Install. Official Sapiens2 runtime requires Python >=3.12, PyTorch >=2.7.
import sys, subprocess, importlib.metadata as metadata
assert sys.version_info >= (3,12), 'Use a Colab Python 3.12+ runtime'
subprocess.run([sys.executable,'-m','pip','install','-q','mediapipe','ultralytics','huggingface_hub','safetensors','packaging','pandas','matplotlib','opencv-python-headless'],check=True)
from pathlib import Path
import torch
from packaging.version import Version
assert Version(torch.__version__.split('+')[0])>=Version('2.7'), 'Use PyTorch >=2.7'
assert torch.cuda.is_available(), 'Select a GPU runtime'
if not Path('/content/sapiens2').exists():
    subprocess.run(['git','clone','https://github.com/facebookresearch/sapiens2.git','/content/sapiens2'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-e','/content/sapiens2'],check=True)
print('Torch:',torch.__version__,'Ultralytics:',metadata.version('ultralytics'))
print('Sapiens2 commit:',subprocess.check_output(['git','-C','/content/sapiens2','rev-parse','HEAD'],text=True).strip())
