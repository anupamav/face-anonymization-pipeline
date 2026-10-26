"""Run from a Colab cell: from run_pipeline import run; result = run().

Stage scripts intentionally share one namespace: this preserves model debug
outputs, the interactive Colab canvas, and the original notebook stage order.
"""
from pathlib import Path
import runpy

PROJECT_DIR=Path(__file__).resolve().parent

def run(**overrides):
    settings=runpy.run_path(str(PROJECT_DIR/'config.py'))
    settings={k:v for k,v in settings.items() if k.isupper()}
    unknown=set(overrides)-set(settings)
    if unknown:raise ValueError(f'Unknown configuration: {sorted(unknown)}')
    settings.update(overrides)
    namespace={'__name__':'child_height_pipeline_runtime',**settings}
    for file in sorted((PROJECT_DIR/'stages').glob('*.py')):
        print(f'\n--- {file.name} ---')
        exec(compile(file.read_text(),str(file),'exec'),namespace)
    print('\nOUTPUT FOLDER:',namespace['OUT'].resolve())
    print('See README.md: Where Colab saves the displayed outputs')
    return namespace['result']

if __name__=='__main__':
    raise SystemExit('In Colab, import run() and call it in a cell; the manual canvas needs notebook output.')
