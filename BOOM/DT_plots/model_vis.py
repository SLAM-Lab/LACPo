import argparse
import pathlib
import subprocess

import sklearn.tree
from sklearn.externals import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', help='path to folder of models', type=pathlib.Path)
    args = parser.parse_args()
    power_models: pathlib.Path = args.models

    for model in power_models.iterdir():
        dt_m = joblib.load(model)
        try:
            script = next(pathlib.Path('./invoke_model_shell_files').glob(f'{model.stem.strip("_1")}*.sh'))
            features = [f.split('.')[-1].split('[')[0] for f in script.read_text().split('-fl')[1].split(' ')[1].split(',')]
        except StopIteration:
            print(f'{model.stem} has no feature infos')
            features = None

        base = pathlib.Path('output') / model.stem
        base.parent.mkdir(exist_ok=True)
        with open(base.with_suffix('.dot'), 'w') as f:
            sklearn.tree.export_graphviz(dt_m, out_file=f, filled=True, rotate=True, feature_names=features)
        subprocess.run(f'dot -Tpdf {f.name} -o {base.with_suffix(".pdf")}'.split())
