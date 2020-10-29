"""
Customized version of generators available in NeuroSAT publication & code
"""
import shutil
from pathlib import Path

from dimacs_to_data import dimacs_to_data
from gen_sr_dimacs import gen_sr_dimacs

N_PAIRS = 1000
MIN_N = 3
MAX_N = 10
MAX_NODES_PAR_BATCH = 5000


def main():
    for mode in ["train", "test"]:
        file_dir = Path(f'data_files/{mode}/sr5')
        if file_dir.exists():
            shutil.rmtree(file_dir)
        file_dir.mkdir(parents=True)
        for i in range(1, 3, 1):
            tmp_dir = file_dir / f'grp{i}'
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            gen_sr_dimacs(str(tmp_dir), N_PAIRS, MIN_N, MAX_N)
            dimacs_to_data(str(tmp_dir), str(file_dir), MAX_NODES_PAR_BATCH)


if __name__ == '__main__':
    main()
