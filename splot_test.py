#!/usr/bin/env python3

from data.splot import SplotData
import os

if __name__ == "__main__":

    MY_DIR = os.path.dirname(os.path.realpath(__file__))

    # for.
    d = SplotData(os.path.join(MY_DIR,"data","splot"), 0.3)
    for n, clauses in d.train_generator():
        print(n)
        print(clauses)

