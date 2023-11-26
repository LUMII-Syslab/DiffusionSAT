# How to Build QuickSampler

###### by Sergejs Kozlovičs, 2023-11-13

Building quicksamler static executable on Ubuntu.

Clone z3 and quicksampler gits, install prerequisites:

```bash
git clone https://github.com/Z3Prover/z3/
git clone https://github.com/RafaelTupynamba/quicksampler.git

sudo apt install git g++ make python3
```

Compile static z3 library (with some modifications for quicksampler):

```bash
cd z3
git checkout 9635ddd8fceb6bdde7dc7725e696e6c123af22f4
cp ../quicksampler/check/sat_params.pyg src/sat/sat_params.pyg
cp ../quicksampler/check/dimacs.cpp src/sat/dimacs.cpp
cp ../quicksampler/check/dimacs_frontend.cpp src/shell/dimacs_frontend.cpp
python scripts/mk_make.py --staticlib
cd build
make
sudo make install
cd ..\..
```

Compile static quicksampler executable:

```bash
cd quicksamler
g++ -g -static -fopenmp -std=c++11 -O3 -o quicksampler quicksampler.cpp ../z3/build/libz3.a
```

For compiling dynamic quicksampler:

```bash
cd quicksamler
sudo apt install z3 libz3-dev
make
```

For using quicksampler:

```
./quicksampler -n <samples> <file.dimacs>
```
