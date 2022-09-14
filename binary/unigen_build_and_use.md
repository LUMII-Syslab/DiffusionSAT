# How to Build and Use Unigen

###### by Sergejs Kozlovičs, 2022-09-13

## About Unigen

Unigen is a uniform solution sampler for CNF. Its homepage is https://github.com/meelgroup/unigen.

## Unigen Usage

Unigen takes a .cnf file in the DIMACS format as an input. 

A special comment line, e.g.,  `c ind 3 4 7 8 10 11 14 17 18 26 30 35 36 39 42 47 60 62 67 0`, can be prepended to the input file to denote which SAT variables must be sampled (if not all of them are needed).

Invocation example:

```bash
unigen --samples 10 --arjun 0 myfile1.cnf  >out1.cnf
```

The number of samples $n$ by default is 500; in our example we chose 10.

Unigen tries to minimize the number of independent variables by default. Thus, we use the `--arjun 0` args to skip this optimization, and force Unigen to output values for all variables we are interested in.

Unigen outputs a lot of comment lines (starting with `c`), the line containing sampled variables (`vp 10 8 7 6 4 3 1 0`), and $\geq n$ samples. Thus, we need to take the last $n$ non-`c` and non-`vp` lines. Each such line contains one sample: a list of positive and negative integers ending with 0. Positive and negative integers correspond to `true` and `false` values of the corresponding CNF variables.

## Building Unigen Static Binary

The following commands need to be executed in order to build and install the static binary of Unigen on Ubuntu (tested on Ubuntu 22.04 LTS).

The static binary (independent on other libs) can be then taken from `/usr/local/bin/unigen` (31 MiB).

```bash
sudo apt-get install build-essential cmake
sudo apt-get install zlib1g-dev libboost-program-options-dev libm4ri-dev
sudo apt install libgmp-dev

mkdir -p ~/unigen.gits
cd ~/unigen.gits

git clone https://github.com/msoos/cryptominisat
cd cryptominisat
mkdir build && cd build
cmake -DSTATICCOMPILE=ON -DUSE_GAUSS=ON ..
make
sudo make install

echo # arjun needed for approxmc
cd ../..
git clone https://github.com/meelgroup/arjun
cd arjun
mkdir build && cd build
cmake -DSTATICCOMPILE=ON ..
make
sudo make install

cd ../..
git clone https://github.com/meelgroup/approxmc/
cd approxmc
mkdir build && cd build
cmake -DSTATICCOMPILE=ON ..
make
sudo make install

cd ../..
git clone https://github.com/meelgroup/unigen/
cd unigen
mkdir build && cd build
cmake -DSTATICCOMPILE=ON ..
make
sudo make install
```