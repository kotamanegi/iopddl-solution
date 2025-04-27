## ASPLOS solution

This is a 4th-place solution for [**The ASPLOS 2025 / EuroSys 2025 Contest on Intra-Operator Parallelism for Distributed Deep Learning**](https://github.com/asplos-contest/2025/blob/main/IOPDDL.md).
The original boilerplate repository is [available here](https://github.com/google/iopddl).

We use [AtCoder library](https://github.com/atcoder/ac-library) located in atcoderlib.hpp as a external library. AtCoder library is currently available under CC0 1.0 Universal license.

-----

Supplemental materials (i.e., utilities and benchmarks) for [**The ASPLOS 2025 / EuroSys 2025 Contest on Intra-Operator Parallelism for Distributed Deep Learning**](https://github.com/asplos-contest/2025/blob/main/IOPDDL.md).

## Example Usage

```
$ git clone --recursive https://github.com/kotamanegi/iopddl-solution.git
$ mkdir iopddl/build && cd iopddl/build && cmake .. && make
$ ./iopddl example.json 10
```

## Decompressing Benchmarks

In order to try out the contest problems, you'll first need to run this command one time after fetching benchmarks from [here](https://github.com/google/iopddl):

```
$ gzip -d benchmarks/*
```
