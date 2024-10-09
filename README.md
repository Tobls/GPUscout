# GPUscout

A tool for discovering data movement-related bottlenecks on NVidia GPUs.

!!! GPUscout is in active development, and is not yet in a production-ready stability !!!

GPUscout is a tool for systematical detection of the root cause of frequent memory performance bottlenecks on NVIDIA GPUs.
It connects three approaches to analysing performance -- static CUDA SASS code analysis, sampling warp stalls, and kernel performance metrics.
Connecting these approaches, GPUscout can identify the problem, locate the code segment where it originates, and assess its importance.

## Requirements

- cmake (3.27+)
- [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (11.8+)
  - [Nvidia Nsight Compute](https://developer.nvidia.com/nsight-compute) should also be automatically installed as a part of it.

## Installation

GPUscout can be installed with cmake; spack package is coming in the near future.

```bash
#mkdir executable
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../inst-dir ..
make all install
```

Note that this tool has been tested with 11.8 on NVIDIA Volta and Turing architectures.

## Running an analysis

### Generate the executable and cubin file

Generate two executables using the `nvcc` compiler:

- Executable generated by normal compilation, say `<executable_name>`. (`nvcc pr.cu -o pr`)
- Executable generated by using the `-cubin` flag with the `nvcc` compiler. If the executable name is prefixed with `cubin-`, i.e. `cubin-<executable_name>` (`nvcc pr.cu -cubin -o cubin-pr`), and is in the same folder as the regular executable, the cubin file path does not need to be entered later on.

### Run the GPUscout script

Run the GPUscout.sh script, which was installed to the defined install directory. Specify the executable to analyze (-e executable) and potentially other parameters:

```bash
./GPUscout -e ../executable/gaussian -a '-q -s 2000'
```

The following input arguments and syntax are supported:

```bash
Usage: GPUscout [-h] [--dry-run] [--verbose] -e executable [-c directory] [--args]"
    -h | --help : Display this help.
    --dry_run : performs only dry_run. A --dry_run will only analyse the SASS instructions. --dry_run will neither read warp stalls nor Nsight metrics
    -v | --verbose : print more verbose output.
    -e | --executable : Path to the executable (compiled with nvcc).
    -c | --cubin : Path to the cubin file (compiled with nvcc, with -cubin). If left empty, the same path as executable and the name cubin-<executable> will be assumed.
    -a | --args : Arguments for running the binary. e.g. --args=\"64 2 2 temp_64 power_64 output_64.txt\"
    -j | --json : Save a JSON-formatted version of the output (Needed for the use of GPUscout-GUI)
```

This should automatically start analysing the code and printing recommendations on the terminal screen.

For older NVIDIA architectures (like Pascal), a dry run option has been provided that reports based on SASS instructions only. This can be run as `GPUscout --dry_run ..... `.

## About

GPUscout has been initially developed by Soumya Sen, and is further maintained by Stepan Vanecek (stepan.vanecek@tum.de) and the [CAPS TUM](https://www.ce.cit.tum.de/en/caps/homepage/). Please contact us in case of questions, bug reporting etc.

GPUscout is available under the Apache-2.0 license. (see [License](https://github.com/caps-tum/sys-sage/blob/master/LICENSE))
