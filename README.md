# GPUscout

A tool for discovering data movement-related bottlenecks on NVidia GPUs.

!!! GPUscout is in active development, and is not yet in a production-ready stability !!!

## Description

//TODO

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

//TODO
`CUPTI` and `libpcsampling` needs to be added to the respective `LD_LIBRARY_PATH`s.

Note that this tool has been tested with 11.8 on NVIDIA Volta and Turing architectures.


## Running an analysis

### Generate the executable and cubin file

Inside the `executable` directory, generate two executables using the `nvcc` compiler:
- Executable generated by normal compilation, say `<executable_name>`. (`nvcc pr.cu -o pr`)
- Executable generated by using the `-cubin` flag with the `nvcc` compiler. The name of this executable should be prefixed with `cubin-`, i.e. `cubin-<executable_name>`. (`nvcc pr.cu -cubin -o cubin-pr`)

Copy both the above generated executables in the `executable` subdirectory.

To start using this tool, add the name of your executable in the `setup.sh` file as `file_name=<executable_name>`. Run the `setup.sh` script as `./setup.sh`.

This should automatically start analysing the code and printing recommendations on the terminal screen.

For older NVIDIA architectures (like Pascal), a dry run option has been provided that reports based on SASS instructions only. This can be 
run as `setup.sh --dry_run`.


## About
GPUscout has been initially developed by Soumya Sen, and is further maintained by Stepan Vanecek (stepan.vanecek@tum.de) and the [CAPS TUM](https://www.ce.cit.tum.de/en/caps/homepage/). Please contact us in case of questions, bug reporting etc.

GPUscout is available under the Apache-2.0 license. (see [License](https://github.com/caps-tum/sys-sage/blob/master/LICENSE))
