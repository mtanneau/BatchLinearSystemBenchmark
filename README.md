# BatchLinearSystemBenchmark

## Installation instructions

⚠️ All steps below assume you are on a machine equipped with a CUDA device and nsight systems installed.
    To verify whether it is the case, run `nvidia-smi` and `nsys -v`, neither of which should error.

### install python

This is only required to browse benchmark results in an interactive browser app.

1. Install `uv`: see these [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
    

### install julia

This is required to execute/profile any of the batch linear solver benchmarks.
You can skip this step if you only want to browse some pre-existing benchmark results.

1. Install julia: see these [installation instructions](https://julialang.org/install/).
1. Instantiate environment
    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    ```
    This will download and install all required packages.
    
    
    ⚠️ This step may take a few minutes, depending on the quality of your connection and disk I/O

    ⚠️ Make sure to run this on a GPU machine, to ensure that CUDA gets compiled properly

    ⚠️ If you are on a shared cluster with limited space on your home directory, make sure to have your `~/.julia` symlinked to a location on a different drive (e.g. scratch or project space).

### download batch linear systems dataset

TODO


## Quick Start

### Visualizing results

To visualize benchmark results run the command below.
Note that this only requires `uv`, everything else will be installed automatically
```bash
uv run app.py
```

### Benchmarking

To benchmark different batch datasets, linear solvers and batch sizes, use the `src/benchmark.jl` entry point:
```bash
julia --project=. src/benchmark.jl --help
```

For instance, to run a benchmark on the sample randomly-generated dataset of 32 matrices of size 256x256:
```bash
julia -t8 --project=. src/benchmark.jl \
    --dataset=data/sample/rand_256_32_gen_uni.h5 \
    --solver=KLU_Sequential --solver=CUDSS_Batch --solver=CUDSS_UniformBatch \
    --batch-size=8 --batch-size=16 --batch-size=32 \
    --num-solve=1 \
    --benchmark \
    --output-dir benchmark/sample
```

⚠️ Be sure to specify the `-t` flag when calling `julia` to execute KLU on multiple cores

### Profiling

To profile:
```bash
nsys profile julia --project=. -t8 src/benchmark.jl \
    --dataset=data/sample/rand_256_32_gen_uni.h5 \
    --solver=CUDSS_Batch --solver=CUDSS_UniformBatch \
    --batch-size=32 \
    --num-solve=1 \
    --profile
```

⚠️ To keep profiler traces readable, only _a single batch size_ can be specified when using the `--profile` option.


## Reference

### Mathematical background

We consider batch linear systems of the form
$$
    A_{k} x_{k} = b_{k}, \forall k \in \{1, ..., K\}
$$
where $K$ denotes the _batch size_.
A batch is _uniform_ if all matrices $A_{k}$ share the same sparsity pattern.

### Performance metrics

* Execution time: end-to-end time to solve all the linear systems in the batch.
* Numerical accuracy: measured via the norm of the residuals $r_{k} = b_{k} - A_{k} x_{k}^*$,
    where $x_{k}^{*}$ is the solution returned by the linear solver

### Linear solvers

| solver | device | description |
|:-------|:-------|:------------|
| `KLU_Sequential` | CPU | [SuiteSparse KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse) on CPU, parallelized across multiple cores (one KLU instance per julia thread)
| `CUDSS_Sequential` | GPU | [NVIDIA CUDSS](https://developer.nvidia.com/cudss), executed sequentially using a manual `for` loop
| `CUDSS_Batch` | GPU | [NVIDIA CUDSS](https://developer.nvidia.com/cudss), executed in non-uniform batch mode
| `CUDSS_UniformBatch` | GPU | [NVIDIA CUDSS](https://developer.nvidia.com/cudss), executed in uniform batch mode. Only supported for uniform linear systems

### Limitations

* All workflows consider 1 factorization, followed by `num-solve` solves, i.e., the factorization of $A_{k}$ is re-used `num-solve` times.
    By default, benchmarks are executed with `num-solve=1`
* the KLU implementation does not re-use the factorization object when working with uniform batches
* the CUDSS solvers do not leverage matching nor iterative refinement.
    Note that this is a limitation of this code, i.e., these features are implemented in CUDSS

## Acknowledgements

This work was made possible thanks to the support (direct or indirect) of the following institutions and people:
* Michael Klamkin ([@klamike](https://github.com/klamike)) and Andrew Rosemberg ([@andrewrosemberg](https://github.com/andrewrosemberg)),
    who contributed to the development of underlying datasets and early benchmarking efforts in [BatchNLPSolver.jl](https://github.com/LearningToOptimize/BatchNLPSolver.jl)
* The organizers of the [2025 PSC/CMU/Pitt Open Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00000rxGGo2AM/se000359),
    and especially our team's mentor, Lars Nyland
* The [Geogia Tech Partnership for an Advanced Computing Environment](https://pace.gatech.edu/) (PACE),
    whose Phoenix cluster was used to develop this tool
* The [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) developers, with special thanks to Alexis Montoison ([@amontoison](https://github.com/amontoison)) for technical advice on how to use CUDSS