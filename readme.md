# CUDA Tutorial

## Introduction

This repository is intended for beginners of GPGPU programming, and includes several demos covering the most common kernels and optimization techniques. The repository also contains a framework to make it easier to run and test kernels.

This is a personal project. The provided implementations are for reference only and are not guaranteed to be optimal. Learners are encouraged to search for additional resources to improve their learning.

## Roadmap

A typical learning roadmap is listed below:

1. `vecadd`: Get started with CUDA programming. Set up environment and learn the basic concepts such as `grid` and `block`.

2. `matmul`: Implement a naive version of matrix multiplication. Learn to use the multidimensional layout of grid and block.

3. `matmul-opt`: Begin the journey into kernel optimization. Additional knowledge of GPU architecture is required to understand how to optimize memory access patterns and control flow of a CUDA kernel. Learn about SIMT, warps and the memory hierarchy of GPU and optimize the kernel using techniques like shared memory and thread coarsening.

4. `wmma`: Get the most out of the hardware by utilizing the tensor cores. Use warp-level matrix multiply-accumulate primitives to considerably speed up matrix multiplication.

5. `triton`: Take a quick look at the next-generation GPGPU programming paradigm. Write a simple vector addition kernel using Triton and compare it with CUDA programming.