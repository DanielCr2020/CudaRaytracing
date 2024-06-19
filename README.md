### An archival repository for my GPU Programming final project

This code is heavily derived from [this implementation of ray tracing in CUDA](https://github.com/rogerallen/raytracinginoneweekend)

This project is a CUDA implementation of the "ray tracing in one weekend" programming project, with some additional kernels put in by me for testing purposes and more functionality.

I have added functionality for more robust image sizing, setting parameters (e.g. image size, samples per pixel, and number of objects) with a file, and random generation.

This project was designed so that the GPU and CPU code could use the exact same random numbers and, in theory, produce the exact same image.

This project was developed on Windows, and thus the makefile is made for that. If you want it to work on linux, it will need to be modified.

#### Running it

`make` in the top level directory to compile both the CPU and GPU versions. 

`./mainGPU.exe > output.ppm` to run the GPU version and redirect the output to a ppm file.

`./mainCPU.exe > output.ppm` to run the CPU version and redirect the output to a ppm file.

The makefile also includes procedures for running the entire process with one command (compiling, running, redirecting output, converting ppm to png with image magick, and opening the image)

##### If you're coming here looking for quality code written by me (since you probably won't find any here), I recommend checking out [my LS project](https://github.com/DanielCr2020/LS) instead!
