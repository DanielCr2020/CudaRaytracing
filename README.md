### An archival repository for my GPU Programming final project

This code is heavily derived from [this implementation of ray tracing in CUDA](https://github.com/rogerallen/raytracinginoneweekend)

This project is a CUDA implementation of the "ray tracing in one weekend" programming project, with some additional kernels put in by me for testing purposes and more functionality.

I have added functionality for more robust image sizing, setting parameters (e.g. image size, samples per pixel, and number of objects) with a file, and random generation.

I am also using constant memory to store the random numbers for a speed improvement. This means that there is an upper limit on how many random numbers can be used.

This project was designed so that the GPU and CPU code could use the exact same random numbers and, in theory, produce the exact same image.

This project was developed on Windows, and thus the makefile is made for that. If you want it to work on linux, it will need to be modified.

The number of objects (besides the ones that were hardcoded in) is equal to loopBound\*loopBound\*2\*2

#### Running it

`make` in the top level directory to compile both the CPU and GPU versions.

`make setup` to compile the setup file.

`make runsetup` to run the setup executable. This executable creates the object generation and raytracing randoms needed for the image generation. This only needs to be run when you want different random numbers, or when you significantly increase the amount of randoms that are required and run the risk of indexing out of bounds.

 This means that running mainGPU or mainCPU multiple times will produce the same image until the random numbers are changed

The programs read from parameters.txt, raytrace_randoms.txt, and object_randoms.txt. Make sure those files exist in the same directory as the executables.

`./mainGPU.exe > output.ppm` to run the GPU version and redirect the output to a ppm file.

`./mainCPU.exe > output.ppm` to run the CPU version and redirect the output to a ppm file.

The makefile also includes procedures for running the entire process with one command (compiling, running, redirecting output, converting ppm to png with image magick, and opening the image)

##### If you're coming here looking for quality code written by me (since you probably won't find any here), I recommend checking out [my LS project](https://github.com/DanielCr2020/LS) instead!
