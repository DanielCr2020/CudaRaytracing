#Windows makefiles are stupid and I can't use $() to set variables
all: clean cpu gpu setup
cpu:
	nvcc mainCPU.cu -o mainCPU -I./ -O3
gpu:
	nvcc mainGPU.cu -o mainGPU -I./ -O3

.PHONY:
	clean echo

clean:
	del *.exe *.exp *.lib *.pdb

runcpu:
	mainCPU.exe > outputCPU.ppm
	magick outputCPU.ppm outputCPU.png
	cmd /c .\outputCPU.png
fullcpu: cpu runcpu

rungpu:
	mainGPU.exe > outputGPU.ppm
	magick outputGPU.ppm outputGPU.png
	cmd /c .\outputGPU.png
fullgpu: gpu rungpu

full: clean fullcpu fullgpu

run: runcpu rungpu

setup:
	g++ setup.cpp -o setup -O3

runsetup:
	setup.exe