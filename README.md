# Harris Corners

This repo contains an example of the Harris corner detection algorithm implemented in pure C++.

## Prerequisites

### OpenCV

[OpenCV](https://opencv.org/) is used as a reference implementation as well as to load images and videos for processing.
CMake will look for OpenCV in standard install locations (see [OpenCV Cmake Docs](https://docs.opencv.org/3.4.5/db/df5/tutorial_linux_gcc_cmake.html) for more info)

### Googletest

Googletest is used for unit testing. It is referenced via gitmodule so cloning this repo should also clone the Googletest repo on github.

### OpenCL

[OpenCL](https://www.khronos.org/opencl/) is used for the OpenCL implementation of the algorithm.
CMake should find OpenCL automatically from a standard install location (see [CMake FindOpenCL Docs](https://cmake.org/cmake/help/latest/module/FindOpenCL.html) for more info)

### OpenMP (optional)

If found, the build will attempt to use [OpenMP](https://www.openmp.org/) to accelerate teh pure C++ implementation
CMake should find OpenMP automatically from a standard install location (see [CMake FindOpenMP Docs](https://cmake.org/cmake/help/latest/module/FindOpenMP.html) for more info)

## Build Instructions

The project can be built using CMake version 3.11 or higher. To build the project, you should be able to open a command prompt in the repo top level and run:

```
cmake --build .
```

Once built, the application is called harris and provides a help message:
```
./harris --help

Harris Corner Detector Demo
Usage: harris [params] input 

	-?, -h, --help, --usage (value:true)
		print this message
	--harris_k, -k (value:0.04)
		The value of the Harris free parameter
	-o, --output
		outputs a version of the input with markers on each corner (.png or .m4v formats are supported)
	--opencv
		Use the OpenCV mathod to extract Harris coreners rather than the pure C++ method
	-s, --show
		displays a window containing a version of the input with markers on each corner
	--smoothing (value:5)
		The size (in pixels) of the gaussian smoothing kernel. This must be an odd number
	--structure (value:9)
		The size (in pixels) of the window used to define the structure tensor of each pixel
	--suppression (value:9)
		The size (in pixels) of the non-maximum suppression window
	--threshold (value:0.5)
		The Harris response suppression threshold defined as a ratio of the maximum response value

	input
		input image or video
```
## Running the demo

Running the demo is as simple as pointing an image or video at it. 
Most formats that are readbale by OpenCV will be supported.

```
./harris --show aruco.m4v
```

The `--show` param is used to display the images with corners highlighted.
After the last image in the sequence is displayed, the application will pause witing for a key to be pressed.
