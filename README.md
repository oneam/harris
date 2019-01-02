# Harris Corners [![Build Status](https://travis-ci.org/oneam/harris.svg)](https://travis-ci.org/oneam/harris)

This repo contains an example of the Harris corner detection algorithm implemented in pure C++.

## Prerequisites

### OpenCV

[OpenCV](https://opencv.org/) is used as a reference implementation as well as to load images and videos for processing.
CMake will look for OpenCV in standard install locations (see [OpenCV Cmake Docs](https://docs.opencv.org/3.4.5/db/df5/tutorial_linux_gcc_cmake.html) for more info)

### Googletest

Googletest is used for unit testing. It is referenced via gitmodule so cloning this repo should also clone the Googletest repo from github.

### OpenCL

[OpenCL](https://www.khronos.org/opencl/) is used for the OpenCL implementation of the algorithm.
CMake should find OpenCL automatically from a standard install location (see [CMake FindOpenCL Docs](https://cmake.org/cmake/help/latest/module/FindOpenCL.html) for more info)

### OpenMP (optional)

If found, the build will attempt to use [OpenMP](https://www.openmp.org/) to accelerate the pure C++ implementation
CMake should find OpenMP automatically from a standard install location (see [CMake FindOpenMP Docs](https://cmake.org/cmake/help/latest/module/FindOpenMP.html) for more info)

## Build Instructions

The project can be built using CMake version 3.11 or higher. To build the project, open a command prompt in the repo top level and run:

```
cmake .
cmake --build .
```

The built application is called harris (or harris.exe) and provides a help message to make it easier to use:
```
./harris --help

Harris Corner Detector Demo
Usage: harris [params] input 

	-?, -h, --help, --usage (value:true)
		Print this message
	-b, --benchmark
		Prints the rendering time for each frame as it's converted
	--cl-device (value:0)
		The index of the device to use when runnning OpenCL algorithm
	--cl-platform (value:0)
		The index of the platform to use when runnning OpenCL algorithm
	--harris_k, -k (value:0.04)
		The value of the Harris free parameter
	-o, --output
		Outputs a version of the input with markers on each corner (use a file that ends with .m4v to output a video)
	--opencl
		Use the OpenCL algorithm rather than the pure C++ method
	--opencv
		Use the OpenCV algorithm rather than the pure C++ method
	-s, --show
		Displays a window containing a version of the input with markers on each corner
	--smoothing (value:5)
		The size (in pixels) of the gaussian smoothing kernel. This must be an odd number
	--structure (value:5)
		The size (in pixels) of the window used to define the structure tensor of each pixel
	--suppression (value:9)
		The size (in pixels) of the non-maximum suppression window
	--threshold (value:0.5)
		The Harris response suppression threshold defined as a ratio of the maximum response value

	input
		Input image or video
```

## Running the demo

Running the demo is as simple as pointing an image or video at it. 

```
./harris --show aruco.m4v
```

Most formats that are readable by OpenCV will be supported.

The `--show` param is used to display the images with corners highlighted.
After the last image in the sequence is displayed, the application will pause waiting for a key to be pressed.

## Running Unit Tests

The project uses the CMake test framework and Googletest. Running the unit tests is as simple as calling:

```
ctest
```

Currently the unit tests simply run a correctness test by inputting a pre created image (lines.png) with well known corner points.

I would have liked to do a more thurough set of unit tests, but isn't that always the case?

## Benchmark

Both pure C++ and OpenCL implenetations were run against the aruco.m4v file included in the repo. the following benchmark timing results were measured:

| Method | Total Time | Average time per frame |
|--------|------------|------------------------|
| OpenCL | 10.9069 s  | 52.1862 ms             |
| C++    | 158.245 s  | 757.154 ms             |

## Implementation Notes

I developed this using MacOS and OpenCL 1.2 using a Radeon video card. While I have attempted to make it build on any system, I haven't had time
or resources to test it on any other system.

I chose to only support ARGB color mode for input images.
I generally prefer this for my current projects since the processing is simpler and most SIMD instructions are
based on processing powers of 2.

## C++

### Implementation description

#### Header Files

* `harris_cpp.h` - Contains the actual Harris Corner Detection algorithm implemented in pure C++
    * It's a class derived from the generic HarrisBAse class used by all implementations
* `image.h` - Contanis an implementation of a generic 2D image of a given pixel format. 3 formats are currently used by the algorithm:
    * `float` - A simple greyscale image that stores values as floating point values 0..1.
    * `Argb32` - a 32bits per pixel ARGB format (standard format used by Windows and OpenCV).
    * `StructureTensor` - Used to create a Structure Tensor image where each pixel represent that structure of surrounding pixels.
* `filter_2d.h` - Contains an implementation of a cross-correlation algorithm for filtering images.
    * It's used for Gaussian smoothing and image differentiation.
    * The implementation also includes a Gaussian kernel creator that builds a kernel that fits nicely within a given size.
* `map_2d.h` - Contains an implementation of MapReduce algorithms for images.
    * It's used for the rest of the algorithm to do things like converting from color to greyscale images, structure tensor creation and non-max suppression.
* `image_conversion.h` - Contains method to convert from color to greyscale floating point images used by default for Harris implementations.
* `numerics.h` - Simple numerical calculations that don't exist in C++ standard libraries.

### Other Notes

I chose to make the implementation header only since I made extensive use of templates, which don't work well with code-behind files.
The implementation is also relatively simple so it only took a few files.

I chose to use [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) for the code with a few exceptions:
* I used `#pragma once` instead fo the standard #ifndef header value.
* I left out the standard preamble in each file since I meant the files to be simple.

I prefer not to use the generic ReduceFunc/MapFunc/CombineFunc template params but found that the compiler had a hard time with `std::function<>`
types. Template type deduction did not work for `std::function<>` types and the optimize code produced by the output was much slower.

## OpenCL

This is my first OpenCL project, so I may have made some design decisions that don't match standard practices.

I decided to use the C++ binding for OpenCL as it did automatic releasing in order to make the code a little cleaner.
I used a local version of cl.hpp since my development environmment didn't have it available.

The implementation of Max value reduction is not my favorite. If I had more time I would see if I could optimize it more.
My assumption is that there is an optimal size for work-items that is bigger than one pixel but smaller than one row. I didn't spend enough time trying to find that optimal size.

If I had more time I would implement a method to cascade the enqueue calls so that the the events are still coordinated but the code is less redundant.
I have an idea how, but haven't found time to implement it.

I couldn't keep the CPU version of the code working on my system. It kept dying with a strange error code "Illegal Instruction: 4" and Google was only vaguely helpful.

## OpenCV

I know it was not a part of the homework but I had initally used it as a reference for my implementation so I left it there for reference.
