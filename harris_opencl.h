#pragma once

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#include "harris_base.h"

namespace harris {

class HarrisOpenCL : public HarrisBase {
public:

    HarrisOpenCL(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size) {
    }

    // Rule of five: Neither movable nor copyable
    HarrisOpenCL(const HarrisOpenCL&) = delete;
    HarrisOpenCL(HarrisOpenCL&&) = delete;
    HarrisOpenCL& operator=(const HarrisOpenCL&) = delete;
    HarrisOpenCL& operator=(HarrisOpenCL&&) = delete;
    ~HarrisOpenCL() override = default;

    // Runs the OpenCL Harris corner detector
    Image<float> FindCorners(const Image<float>& image) override {
        throw std::runtime_error("OpenCL is not implemented yet");
    }
};
}

