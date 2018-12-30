#pragma once

#include "harris_base.h"
#include "harris_corner_detector.h"

namespace harris {

class HarrisCpp : public HarrisBase {
public:

    HarrisCpp(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size) {
    }

    // Rule of five: Neither movable nor copyable
    HarrisCpp(const HarrisCpp&) = delete;
    HarrisCpp(HarrisCpp&&) = delete;
    HarrisCpp& operator=(const HarrisCpp&) = delete;
    HarrisCpp& operator=(HarrisCpp&&) = delete;
    ~HarrisCpp() override = default;

    // Runs the pure C++ Harris corner detector
    Image<float> FindCorners(const Image<float>& image) override {
        auto harris_img = HarrisCorners(image, smoothing_size_, structure_size_, k_, threshold_ratio_, suppression_size_);
        return harris_img;
    }
};
}

