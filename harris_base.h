#pragma once

#include "image.h"
#include "image_conversion.h"

namespace harris {

class HarrisBase {
public:

    HarrisBase(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
    smoothing_size_(smoothing_size),
    structure_size_(structure_size),
    k_(harris_k),
    threshold_ratio_(threshold_ratio),
    suppression_size_(suppression_size) {
        if(smoothing_size <= 0 || smoothing_size % 2 == 0) throw std::invalid_argument("smoothing_size must be a positive odd number");
        if(structure_size <= 0 || structure_size % 2 == 0) throw std::invalid_argument("structure_size must be a positive odd number");
        if(suppression_size <= 0 || suppression_size % 2 == 0) throw std::invalid_argument("suppression_size must be a positive odd number");
        if(harris_k <= 0) throw std::invalid_argument("harris_k must be positive");
        if(threshold_ratio < 0 || threshold_ratio > 1) throw std::invalid_argument("threshold_ratio must be between 0 and 1");
    }

    // Rule of five: Neither movable nor copyable
    HarrisBase(const HarrisBase&) = delete;
    HarrisBase(HarrisBase&&) = delete;
    HarrisBase& operator=(const HarrisBase&) = delete;
    HarrisBase& operator=(HarrisBase&&) = delete;
    virtual ~HarrisBase() = default;

    virtual Image<float> FindCorners(const Image<Argb32>& image) = 0;

    int smoothing_size() const { return smoothing_size_; }
    int structure_size() const { return structure_size_; }
    int suppression_size() const { return suppression_size_; }
    float k() const { return k_; }
    float threshold_ratio() const { return threshold_ratio_; }

protected:
    int smoothing_size_;
    int structure_size_;
    float k_;
    float threshold_ratio_; 
    int suppression_size_;
};
}