#pragma once

#include <initializer_list>
#include <vector>

#include "numerics.h"
#include "image.h"

namespace harris {

// 2d cross-correlation kernel used in the Filter2d function
class FilterKernel {
public:
    // Rule of five: moveable and copyable
    FilterKernel(const FilterKernel&) = default;
    FilterKernel(FilterKernel&&) = default;
    FilterKernel& operator=(const FilterKernel&) = default;
    FilterKernel& operator=(FilterKernel&&) = default;
    virtual ~FilterKernel() = default;

    FilterKernel(int width, int height, std::initializer_list<float> values) : 
        width_(width),
        height_(height),
        data_(std::move(values)) {
        if (width <= 0) throw std::invalid_argument("width must be greater than or equal to 0");
        if (height <= 0) throw std::invalid_argument("height must be greater than or equal to 0");
        if (width % 2 == 0) throw std::invalid_argument("width must be odd");
        if (height % 2 == 0) throw std::invalid_argument("height must be odd");
        if (data_.size() != width*height) throw std::invalid_argument("There must be exactly width*height values in the kernel");
    }

    FilterKernel(int width, int height, std::vector<float> values) : 
        width_(width),
        height_(height),
        data_(std::move(values)) {
        if (width <= 0) throw std::invalid_argument("width must be greater than or equal to 0");
        if (height <= 0) throw std::invalid_argument("height must be greater than or equal to 0");
        if (width % 2 == 0) throw std::invalid_argument("width must be odd");
        if (height % 2 == 0) throw std::invalid_argument("height must be odd");
        if (data_.size() != width*height) throw std::invalid_argument("There must be exactly width*height values in the kernel");
    }

    // Accessors

    int width() const { return width_; }
    int height() const { return height_; }

    // Const data accessor.
    // This will be a buffer with size at least height*width bytes organized in raster-scan order.
    // (i.e. each pixel is indexed at data()[y * width + x])
    const float* data() const { return data_.data(); }

    // Non-const data accessor.
    // This will be a buffer with size at least height*width bytes organized in raster-scan order.
    // (i.e. each pixel is indexed at data()[y * width + x])
    float* data() { return data_.data(); }

    // Const kernel value accessor.
    // Each row will be width langth and can be accessed via row_ptr[x]
    const float* RowPtr(int y) const { return data_.data() + y * width_; }
    
private:
    int width_;
    int height_;
    std::vector<float> data_;
};

// Runs a 2d cross-correlation filter over an image.
// The output image will be the same size as the input image.
// The pixels beyond the edge of the image used for filtering will be derived from the reflection of edge pixels 
Image<float> Filter2d(const Image<float>& src, const FilterKernel& kernel) {
    const int width = src.width();
    const int height = src.height();
    const int max_x = width - 1;
    const int max_y = height - 1;
    const int kernel_width = kernel.width();
    const int kernel_height = kernel.height();
    const int kernel_x_offset = kernel_width / 2;
    const int kernel_y_offset = kernel_height / 2;
    Image<float> dest(width, height);

    #pragma omp parallel for
    for(auto dest_y=0; dest_y < height; ++dest_y) {
        auto dest_row = dest.RowPtr(dest_y);
        for(auto dest_x=0; dest_x < width; ++dest_x) {
            auto dest_pixel = 0.0f;
            for(auto kernel_y=0; kernel_y < kernel_height; ++kernel_y) {
                const auto src_y = Reflect(dest_y + kernel_y - kernel_y_offset, 0, max_y);
                const auto src_row = src.RowPtr(src_y);
                const auto kernel_row = kernel.RowPtr(kernel_y);
                for(auto kernel_x=0; kernel_x < kernel_width; ++kernel_x) {
                    const auto src_x = Reflect(dest_x + kernel_x - kernel_x_offset, 0, max_x);
                    const auto src_pixel = src_row[src_x];
                    const auto kernel_value = kernel_row[kernel_x];
                    dest_pixel += src_pixel * kernel_value;
                }
            }
            dest_row[dest_x] = dest_pixel;
        }
    }

    return dest;
}

// Creates a normalized gaussian filter kernel with the given size.
// The signma value for the filter is derived from the size such that >95% of the volume of the shape is contained within the filter
FilterKernel GaussianKernel(int size) {
    if (size <= 0 || size % 2 == 0) throw std::invalid_argument("size parameter must be a positive odd number");
    std::vector<float> kernel_values;

    // Define sigma such that the ~95% percent of the curve fits in the window (https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)
    const auto sigma = static_cast<float>(size - 1) / 4.0f;

    // Define gaussian value for each point in the kernel
    float sum = 0.0f;
    int offset = size / 2;
    for(auto y=0; y < size; ++y)
    for(auto x=0; x < size; ++x) {
        auto x_f = static_cast<float>(x - offset);
        auto y_f = static_cast<float>(y - offset);
        auto value = std::exp(-(x_f * x_f + y_f * y_f) / (2.0f * sigma * sigma));
        sum += value;
        kernel_values.push_back(value);
    }

    // Normalize the kernel
    for(auto& value : kernel_values) {
        value /= sum;
    }

    return FilterKernel(size, size, std::move(kernel_values));
}

}