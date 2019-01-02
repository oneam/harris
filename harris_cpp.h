#pragma once

#include <cmath>
#include <vector>

#include "harris_base.h"
#include "image.h"
#include "image_conversion.h"
#include "filter_2d.h"
#include "map_2d.h"

namespace harris {

class HarrisCpp : public HarrisBase {
public:

    HarrisCpp(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size),
        gaussian_kernel_(GaussianKernel(smoothing_size)),
        diff_x_(3, 1, {1.f,  0.f, -1.f}), // The x differentiation operator from Sobel without the gaussian smoothing
        diff_y_(1, 3, {1.f,  0.f, -1.f}) { // The y differentiation operator from Sobel without the gaussian smoothing
    }

    // Rule of five: Neither movable nor copyable
    HarrisCpp(const HarrisCpp&) = delete;
    HarrisCpp(HarrisCpp&&) = delete;
    HarrisCpp& operator=(const HarrisCpp&) = delete;
    HarrisCpp& operator=(HarrisCpp&&) = delete;
    ~HarrisCpp() override = default;

    // Runs the pure C++ Harris corner detector
    Image<float> FindCorners(const Image<Argb32>& image) override {
        // Convert to float image
        const auto float_image = ToFloat(image);

        // Compute the structure tensor image
        const auto structure_tensor = StructureTensorImage(float_image);

        // Compute the Harris response
        const auto response = Map<float>(structure_tensor, [k = k_](StructureTensor s) { return (s.xx * s.yy - s.xy * s.xy) - k * (s.xx + s.yy) * (s.xx + s.yy); });

        // Find the maximum response value
        const auto max_r = Reduce<float>(response, 0.0f, [](float acc, float p) { return std::max(acc, p); });

        // Run non-maximal suppression with thresholding. The threshold is some fraction of the maximum response.
        const auto threshold = max_r * threshold_ratio_;
        const auto corners = NonMaxSuppression(response, threshold);

        return corners;
    }

private:
    FilterKernel gaussian_kernel_;
    FilterKernel diff_x_;
    FilterKernel diff_y_;

    // Computes the structure tensor image for a given image.
    Image<StructureTensor> StructureTensorImage(const Image<float>& src) {
        int half_window = structure_size_ / 2;

        const auto i_smooth = Filter2d(src, gaussian_kernel_);
        const auto i_x = Filter2d(i_smooth, diff_x_);
        const auto i_y = Filter2d(i_smooth, diff_y_);
        auto dest = CombineWithIndex<StructureTensor>(
            i_x,
            i_y,
            [&] (float s_x, float s_y, Point p) {
                // Reduce the structure around the pixel using sum of products
                Range range(p.x - half_window, p.y - half_window, p.x + half_window, p.y + half_window);
                return ReduceRange<StructureTensor>(i_x, i_y, range, StructureTensor(), [](StructureTensor s, float s_x, float s_y) {
                    s.xx += s_x * s_x;
                    s.yy += s_y * s_y;
                    s.xy += s_x * s_y;
                    return s;
                });
            });

        return dest;
    }

    // Computes a windowed non-maximal suppression image with a global threshold.
    Image<float> NonMaxSuppression(const Image<float>& src, float threshold) {
        int half_window = suppression_size_ / 2;

        auto dest = MapWithIndex<float>(
            src,
            [&](float src_pixel, Point p) {
                // If the pixel is below threshold, stop here, otherwise determine if it's the max in range
                if (src_pixel < threshold) return 0.0f;
                Range range(p.x - half_window, p.y - half_window, p.x + half_window, p.y + half_window);
                return ReduceRange<float>(src, range, src_pixel, [](float acc, float src_pixel) {
                    return acc < src_pixel ? 0.0f : acc;
                });
            });

        return dest;
    }
};
}

