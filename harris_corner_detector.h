#pragma once

#include <cmath>
#include <vector>

#include "image.h"
#include "filter_2d.h"
#include "map_2d.h"

namespace harris {

// Creates a nomralized gaussian filter kernel with the given size and sigma value.
FilterKernel GaussianKernel(float sigma, int size) {
    if (size <= 0 || size % 2 == 0) throw std::invalid_argument("size paramter must be a positive odd number");
    std::vector<float> kernel_values;

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

// Applies a nomralized gaussian filter with the given size and sigma value.
Image<float> Gaussian(const Image<float>& src, float sigma, int size) {
    auto kernel = GaussianKernel(sigma, size);
    return Filter2d(src, kernel);
}

// Calculates the dx derivative of the image using the Sobel operator (assuming that gaussian smoothing has already been done)
Image<float> SobelX(const Image<float>& src) {
    static FilterKernel sobel_x(3, 1, {1.f,  0.f, -1.f});
    return Filter2d(src, sobel_x);
}

// Calculates the dy derivative of the image using the Sobel operator (assuming that gaussian smoothing has already been done)
Image<float> SobelY(const Image<float>& src) {
    static FilterKernel sobel_y(1, 3, {1.f,  0.f, -1.f});
    return Filter2d(src, sobel_y);
}

// Computes the structure tensor image for a given image.
// This method uses the CombineWindowed method which is nicer conceptually but much slower.
Image<StructureTensor> StructureTensorImageUsingCombine(const Image<float>& src, int window_size = 5) {
    const auto i_smooth = Gaussian(src, 1.0f, 5);
    const auto i_x = SobelX(i_smooth);
    const auto i_y = SobelY(i_smooth);
    auto dest = CombineWindowed<StructureTensor>(
        i_x, 
        i_y, 
        window_size, 
        [](float i_x, float i_y) { return StructureTensor(); },
        [] (StructureTensor s, float i_x, float i_y) {
            s.xx += i_x * i_x;
            s.xy += i_x * i_y;
            s.yy += i_y * i_y;
            return s;
        });

    return dest;
}

// Computes a windowed non-maximal suppression image with a global threshold.
// This method uses the MapWindowed method which is nicer conceptually but much slower.
Image<float> NonMaxSuppressionUsingMap(const Image<float>& src, int window_size, float threshold) {
    auto dest = MapWindowed<float, float>(
        src, 
        window_size,
        [threshold](float src_pixel) { return (src_pixel > threshold) ? src_pixel : 0.0f; },
        [](float acc, float src_pixel) { return (src_pixel > acc) ? 0.0f : acc; },
        [](float acc) { return (acc > 0.0f) ? 1.0f : 0.0f; });

    return dest;
}

// Computes the structure tensor image for a given image.
Image<StructureTensor> StructureTensorImage(const Image<float>& src, int smoothing_size = 5, int structure_size = 5) {
    const auto i_smooth = Gaussian(src, 1.0f, smoothing_size);
    const auto i_x = SobelX(i_smooth);
    const auto i_y = SobelY(i_smooth);

    const int width = src.width();
    const int height = src.height();
    const int max_x = width - 1;
    const int max_y = height - 1;
    const int half_window = structure_size / 2;
    Image<StructureTensor> dest(width, height);

    #pragma omp parallel for
    for(auto dest_y = 0; dest_y < height; ++dest_y) {
        auto dest_row = dest.RowPtr(dest_y);
        auto i_x_row = i_x.RowPtr(dest_y);
        auto i_y_row = i_y.RowPtr(dest_y);
        for(auto dest_x = 0; dest_x < width; ++dest_x) {
            auto s_xx = 0.0f;
            auto s_xy = 0.0f;
            auto s_yy = 0.0f;
            for(auto window_y = dest_y - half_window; window_y <= dest_y + half_window; ++window_y) {
                const auto src_y = Reflect(window_y, 0, max_y);
                const auto window_i_x_row = i_x.RowPtr(src_y);
                const auto window_i_y_row = i_y.RowPtr(src_y);
                for(auto window_x = dest_x - half_window; window_x <= dest_x + half_window; ++window_x) {
                    const auto src_x = Reflect(window_x, 0, max_x);
                    const auto window_i_x_pixel = window_i_x_row[src_x];
                    const auto window_i_y_pixel = window_i_y_row[src_x];
                    s_xx += window_i_x_pixel * window_i_x_pixel;
                    s_xy += window_i_x_pixel * window_i_y_pixel;
                    s_yy += window_i_y_pixel * window_i_y_pixel;
                }
            }

            dest_row[dest_x] = StructureTensor(s_xx, s_xy, s_yy);
        }
    }

    return dest;
}

// Computes a windowed non-maximal suppression image with a global threshold
Image<float> NonMaxSuppression(const Image<float>& src, int window_size, float threshold) {
    if(window_size <= 0 || window_size % 2 == 0) throw std::invalid_argument("window_size must be a positive odd number");

    const int width = src.width();
    const int height = src.height();
    const int half_window = window_size / 2;
    Image<float> dest(width, height);

    #pragma omp parallel for
    for(auto dest_y=0; dest_y < height; ++dest_y) {
        auto dest_ptr = dest.RowPtr(dest_y);
        auto src_ptr = src.RowPtr(dest_y);
        for(auto dest_x=0; dest_x < width; ++dest_x) {
            auto dest_pixel = src_ptr[dest_x];
            if (dest_pixel < threshold) {
                dest_pixel = 0.0f;
                dest_ptr[dest_x] = dest_pixel;
                continue;
            }

            for(auto window_y = dest_y - half_window; dest_pixel > 0.0f && window_y <= dest_y + half_window; ++window_y) {
                if (window_y < 0 || window_y >= height) continue;
                const auto window_row = src.RowPtr(window_y);
                for(auto window_x = dest_x - half_window; window_x <= dest_x + half_window; ++window_x) {
                    if (window_x < 0 || window_x >= width) continue;
                    const auto window_pixel = window_row[window_x];
                    if (window_pixel > dest_pixel) {
                        dest_pixel = 0.0f;
                        break;
                    }
                }
            }

            dest_ptr[dest_x] = (dest_pixel > 0.0f) ? 1.0f : 0.0f;
        }
    }

    return dest;
}

// Runs Harris corner detection algorithm, producing a binary image where 1.0 is a corner and the rest of the image is 0.0
Image<float> HarrisCorners(const Image<float>& src, int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04f, float threshold_ratio = 0.5f, int suppression_size = 9) {
    const auto s = StructureTensorImage(src, smoothing_size, structure_size);
    const auto r = Map<float, StructureTensor>(s, [harris_k](StructureTensor s) { return (s.xx * s.yy - s.xy * s.xy) - harris_k * (s.xx + s.yy) * (s.xx + s.yy); });
    const auto max_r = Reduce<float, float>(r, 0.0f, [](float acc, float p) { return std::max(acc, p); });
    const auto threshold = max_r * threshold_ratio;
    const auto corners = NonMaxSuppression(r, suppression_size, threshold);

    return corners;
}

}