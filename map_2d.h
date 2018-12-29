#pragma once

#include <functional>

#include "image.h"

namespace harris {

// Maps an image using a simple functor (take one pixel and produce one pixel)
// The output image is the same size as the input image
// func has the form Dest(Src) and is called for each src pixel and the output is used as the output pixel
template <class Dest, class Src, typename MapFunc>
Image<Dest> Map(const Image<Src>& src, MapFunc func) {
    const int width = src.width();
    const int height = src.height();
    Image<Dest> dest(width, height);

    #pragma omp parallel for
    for(auto y=0; y < height; ++y) {
        const auto src_ptr = src.RowPtr(y);
        auto dest_ptr = dest.RowPtr(y);
        for(auto x=0; x < width; ++x) {
            const auto src_pixel = src_ptr[x];
            const auto dest_pixel = func(src_pixel);
            dest_ptr[x] = dest_pixel;
        }
    }

    return dest;
}

// Reduces an image to a single value based on an accumulator function.
// The function takes an accumulator value and a pixel value and returns a new accumulator value (float ReduceFunc(float acc, float pixel))
// func has the form Acc(Acc, Src) and is called for each src pixel and the final value for func is returned by the function
template <class Acc, class Src, typename ReduceFunc>
Acc Reduce(const Image<Src>& src, Acc acc, ReduceFunc func) {
    const int width = src.width();
    const int height = src.height();

    #pragma omp parallel for
    for(auto y=0; y < height; ++y) {
        const auto src_ptr = src.RowPtr(y);
        for(auto x=0; x < width; ++x) {
            const auto src_pixel = src_ptr[x];
            acc = func(acc, src_pixel);
        }
    }

    return acc;
}

// Combines multiple images using a simple functor (take one pixel from each src and produces one pixel)
// The source images must be the same size.
// The output image is the same size as the input images.
// func has the form Dest(Src, Src) and is called for each pair of input pixels and the result is used as the output pixel.
template <class Dest, class Src, typename CombineFunc> 
Image<Dest> Combine(const Image<Src>& src1, const Image<Src>& src2, CombineFunc func) {
    if (src1.width() != src2.width()) throw std::invalid_argument("src images must be the same size");
    if (src1.height() != src2.height()) throw std::invalid_argument("src images must be the same size");
    const int width = src1.width();
    const int height = src1.height();
    Image<Dest> dest(width, height);

    #pragma omp parallel for
    for(auto y=0; y < height; ++y) {
        const auto src1_ptr = src1.RowPtr(y);
        const auto src2_ptr = src2.RowPtr(y);
        auto dest_ptr = dest.RowPtr(y);
        for(auto x=0; x < width; ++x) {
            const auto src1_pixel = src1_ptr[x];
            const auto src2_pixel = src2_ptr[x];
            const auto dest_pixel = func(src1_pixel, src2_pixel);
            dest_ptr[x] = dest_pixel;
        }
    }

    return dest;
}

// Maps an image using a windowed accumulator.
// The output image is the same size as the input image.
// init_func has the form Acc(Src) is called for each src pixel and provides an initial accumulator value for the window
// window_func has the form Acc(Acc, Src) is called for source pixel in the window along with the last accumulator value
// final_func has the form Dest(Acc) and will be called with the last accumulator value.
// The return value from final_func will used for the pixel.
template <class Dest, class Acc, class Src, typename InitFunc, typename WindowFunc, typename FinalFunc> 
Image<Dest> MapWindowed(
    const Image<Src>& src, 
    int window_size, 
    InitFunc init_func,
    WindowFunc window_func,
    FinalFunc final_func) {
    if(window_size <= 0 || window_size % 2 == 0) throw std::invalid_argument("window_size must be a positive odd number");

    const int width = src.width();
    const int height = src.height();
    const int max_x = width - 1;
    const int max_y = height - 1;
    const int half_window = window_size / 2;
    Image<Dest> dest(width, height);

    #pragma omp parallel for
    for(auto dest_y=0; dest_y < height; ++dest_y) {
        auto dest_row = dest.RowPtr(dest_y);
        auto src_row = src.RowPtr(dest_y);
        for(auto dest_x=0; dest_x < width; ++dest_x) {
            const auto src_pixel = src_row[dest_x];
            auto acc = init_func(src_pixel);

            for(auto window_y = dest_y - half_window; window_y <= dest_y + half_window; ++window_y) {
                const auto window_src_y = Reflect(window_y, 0, max_y);
                const auto window_src_row = src.RowPtr(window_src_y);
                for(auto window_x = dest_x - half_window; window_x <= dest_x + half_window; ++window_x) {
                    const auto window_src_x = Reflect(window_x, 0, max_x);
                    const auto window_src_pixel = window_src_row[window_src_x];
                    acc = window_func(acc, window_src_pixel);
                }
            }

            dest_row[dest_x] = final_func(acc);
        }
    }

    return dest;
}

// Combines multiple images using a windowed accumulator.
// (i.e. Each source pixel in a window around a dest pixel will be accumulated to a final value)
// The source images must be the same size.
// The output image is the same size as the input images.
// init_func has the form Dest(Src, Src) and is called for each pair of src pixels and provides an initial accumulator value for the window
// window_func has the form Dest(Dest, Src, Src) and is called for each pair of source pixels in the window along with the last returned value (i.e. accumulated value)
// The last value returned by window_func will be the value used for the pixel.
template <class Dest, class Src, typename InitFunc, typename WindowFunc> 
Image<Dest> CombineWindowed(
    const Image<Src>& src1, 
    const Image<Src>& src2, 
    int window_size, 
    InitFunc init_func, 
    WindowFunc window_func) {

    if (src1.width() != src2.width()) throw std::invalid_argument("src images must be the same size");
    if (src1.height() != src2.height()) throw std::invalid_argument("src images must be the same size");
    if (window_size <= 0 || window_size % 2 == 0) throw std::invalid_argument("window_size must be a positive odd number");
    const int width = src1.width();
    const int height = src1.height();
    const int max_x = width - 1;
    const int max_y = height - 1;
    const int half_window = window_size / 2;
    Image<Dest> dest(width, height);

    #pragma omp parallel for
    for(auto dest_y=0; dest_y < height; ++dest_y) {
        auto dest_row = dest.RowPtr(dest_y);
        auto src1_row = src1.RowPtr(dest_y);
        auto src2_row = src2.RowPtr(dest_y);
        for(auto dest_x=0; dest_x < width; ++dest_x) {
            auto acc = init_func(src1_row[dest_x], src2_row[dest_x]);
            for(auto window_y = -half_window; window_y <= half_window; ++window_y) {
                const auto window_src_y = Reflect(dest_y + window_y, 0, max_y);
                const auto window1_row = src1.RowPtr(window_src_y);
                const auto window2_row = src2.RowPtr(window_src_y);
                for(auto window_x = -half_window; window_x <= half_window; ++window_x) {
                    const auto window_src_x = Reflect(dest_x + window_x, 0, max_x);
                    const auto window1_pixel = window1_row[window_src_x];
                    const auto window2_pixel = window2_row[window_src_x];
                    acc = window_func(acc, window1_pixel, window2_pixel);
                }
            }

            dest_row[dest_x] = acc;
        }
    }

    return dest;
}

}