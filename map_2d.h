#pragma once

#include <functional>

#include "image.h"

namespace harris {

// Maps an image using a simple functor (take one pixel and produce one pixel)
// The output image is the same size as the input image
// func has the form Dest MapFunc(Src) and is called for each src pixel and the output is used as the output pixel
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

// Represents an index to a pixel on an image
struct Point {
    Point(int x, int y) : x(x), y(y) {

    }

    int x;
    int y;
};

// Maps an image using a simple functor (take one pixel and produce one pixel)
// The output image is the same size as the input image
// func has the form Dest MapFunc(Src, Point) and is called for each src pixel and the output is used as the output pixel
template <class Dest, class Src, typename MapFunc>
Image<Dest> MapWithIndex(const Image<Src>& src, MapFunc func) {
    const int width = src.width();
    const int height = src.height();
    Image<Dest> dest(width, height);

    #pragma omp parallel for
    for(auto y=0; y < height; ++y) {
        const auto src_ptr = src.RowPtr(y);
        auto dest_ptr = dest.RowPtr(y);
        for(auto x=0; x < width; ++x) {
            const auto src_pixel = src_ptr[x];
            const auto dest_pixel = func(src_pixel, Point{x, y});
            dest_ptr[x] = dest_pixel;
        }
    }

    return dest;
}

// Reduces an image to a single value based on an accumulator function.
// The function takes an accumulator value and a pixel value and returns a new accumulator value (float ReduceFunc(float acc, float pixel))
// func has the form Acc ReduceFunc(Acc, Src) and is called for each src pixel and the final value for func is returned by the function
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

// Represents a range of pixels starting at (x1, y1) and ending at (x2, y2) (inclusive)
struct Range {
    Range(int x1, int y1, int x2, int y2) :
        x1(x1),
        y1(y1),
        x2(x2),
        y2(y2) {

    }

    int x1;
    int y1;
    int x2;
    int y2;
};

// Reduces a range of an image to a single value based on an accumulator function.
// func has the form Acc ReduceFunc(Acc, Src) and is called for each src pixel and the final value for func is returned by the function
template <class Acc, class Src, typename ReduceFunc>
Acc ReduceRange(const Image<Src>& src, const Range& range, Acc acc, ReduceFunc func) {
    const auto max_x = src.width() - 1;
    const auto max_y = src.height() - 1;
    
    #pragma omp parallel for
    for(auto y = range.y1; y <= range.y2; ++y) {
        const auto safe_y = Reflect(y, 0, max_y);
        const auto src_ptr = src.RowPtr(safe_y);
        for(auto x = range.x1; x <= range.x2; ++x) {
            const auto safe_x = Reflect(x, 0, max_x);
            const auto src_pixel = src_ptr[safe_x];
            acc = func(acc, src_pixel);
        }
    }

    return acc;
}

// Reduces a range of an image to a single value based on an accumulator function.
// func has the form Acc ReduceFunc(Acc, Src, Src) and is called for each src pixel and the final value for func is returned by the function
template <class Acc, class Src, typename ReduceFunc>
Acc ReduceRange(const Image<Src>& src1, const Image<Src>& src2, const Range& range, Acc acc, ReduceFunc func) {
    const auto max_x = src1.width() - 1;
    const auto max_y = src1.height() - 1;
    
    #pragma omp parallel for
    for(auto y = range.y1; y <= range.y2; ++y) {
        const auto safe_y = Reflect(y, 0, max_y);
        const auto src1_row = src1.RowPtr(safe_y);
        const auto src2_row = src2.RowPtr(safe_y);
        for(auto x = range.x1; x <= range.x2; ++x) {
            const auto safe_x = Reflect(x, 0, max_x);
            const auto src1_pixel = src1_row[safe_x];
            const auto src2_pixel = src2_row[safe_x];
            acc = func(acc, src1_pixel, src2_pixel);
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

// Combines multiple images using a simple functor (take one pixel from each src and produces one pixel)
// The source images must be the same size.
// The output image is the same size as the input images.
// func has the form Dest CombineFunc(Src, Src, Point) and is called for each pair of input pixels and the result is used as the output pixel.
template <class Dest, class Src, typename CombineFunc> 
Image<Dest> CombineWithIndex(const Image<Src>& src1, const Image<Src>& src2, CombineFunc func) {
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
            const auto dest_pixel = func(src1_pixel, src2_pixel, Point{x,y});
            dest_ptr[x] = dest_pixel;
        }
    }

    return dest;
}

}