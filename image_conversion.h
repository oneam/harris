#pragma once

#include <cmath>

#include "numerics.h"
#include "image.h"
#include "map_2d.h"

namespace harris {

Image<float> ToFloat(const Image<Argb32>& src) {
    Image<float> dest = Map<float>(src, [](Argb32 src_pixel) {
        // Extract the floating point color components
        const auto r = src_pixel.RedFloat();
        const auto g = src_pixel.GreenFloat();
        const auto b = src_pixel.BlueFloat();

        // Using Rec.709 luma conversion as per sRGB
        const auto luma = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        return luma;
    });

    return dest;
}

Image<Argb32> ToArgb32(const Image<float>& src) {
    Image<Argb32> dest = Map<Argb32>(src, [](float src_pixel) { 
        return Argb32(1.0f, src_pixel, src_pixel, src_pixel); 
    });

    return dest;
}

}