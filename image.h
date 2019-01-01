#pragma once

#include "numerics.h"

#include <initializer_list>
#include <vector>

namespace harris {

// A 32 bits per pixel with full color (sRGB) with an alpha channel.
// Each pixel is represented as a uint32_t with a hex value of 0xAARRGGBBU
// (This is not the same as byte order on many systems)
struct Argb32 {
    uint32_t data;

    // Creates an Argb32 using floating point values clamped to the range (0,1)
    Argb32(float alpha, float red, float green, float blue) 
    {
        const auto a = static_cast<uint32_t>(round(Clamp(alpha, 0.0f, 1.0f) * 255.0f));
        const auto r = static_cast<uint32_t>(round(Clamp(red, 0.0f, 1.0f) * 255.0f));
        const auto g = static_cast<uint32_t>(round(Clamp(green, 0.0f, 1.0f) * 255.0f));
        const auto b = static_cast<uint32_t>(round(Clamp(blue, 0.0f, 1.0f) * 255.0f));
        data = (((((a << 8) + r) << 8) + g) << 8) + b;
    }

    // Creates an Argb32 using integer values clamped to the range (0,255)
    Argb32(int alpha, int red, int green, int blue) {
        const auto a = static_cast<uint32_t>(Clamp(alpha, 0, 255));
        const auto r = static_cast<uint32_t>(Clamp(red, 0, 255));
        const auto g = static_cast<uint32_t>(Clamp(green, 0, 255));
        const auto b = static_cast<uint32_t>(Clamp(blue, 0, 255));
        data = (((((a << 8) + r) << 8) + g) << 8) + b;
    }

    float AlphaFloat() const { return static_cast<float>(alpha()) / 255.0f; }
    float RedFloat() const { return static_cast<float>(red()) / 255.0f; }
    float GreenFloat() const { return static_cast<float>(green()) / 255.0f; }
    float BlueFloat() const { return static_cast<float>(blue()) / 255.0f; }

    uint8_t alpha() const { return static_cast<uint8_t>(data >> 24 & 0xffU); }
    uint8_t red() const { return static_cast<uint8_t>(data >> 16 & 0xffU); }
    uint8_t green() const { return static_cast<uint8_t>(data >> 8 & 0xffU); }
    uint8_t blue() const { return static_cast<uint8_t>(data & 0xffU); }
};

// A pixel containing a structure tensor.
struct StructureTensor {

    // Creates an structure tensor with values 0
    StructureTensor() : xx(0), yy(0), xy(0) {
    }

    // Creates an structure tensor using floating point values
    StructureTensor(float s_xx, float s_yy, float s_xy) : xx(s_xx), yy(s_yy), xy(s_xy) {
    }

    float xx;
    float yy;
    float xy;
};

// Templated image type.
// All images must provide a pixel type and can be accessed via row pointer for that type.
template <class P>
class Image {
public:
    using PixelType = P;
    
    // Rule of five: moveable and copyable
    Image(const Image&) = default;
    Image(Image&&) = default;
    Image& operator=(const Image&) = default;
    Image& operator=(Image&&) = default;
    virtual ~Image() = default;

    // Creates an empty image
    Image() : 
        width_(0), 
        height_(0),
        stride_(0),
        data_() {
        }

    // Creates an image of a given size
    Image(int width, int height) : 
        width_(width), 
        height_(height),
        stride_(width*sizeof(P)),
        data_(width*height*sizeof(P)) {
            if (width <= 0) throw std::invalid_argument("The width parameter must be larger than zero");
            if (height <= 0) throw std::invalid_argument("The height parameter must be larger than zero");
        }

    // Creates an image from vector input data.
    Image(std::vector<uint8_t> data, int width, int height, size_t stride) :
        width_(width),
        height_(height),
        stride_(stride),
        data_(std::move(data)) {
            if (width <= 0) throw std::invalid_argument("The width parameter must be larger than zero");
            if (height <= 0) throw std::invalid_argument("The height parameter must be larger than zero");
            if (stride < width*4) throw std::invalid_argument("The stride paramter is not large enough to fit the width of the image");
            if (data_.size() < stride*height) throw std::invalid_argument("The data parameter is not large enough to fit the entire image.");
        }

    // Creates an image by copying data directly from memory.
    Image(const uint8_t* data, int width, int height, size_t stride) :
        width_(width),
        height_(height),
        stride_(stride),
        data_(data, data + stride*height) {
            if (width <= 0) throw std::invalid_argument("The width parameter must be larger than zero");
            if (height <= 0) throw std::invalid_argument("The height parameter must be larger than zero");
            if (stride < width*sizeof(P)) throw std::invalid_argument("The stride paramter is not large enough to fit the width of the image");
        }

    // Accessors

    int width() const { return width_; }
    int height() const { return height_; }
    size_t stride() const { return stride_; }

    // Const data accessor.
    // This will be a buffer with size at least height*stride bytes organized in raster-scan order.
    // (i.e. each pixel is indexed at data()[y * stride + x])
    const uint8_t* data() const { return data_.data(); }

    // Non-const data accessor.
    // This will be a buffer with size at least height*stride bytes organized in raster-scan order.
    // (i.e. each pixel is indexed at data()[y * stride + x])
    uint8_t* data() { return data_.data(); }

    // Used to check for an empty image
    bool empty() const { return width_ <= 0; }
    operator bool() const { return !empty(); }

    // Const pixel accessor.
    // This will be a pointer to the first pixel in the given row.
    // Accessing by row provides generally more performant way to access pixel data.
    const PixelType* RowPtr(int y) const { return reinterpret_cast<const PixelType*>(data_.data() + y * stride_); }

    // Non-const pixel accessor.
    // This will be a pointer to the first pixel in the given row.
    // Accessing by row provides generally more performant way to access pixel data.
    PixelType* RowPtr(int y) { return reinterpret_cast<PixelType*>(data_.data() + y * stride_); }

private:
    int width_;
    int height_;
    int stride_;
    std::vector<uint8_t> data_;
};

}
