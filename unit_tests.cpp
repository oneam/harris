#include "gtest/gtest.h"

#include "opencv2/opencv.hpp"

#include "harris_cpp.h"
#include "harris_opencl.h"
#include "harris_opencv.h"
#include "image.h"

using namespace harris;

Image<Argb32> LoadImage(std::string filename) {
    cv::Mat mat = cv::imread(filename, cv::IMREAD_UNCHANGED);
    return Image<Argb32>(mat.data, mat.cols, mat.rows, mat.step[0]);
}

void CheckCorners(const Image<float>& output) {
    for(auto row = 0; row < output.height(); ++row) {
        const auto output_row = output.RowPtr(row);
        for(auto col = 0; col < output.width(); ++col) {
            const auto output_pixel = output_row[col];
            if (output_pixel > 0) {
                ASSERT_GT(row, 400) << "At point (" << col << "," << row << "): Corners must all be in the lower right quadrant";
                ASSERT_GT(col, 400) << "At point (" << col << "," << row << "): Corners must all be in the lower right quadrant";
                ASSERT_EQ((row - 10) % 20, 0) << "At point (" << col << "," << row << "): Corners must align with odd multiples of 10";
                ASSERT_EQ((col - 10) % 20, 0) << "At point (" << col << "," << row << "): Corners must align with odd multiples of 10";
            } else {
                ASSERT_FALSE(row > 400 && col > 400 && ((row - 10) % 20) == 0 && ((col - 10) % 20) == 0) << "At point (" << col << "," << row << "): There should be a point here";
            }
        }
    }
}

// Tests pure C++ implementation
TEST(AlgorithmTest, Cpp) {
    HarrisCpp harris;
    auto input = LoadImage("lines.png");
    auto output = harris.FindCorners(input);
    CheckCorners(output);
}

// Tests pure C++ implementation
TEST(AlgorithmTest, OpenCL) {
    HarrisOpenCL harris;
    auto input = LoadImage("lines.png");
    auto output = harris.FindCorners(input);
    CheckCorners(output);
}

// Tests pure C++ implementation
TEST(AlgorithmTest, OpenCV) {
    HarrisOpenCV harris;
    auto input = LoadImage("lines.png");
    auto output = harris.FindCorners(input);
    CheckCorners(output);
}
