#pragma once
// This is an OpenCV implementation of the algorithm used as a reference for other implementations.

#include "harris_base.h"
#include "opencv2/opencv.hpp"

namespace harris {


class HarrisOpenCV : public HarrisBase {
public:

    HarrisOpenCV(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size) {
    }

    // Rule of five: Neither movable nor copyable
    HarrisOpenCV(const HarrisOpenCV&) = delete;
    HarrisOpenCV(HarrisOpenCV&&) = delete;
    HarrisOpenCV& operator=(const HarrisOpenCV&) = delete;
    HarrisOpenCV& operator=(HarrisOpenCV&&) = delete;
    ~HarrisOpenCV() override = default;

    Image<float> FindCorners(const Image<float>& image) override {
        cv::Mat image_mat;
        cv::Mat corners_mat;
        ToMat(image, image_mat, CV_32F);
        FindCornersOpenCV(image_mat, corners_mat);
        return ToImage(corners_mat);
    }

private:
    // Converts an Image<Float> to an OpenCV matrix, useful for displaying by OpenCV.
    template<typename P>
    void ToMat(const Image<P>& src, cv::Mat& dest, int type) {
        dest.create(src.height(), src.width(), type);
        std::memmove(dest.data, src.data(), src.stride() * src.height());
    }

    // Converts an Image<Float> to an OpenCV matrix, useful for displaying by OpenCV.
    Image<float> ToImage(cv::Mat& dest) {
        return Image<float>(dest.data, dest.cols, dest.rows, dest.step[0]);
    }

    // Non-Maximal suppresion with thresholding implemented using standard OpenCV components
    void NonMaxSuppression(cv::Mat src, cv::Mat& dest, int block_size, double threshold) {
        if (src.type() != CV_32F) throw std::invalid_argument("src must be float image");
        dest.create(src.rows, src.cols, CV_32F);
        const auto half_block = block_size / 2;
        for (auto row = 0; row < src.rows; ++row) {
            auto src_row = src.ptr<float>(row);
            auto dest_row = dest.ptr<float>(row);
            for (auto col = 0; col < src.cols; ++col) {
                const auto src_pixel = src_row[col];
                if (src_pixel < threshold) {
                    dest_row[col] = 0.0f;
                    continue;
                }

                auto dest_pixel = src_pixel;
                for (auto w_row = row - half_block; dest_pixel > 0.0 && w_row <= row + half_block; ++w_row) {
                    if (w_row < 0) continue;
                    if (w_row >= src.rows) continue;
                    auto window_row = src.ptr<float>(w_row);
                    for (auto w_col = col - half_block; w_col <= col + half_block; ++w_col) {
                        if (w_col < 0) continue;
                        if (w_col >= src.cols) continue;
                        const auto window_pixel = window_row[w_col];
                        if (window_pixel > dest_pixel) {
                            dest_pixel = 0.0f;
                            break;
                        }
                    }
                }
                dest_row[col] = dest_pixel;
            }
        }
    }

    // Harris corener detection implemented using standard OpenCV components
    void FindCornersOpenCV(const cv::Mat& float_image, cv::Mat& corners_mat) {
        cv::Mat harris_img;
        cv::Mat bw_image;
        cv::cornerHarris(float_image, harris_img, structure_size_, smoothing_size_, k_);
        double min, max;
        cv::minMaxLoc(harris_img, &min, &max);
        NonMaxSuppression(harris_img, corners_mat, suppression_size_, min + threshold_ratio_ * (max - min));
    }
};
}

