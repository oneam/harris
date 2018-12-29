#pragma once
// This is an OpenCV implementation of the algorithm used as a reference for other implementations.

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
            for (auto w_row = row - half_block; w_row <= row + half_block; ++w_row) {
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
void RunHarrisOpenCV(const cv::Mat& input_image, cv::Mat& corner_mat, int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) {
    if(smoothing_size <= 0 || smoothing_size % 2 == 0) throw std::invalid_argument("smoothing_size must be a positive odd number");
    if(structure_size <= 0 || structure_size % 2 == 0) throw std::invalid_argument("structure_size must be a positive odd number");
    if(suppression_size <= 0 || suppression_size % 2 == 0) throw std::invalid_argument("suppression_size must be a positive odd number");
    if(harris_k <= 0) throw std::invalid_argument("harris_k must be positive");
    if(threshold_ratio < 0 || threshold_ratio > 1) throw std::invalid_argument("threshold_ratio must be between 0 and 1");

    cv::Mat harris_img;
    cv::Mat bw_image;
    cv::Mat float_image;
    cv::cvtColor(input_image, bw_image, cv::COLOR_RGB2GRAY);
    bw_image.convertTo(float_image, CV_32F, 1.0/255.0);
    cv::cornerHarris(float_image, harris_img, structure_size, smoothing_size, harris_k);
    double min, max;
    cv::minMaxLoc(harris_img, &min, &max);
    NonMaxSuppression(harris_img, corner_mat, suppression_size, min + threshold_ratio * (max - min));
}

