#include <functional>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "harris_opencv.h"

#include "image.h"
#include "image_conversion.h"
#include "harris_corner_detector.h"

const cv::String keys =
    "{help h usage ? |      | print this message                                                                                }"
    "{@input         |      | input image or video                                                                              }"
    "{o output       |      | outputs a version of the input with markers on each corner (.png or .m4v formats are supported)   }"
    "{s show         |      | displays a window containing a version of the input with markers on each corner                   }"
    "{smoothing      |    5 | The size (in pixels) of the gaussian smoothing kernel. This must be an odd number                 }"
    "{structure      |    9 | The size (in pixels) of the window used to define the structure tensor of each pixel              }"
    "{suppression    |    9 | The size (in pixels) of the non-maximum suppression window                                        }"
    "{k harris_k     | 0.04 | The value of the Harris free parameter                                                            }"
    "{threshold      |  0.5 | The Harris response suppression threshold defined as a ratio of the maximum response value        }"
    "{opencv         |      | Use the OpenCV algorithm rather than the pure C++ method                                          }"
    ;

using namespace harris;

// Converts an Image<Float> to an OpenCV matrix, useful for displaying by OpenCV.
void ToMat(const Image<float>& src, cv::Mat& dest) {
    dest.create(src.height(), src.width(), CV_32F);
    std::memmove(dest.data, src.data(), src.stride() * src.height());
}

// Measures the time taken by a lambda function in ms
double MeasureTimeMs(std::function<void()> func) {
        // Start the timer
        auto start = std::chrono::high_resolution_clock::now();

        // Run the function
        func();

        // Stop the timer and record the time
        auto end = std::chrono::high_resolution_clock::now();
        auto time_in_ms = static_cast<double>((end - start).count()) / 1e6;

        return time_in_ms;
}

// Runs the pure C++ Harris corner detector given an CV_8UC4 OpenCV matrix image
void RunHarrisCpp(const cv::Mat& input_mat, cv::Mat& corner_mat, int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) {
    if(smoothing_size <= 0 || smoothing_size % 2 == 0) throw std::invalid_argument("sobel_size must be a positive odd number");
    if(structure_size <= 0 || structure_size % 2 == 0) throw std::invalid_argument("structure_size must be a positive odd number");
    if(suppression_size <= 0 || suppression_size % 2 == 0) throw std::invalid_argument("suppression_size must be a positive odd number");
    if(harris_k <= 0) throw std::invalid_argument("harris_k must be positive");
    if(threshold_ratio < 0 || threshold_ratio > 1) throw std::invalid_argument("threshold_ratio must be between 0 and 1");
    if (input_mat.type() != CV_8UC4) throw std::invalid_argument("This method only works with ARGB32 images");

    const auto color_img = Image<Argb32>(input_mat.data, input_mat.cols, input_mat.rows, input_mat.step[0]);
    const auto float_img = ToFloat(color_img);
    const auto harris_img = HarrisCorners(float_img, smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size);
    ToMat(harris_img, corner_mat);
}

// Takes a Harris corner matrix and puts rectangles at each point on the corresponding image matrix
void HighlightCorners(cv::Mat corners, cv::Mat image, int block_size = 5) {
    const auto half_block = block_size / 2;
    for (auto row = 0; row < corners.rows; ++row) {
        auto corner_row = corners.ptr<float>(row);
        for (auto col = 0; col < corners.cols; ++col) {
            if (corner_row[col] <= 0.0f) continue;
            cv::rectangle(image, cv::Rect(col - half_block, row - half_block, block_size, block_size), cv::Scalar(0, 0, 255), 2);
        }
    }
}

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Harris Corner Detector Demo");

    // Check for command line errors or --help param
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Extract command line parameters
    auto show_enabled = parser.has("show");
    auto use_opencv = parser.has("opencv");
    auto smoothing_size = parser.get<int>("smoothing");
    auto structure_size = parser.get<int>("structure");
    auto suppression_size = parser.get<int>("suppression");
    auto harris_k = parser.get<float>("harris_k");
    auto threshold_ratio = parser.get<float>("threshold");

    // Read the input image
    auto input_file = parser.get<cv::String>("@input");
    auto input_image = cv::imread(input_file, cv::IMREAD_UNCHANGED);
    cv::VideoCapture input_video;

    // Check if the input is a single image or video
    auto is_image_input = !input_image.empty();
    auto is_video_input = !is_image_input && input_video.open(input_file);

    // If image can't be loaded, exit now
    if (!is_image_input && !is_video_input) {
        std::cerr << "Failed to load input file " << input_file << std::endl;
        return 2;
    }

    auto total_time_ms = 0.0;
    auto num_frames = 0.0;

    // Loop through each image, run Harris corner detection and display the output (if set)
    auto has_image = is_image_input || input_video.read(input_image);
    while(has_image) {
        // Convert image to Argb32 (the only supported format for RunHarris)
        if (input_image.type() == CV_8UC3) {
            cv::cvtColor(input_image, input_image, cv::COLOR_RGB2RGBA);
        }

        // Run Harris corner detection
        cv::Mat corners;
        const auto time_in_ms = MeasureTimeMs([&]() {
            if (use_opencv) {
                RunHarrisOpenCV(input_image, corners, smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size);
            } else {
                RunHarrisCpp(input_image, corners, smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size);
            }
        });

        // Record the time
        total_time_ms += time_in_ms;
        ++num_frames;

        // If show is enabled, display the image
        if (show_enabled) {
            HighlightCorners(corners, input_image);
            cv::imshow("Corners", input_image);
            cv::waitKey(1);
        }

        // If this is a video, move to the next frame
        has_image = is_video_input && input_video.read(input_image);
    }

    // Print the statistics for the 
    std::cout << num_frames << " frames were processed with an average processing time of " << total_time_ms / num_frames << " ms\n";

    // If show is enabled, pause on the last image.
    if (show_enabled) cv::waitKey(0);
    return 0;
}
