#include <functional>
#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include "harris_cpp.h"
#include "harris_opencv.h"
#include "harris_opencl.h"

const cv::String keys =
    "{help h usage ? |      | Print this message                                                                                            }"
    "{@input         |      | Input image or video                                                                                          }"
    "{o output       |      | Outputs a version of the input with markers on each corner (use a file that ends with .m4v to output a video) }"
    "{s show         |      | Displays a window containing a version of the input with markers on each corner                               }"
    "{b benchmark    |      | Prints the rendering time for each frame as it's converted                                                    }"
    "{smoothing      |    5 | The size (in pixels) of the gaussian smoothing kernel. This must be an odd number                             }"
    "{structure      |    5 | The size (in pixels) of the window used to define the structure tensor of each pixel                          }"
    "{suppression    |    9 | The size (in pixels) of the non-maximum suppression window                                                    }"
    "{k harris_k     | 0.04 | The value of the Harris free parameter                                                                        }"
    "{threshold      |  0.5 | The Harris response suppression threshold defined as a ratio of the maximum response value                    }"
    "{opencv         |      | Use the OpenCV algorithm rather than the pure C++ method                                                      }"
    "{opencl         |      | Use the OpenCL algorithm rather than the pure C++ method                                                      }"
    "{cl-platform    |    0 | The index of the platform to use when runnning OpenCL algorithm                                               }"
    "{cl-device      |   -1 | The index of the device to use when runnning OpenCL algorithm (use -1 to select first GPU if available)       }"
    ;

using namespace harris;

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

// Takes a Harris corner matrix and puts rectangles at each point on the corresponding image matrix
void HighlightCorners(Image<float> corners, cv::Mat image, int block_size = 5) {
    const auto half_block = block_size / 2;
    for (auto row = 0; row < corners.height(); ++row) {
        auto corner_row = corners.RowPtr(row);
        for (auto col = 0; col < corners.width(); ++col) {
            if (corner_row[col] <= 0.0f) continue;
            cv::rectangle(image, cv::Rect(col - half_block, row - half_block, block_size, block_size), cv::Scalar(0, 0, 255), 1);
        }
    }
}

// Returns true if a string ends with a given substring
inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Harris Corner Detector Demo");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Extract command line parameters
    auto input_file = parser.get<cv::String>("@input");
    auto show_enabled = parser.has("show");
    auto benchmark_enabled = parser.has("benchmark");
    auto output_enabled = parser.has("output");
    auto output_file = output_enabled ? parser.get<cv::String>("output") : cv::String();
    auto use_opencv = parser.has("opencv");
    auto use_opencl = parser.has("opencl");
    auto smoothing_size = parser.get<int>("smoothing");
    auto structure_size = parser.get<int>("structure");
    auto suppression_size = parser.get<int>("suppression");
    auto harris_k = parser.get<float>("harris_k");
    auto threshold_ratio = parser.get<float>("threshold");
    auto cl_platform = parser.get<int>("cl-platform");
    auto cl_device = parser.get<int>("cl-device");

    // Check for command line errors or --help param
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Read the input image
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

    // Create the appropriate harris algorithm
    std::shared_ptr<HarrisBase> harris;
    if (use_opencv) {
        harris = std::make_shared<HarrisOpenCV>(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size);
    } else if (use_opencl) {
        harris = std::make_shared<HarrisOpenCL>(cl_platform, cl_device, smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size);
    } else {
        harris = std::make_shared<HarrisCpp>(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size);
    }

    // Create the output video if requested
    auto test = output_file.rfind(".m4v");
    bool is_video_output = output_file.rfind(".m4v") == output_file.length() - 4;
    cv::VideoWriter output_video;
    if (is_video_output) {
        const auto width = is_video_input ? static_cast<int>(input_video.get(cv::CAP_PROP_FRAME_WIDTH)) : input_image.cols;
        const auto height = is_video_input ? static_cast<int>(input_video.get(cv::CAP_PROP_FRAME_HEIGHT)) : input_image.rows;
        const auto fps = is_video_input ? input_video.get(cv::CAP_PROP_FPS) : 29.97;
        if (!output_video.open(output_file, CV_FOURCC('a', 'v', 'c', '1'), fps, cv::Size(width, height))) { // AVC1 is the fourcc code for h.264
            std::cerr << "Failed to load output file " << output_file << std::endl;
            return 3;
        }
    }

    // Placeholders for timing information
    auto total_time_ms = 0.0;
    auto num_frames = 0.0;

    // Loop through each image, run Harris corner detection and display the output (if set)
    auto has_image = is_image_input || input_video.read(input_image);
    while(has_image) {
        // Convert image to Argb32 (the only supported format for RunHarris)
        if (input_image.type() == CV_8UC3) {
            cv::cvtColor(input_image, input_image, cv::COLOR_BGR2BGRA);
        }

        // Run Harris corner detection
        const Image<Argb32> input(input_image.data, input_image.cols, input_image.rows, input_image.step[0]);
        Image<float> corners;
        const auto time_in_ms = MeasureTimeMs([&]() { corners = harris->FindCorners(input); });

        // Record the time
        total_time_ms += time_in_ms;
        ++num_frames;

        // If we are going to output a
        if (show_enabled || output_enabled) {
            HighlightCorners(corners, input_image);
        }

        if (show_enabled) {
            cv::imshow("Corners", input_image);
            cv::waitKey(1);
        }

        if (!show_enabled && !benchmark_enabled) {
            // Put a dot on screen to give an indication that something is happening
            std::cout << "." << std::flush;
        }

        if (benchmark_enabled) {
            std::cout << time_in_ms << "ms" << std::endl;
        }

        if (output_enabled && is_video_output) {
            cv::cvtColor(input_image, input_image, cv::COLOR_BGRA2BGR);
            output_video.write(input_image);
        }

        // If this is a video, move to the next frame
        has_image = is_video_input && input_video.read(input_image);
    }

    // If this is not a video, just output the last frame
    if (output_enabled && !is_video_output) {
        cv::cvtColor(input_image, input_image, cv::COLOR_BGRA2BGR);
        cv::imwrite(output_file, input_image);
    }

    // Print the statistics for the 
    std::cout << "\n" << num_frames << " frames were processed in " << total_time_ms / 1e3 << " seconds with an average processing time of " << total_time_ms / num_frames << " ms\n";

    // If show is enabled, pause on the last image.
    if (show_enabled) {
        std::cout << "Highlight the image preview and press any key to exit..." << std::endl;
        cv::waitKey(0);
    }

    return 0;
}
