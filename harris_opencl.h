#pragma once
// Harris corner detection algorithm implemented using OpenCL

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "harris_base.h"
#include "filter_2d.h"

namespace harris {

class HarrisOpenCL : public HarrisBase {
public:

    HarrisOpenCL(int platform_num = 0, int device_num = -1, int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size),
        gaussian_(GaussianKernel(smoothing_size)) {

        cl::Platform::get(&platforms_);
        std::cout << "Found " << platforms_.size() << " platform(s)" << std::endl;
        for (const auto& platform : platforms_) {
            std::cout << "\t" <<  platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }

        platforms_[platform_num].getDevices(CL_DEVICE_TYPE_ALL, &devices_);

        std::cout << "Found " << devices_.size() << " devices(s)" << std::endl;
        for (const auto& device : devices_) {
            std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }

        // If device_num is default choose either the first GPU device or the first device if no GPU is available.
        if (device_num < 0) {
            for (auto i = 0; i < devices_.size(); ++i) {
                auto device_type = devices_[i].getInfo<CL_DEVICE_TYPE>();
                if (device_type == CL_DEVICE_TYPE_GPU) {
                    device_num = i;
                    break;
                }
            }

            if (device_num < 0) device_num = 0;
        }

        context_ = cl::Context(devices_[device_num]);

        // GPU and CPU types use different single channel image formats (CL_R or CL_Rx) so I need to figure out which one to use.
        // TODO: It might be better to just switch these all to float*
        std::vector<cl::ImageFormat> supportedFormats;
        context_.getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, &supportedFormats);
        std::cout << "Found " << supportedFormats.size() << " supported format(s)" << std::endl;
        cl::ImageFormat float_format;
        for (const auto& format : supportedFormats) {
            if (format.image_channel_data_type == CL_FLOAT && (format.image_channel_order == CL_R || format.image_channel_order == CL_Rx)) float_format_ = format;
        }


        program_ = CreateProgram("harris.cl", context_);
        BuildProgram(program_, std::vector<cl::Device>({ devices_[device_num] }));
        queue_ = cl::CommandQueue(context_, devices_[device_num]);
    }

    // Rule of five: Neither movable nor copyable
    HarrisOpenCL(const HarrisOpenCL&) = delete;
    HarrisOpenCL(HarrisOpenCL&&) = delete;
    HarrisOpenCL& operator=(const HarrisOpenCL&) = delete;
    HarrisOpenCL& operator=(HarrisOpenCL&&) = delete;
    ~HarrisOpenCL() override = default;

    // Runs the OpenCL Harris corner detector
    Image<float> FindCorners(const Image<Argb32>& image) override {
        const auto width = static_cast<size_t>(image.width());
        const auto height = static_cast<size_t>(image.height());

        try
        {
            cl::Image2D argb_image(
                context_, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                cl::ImageFormat{ CL_RGBA, CL_UNORM_INT8 },
                width,
                height,
                image.stride(),
                const_cast<uint8_t*>(image.data()));

            cl::Kernel argb32_to_float_kernel(program_, "Argb32ToFloat");

            cl::Image2D float_image(
                context_, 
                CL_MEM_READ_WRITE,
                float_format_,
                width,
                height);

            argb32_to_float_kernel.setArg(0, argb_image);
            argb32_to_float_kernel.setArg(1, float_image);

            cl::Event argb32_to_float_complete;
            queue_.enqueueNDRangeKernel(
                argb32_to_float_kernel,
                cl::NullRange,
                cl::NDRange{ width, height },
                cl::NullRange,
                nullptr,
                &argb32_to_float_complete);

            cl::Kernel smoothing_kernel(program_, "Smoothing");

            cl::Buffer gaussian_buffer(
                context_, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                sizeof(float) * gaussian_.width() * gaussian_.height(), 
                gaussian_.data());

            cl::Image2D smooth_image(
                context_, 
                CL_MEM_READ_WRITE, 
                float_format_,
                width,
                height);

            smoothing_kernel.setArg(0, float_image);
            smoothing_kernel.setArg(1, gaussian_buffer);
            smoothing_kernel.setArg(2, smooth_image);

            cl::Event smoothing_complete;
            std::vector<cl::Event> smoothing_prereqs({ argb32_to_float_complete });
            queue_.enqueueNDRangeKernel(
                smoothing_kernel,
                cl::NullRange,
                cl::NDRange{ width, height},
                cl::NullRange,
                &smoothing_prereqs,
                &smoothing_complete);

            cl::Kernel diff_x_kernel(program_, "DiffX");

            cl::Image2D i_x_image(
                context_, 
                CL_MEM_READ_WRITE, 
                float_format_,
                width,
                height);

            diff_x_kernel.setArg(0, smooth_image);
            diff_x_kernel.setArg(1, i_x_image);

            cl::Event diff_x_complete;
            std::vector<cl::Event> diff_x_prereqs({ smoothing_complete });
            queue_.enqueueNDRangeKernel(
                diff_x_kernel,
                cl::NullRange,
                cl::NDRange{ width, height },
                cl::NullRange,
                &diff_x_prereqs,
                &diff_x_complete);

            cl::Kernel diff_y_kernel(program_, "DiffY");

            cl::Image2D i_y_image(
                context_, 
                CL_MEM_READ_WRITE, 
                float_format_,
                width,
                height);

            diff_y_kernel.setArg(0, smooth_image);
            diff_y_kernel.setArg(1, i_y_image);

            cl::Event diff_y_complete;
            std::vector<cl::Event> diff_y_prereqs({ smoothing_complete });
            queue_.enqueueNDRangeKernel(
                diff_y_kernel,
                cl::NullRange,
                cl::NDRange{ width, height },
                cl::NullRange,
                &diff_y_prereqs,
                &diff_y_complete);

            cl::Kernel structure_kernel(program_, "Structure");

            cl::Image2D structure_image(
                context_, 
                CL_MEM_READ_WRITE, 
                cl::ImageFormat{ CL_RGBA, CL_FLOAT },
                width,
                height);

            structure_kernel.setArg(0, i_x_image);
            structure_kernel.setArg(1, i_y_image);
            structure_kernel.setArg(2, structure_image);

            cl::Event structure_complete;
            std::vector<cl::Event> structure_prereqs({ diff_x_complete, diff_y_complete });
            queue_.enqueueNDRangeKernel(
                structure_kernel,
                cl::NullRange,
                cl::NDRange{ width, height },
                cl::NullRange,
                &structure_prereqs,
                &structure_complete);

            cl::Kernel response_kernel(program_, "Response");

            cl::Image2D response_image(
                context_, 
                CL_MEM_READ_WRITE, 
                float_format_,
                width,
                height);

            response_kernel.setArg(0, structure_image);
            response_kernel.setArg(1, response_image);

            cl::Event response_complete;
            std::vector<cl::Event> response_prereqs({ structure_complete });
            queue_.enqueueNDRangeKernel(
                response_kernel,
                cl::NullRange,
                cl::NDRange{ width, height },
                cl::NullRange,
                &response_prereqs,
                &response_complete);

            cl::Kernel row_max_kernel(program_, "RowMax");

            cl::Buffer row_max_buffer(
                context_, 
                CL_MEM_READ_WRITE, 
                sizeof(float) * height);

            row_max_kernel.setArg(0, response_image);
            row_max_kernel.setArg(1, row_max_buffer);

            cl::Event row_max_complete;
            std::vector<cl::Event> row_max_prereqs({ response_complete });
            queue_.enqueueNDRangeKernel(
                row_max_kernel,
                cl::NullRange,
                cl::NDRange{ height },
                cl::NullRange,
                &row_max_prereqs,
                &row_max_complete);

            cl::Kernel max_kernel(program_, "Max");

            max_kernel.setArg(0, height);
            max_kernel.setArg(1, row_max_buffer);

            cl::Event max_complete;
            std::vector<cl::Event> max_prereqs({ row_max_complete });
            queue_.enqueueTask(
                max_kernel,
                &max_prereqs,
                &max_complete
            );

            cl::Kernel suppression_kernel(program_, "NonMaxSuppression");

            cl::Image2D corner_image(
                context_,
                CL_MEM_READ_WRITE,
                float_format_,
                width,
                height);

            suppression_kernel.setArg(0, response_image);
            suppression_kernel.setArg(1, row_max_buffer);
            suppression_kernel.setArg(2, corner_image);

            cl::Event suppression_complete;
            std::vector<cl::Event> suppression_prereqs({ response_complete });
            queue_.enqueueNDRangeKernel(
                suppression_kernel,
                cl::NullRange,
                cl::NDRange{ width, height },
                cl::NullRange,
                &suppression_prereqs,
                &suppression_complete);

            Image<float> corners(width, height);
            std::vector<cl::Event> read_prereqs({ suppression_complete });
            queue_.enqueueReadImage(
                corner_image,
                CL_TRUE,
                sizes({}),
                sizes({ width, height, 1 }),
                corners.stride(),
                0,
                corners.data(),
                &read_prereqs);

            return corners;
        }
        catch(const cl::Error& e)
        {
            std::cerr << e.what() << ": " << e.err() << '\n';
            throw;
        }
    }

private:
    std::vector<cl::Device> devices_;
    std::vector<cl::Platform> platforms_;
    cl::Context context_;
    cl::Program program_;
    cl::CommandQueue queue_;
    cl::ImageFormat float_format_;
    FilterKernel gaussian_;

    cl::Program CreateProgram(const std::string& source_file, const cl::Context& context)
    {
        std::ifstream in(source_file);
        std::stringstream source_stream;
        source_stream << in.rdbuf();
        auto source = source_stream.str();

        return cl::Program(context, source);
    }

    void BuildProgram(cl::Program& program, const std::vector<cl::Device>& devices) {
        std::stringstream options_stream;
        options_stream << " -D HALF_SMOOTHING=" << smoothing_size_ / 2; 
        options_stream << " -D HALF_STRUCTURE=" << structure_size_ / 2; 
        options_stream << " -D HALF_SUPPRESSION=" << suppression_size_ / 2; 
        options_stream << " -D HARRIS_K=" << k_;
        options_stream << " -D THRESHOLD_RATIO=" << threshold_ratio_;
        const auto options = options_stream.str();

        
        try
        {
            program.build(devices, options.c_str());
        }
        catch(const cl::Error& e)
        {
            if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
                // If program failed to build list the build log on stderror.
                std::cerr << "Program build failed. Build log:\n\n";

                for(auto device : devices) {
                    std::cerr << device.getInfo<CL_DEVICE_NAME>() << "\n\n";
                    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
                }

                throw;
            }
        }
    }

    // cl.hpp has no usueful way to make a size_t<3> even though it uses them all over the place. **sigh**
    cl::size_t<3> sizes(std::initializer_list<size_t> size_values) {
        cl::size_t<3> result;
        auto i = 0;
        for(auto s : size_values) {
            result[i] = s;
            ++i;
        }

        return result;
    }

};
}

