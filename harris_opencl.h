#pragma once
// Harris corner detection algorithm implemented using OpenCL

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>


#include "cl.hpp"
#include "harris_base.h"
#include "filter_2d.h"

namespace harris {

class HarrisOpenCL : public HarrisBase {
public:

    HarrisOpenCL(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size),
        gaussian_(GaussianKernel(smoothing_size)) {

        platform_ids_ = GetPlatformIds();
        std::cout << "Found " << platform_ids_.size() << " platform(s)" << std::endl;
        for (auto platform_id : platform_ids_) {
            std::cout << "\t" <<  GetPlatformName(platform_id) << std::endl;
        }
 
        device_ids_ = GetDeviceIds(platform_ids_[0]);
        std::cout << "Found " << device_ids_.size() << " devices(s)" << std::endl;
        for (auto device_id : device_ids_) {
            std::cout << "\t" << GetDeviceName(device_id) << std::endl;
        }
        
        context_ = CreateContext(platform_ids_[0], device_ids_);
        std::cout << "Context created" << std::endl;

        program_ = CreateProgram("harris.cl", context_);
        std::cout << "Program created" << std::endl;

        BuildProgram(program_, device_ids_);
        std::cout << "Program built" << std::endl;
    }

    // Rule of five: Neither movable nor copyable
    HarrisOpenCL(const HarrisOpenCL&) = delete;
    HarrisOpenCL(HarrisOpenCL&&) = delete;
    HarrisOpenCL& operator=(const HarrisOpenCL&) = delete;
    HarrisOpenCL& operator=(HarrisOpenCL&&) = delete;

    ~HarrisOpenCL() override {
        clReleaseProgram(program_);
        clReleaseContext(context_);
    }

    // Runs the OpenCL Harris corner detector
    Image<float> FindCorners(const Image<float>& image) override {
        cl_int error;
        cl_command_queue command_queue;
        cl_kernel smoothing_kernel, diff_x_kernel, diff_y_kernel, structure_kernel, response_kernel, row_max_kernel, max_kernel, suppression_kernel;
        cl_mem src_image, gaussian_buffer, smooth_image, i_x_image, i_y_image, structure_image, response_image, row_max_buffer, corner_image;
        cl_event smoothing_complete, diff_x_complete, diff_y_complete, structure_complete, response_complete, row_max_complete, max_complete, suppression_complete;

        try
        {
            command_queue = clCreateCommandQueue(context_, device_ids_[1], 0, &error);
            CheckError(error, "Failed to create command queue");

            smoothing_kernel = clCreateKernel(program_, "Smoothing", &error);
            CheckError(error, "Failed to create smoothing kernel");

            static const cl_image_format src_image_format = { CL_R, CL_FLOAT };
            const cl_image_desc src_image_desc = { CL_MEM_OBJECT_IMAGE2D, static_cast<size_t>(image.width()), static_cast<size_t>(image.height()), 0, 0, image.stride(), 0, 0, 0, nullptr };
            src_image = clCreateImage(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &src_image_format, &src_image_desc, const_cast<uint8_t*>(image.data()), &error);
            CheckError(error, "Failed to create input image");

            cl_mem gaussian_buffer = clCreateBuffer(
                context_, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                sizeof(float) * gaussian_.width() * gaussian_.height(), 
                const_cast<float*>(gaussian_.data()), 
                &error);
            CheckError(error, "Failed to create smoothing kernel buffer");

            static const cl_image_format internal_image_format = { CL_R, CL_FLOAT };
            const cl_image_desc internal_image_desc = { CL_MEM_OBJECT_IMAGE2D, static_cast<size_t>(image.width()), static_cast<size_t>(image.height()), 0, 0, 0, 0, 0, 0, nullptr };
            smooth_image = clCreateImage(context_, CL_MEM_READ_WRITE, &internal_image_format, &internal_image_desc, nullptr, &error);
            CheckError(error, "Failed to create smooth image");

            clSetKernelArg(smoothing_kernel, 0, sizeof(cl_mem), &src_image);
            clSetKernelArg(smoothing_kernel, 1, sizeof(cl_mem), &gaussian_buffer);
            clSetKernelArg(smoothing_kernel, 2, sizeof(cl_mem), &smooth_image);

            std::size_t offset[3] = { 0 };
            std::size_t size[3] = { static_cast<size_t>(image.width()), static_cast<size_t>(image.height()), 1 };
            error = clEnqueueNDRangeKernel(command_queue, smoothing_kernel, 2, offset, size, nullptr, 0, nullptr, &smoothing_complete);
            CheckError(error, "Failed to start Smoothing kernel");

            diff_x_kernel = clCreateKernel(program_, "DiffX", &error);
            CheckError(error, "Failed to create DiffX kernel");

            i_x_image = clCreateImage(context_, CL_MEM_READ_WRITE, &internal_image_format, &internal_image_desc, nullptr, &error);
            CheckError(error, "Failed to create i_x image");

            clSetKernelArg(diff_x_kernel, 0, sizeof(cl_mem), &smooth_image);
            clSetKernelArg(diff_x_kernel, 1, sizeof(cl_mem), &i_x_image);

            const cl_event diff_x_prereqs[] = { smoothing_complete };
            error = clEnqueueNDRangeKernel(command_queue, diff_x_kernel, 2, offset, size, nullptr, 1, diff_x_prereqs, &diff_x_complete);
            CheckError(error, "Failed to start DiffX kernel");

            diff_y_kernel = clCreateKernel(program_, "DiffY", &error);
            CheckError(error, "Failed to create DiffY kernel");

            i_y_image = clCreateImage(context_, CL_MEM_READ_WRITE, &internal_image_format, &internal_image_desc, nullptr, &error);
            CheckError(error, "Failed to create i_y image");

            clSetKernelArg(diff_y_kernel, 0, sizeof(cl_mem), &smooth_image);
            clSetKernelArg(diff_y_kernel, 1, sizeof(cl_mem), &i_y_image);

            const cl_event diff_y_prereqs[] = { smoothing_complete };
            error = clEnqueueNDRangeKernel(command_queue, diff_y_kernel, 2, offset, size, nullptr, 1, diff_y_prereqs, &diff_y_complete);
            CheckError(error, "Failed to start DiffY kernel");

            structure_kernel = clCreateKernel(program_, "Structure", &error);
            CheckError(error, "Failed to create Structure kernel");

            static const cl_image_format structure_tensor_format = { CL_RGBA, CL_FLOAT };
            const cl_image_desc structure_tensor_desc = { CL_MEM_OBJECT_IMAGE2D, static_cast<size_t>(image.width()), static_cast<size_t>(image.height()), 0, 0, 0, 0, 0, 0, nullptr };
            structure_image = clCreateImage(context_, CL_MEM_READ_WRITE, &structure_tensor_format, &structure_tensor_desc, nullptr, &error);
            CheckError(error, "Failed to create structure image");

            clSetKernelArg(structure_kernel, 0, sizeof(cl_mem), &i_x_image);
            clSetKernelArg(structure_kernel, 1, sizeof(cl_mem), &i_y_image);
            clSetKernelArg(structure_kernel, 2, sizeof(cl_mem), &structure_image);

            const cl_event structure_prereqs[] = { diff_x_complete, diff_y_complete };
            error = clEnqueueNDRangeKernel(command_queue, structure_kernel, 2, offset, size, nullptr, 2, structure_prereqs, &structure_complete);
            CheckError(error, "Failed to start Structure kernel");

            response_kernel = clCreateKernel(program_, "Response", &error);
            CheckError(error, "Failed to create Response kernel");

            response_image = clCreateImage(context_, CL_MEM_READ_WRITE, &internal_image_format, &internal_image_desc, nullptr, &error);
            CheckError(error, "Failed to create response image");

            clSetKernelArg(response_kernel, 0, sizeof(cl_mem), &structure_image);
            clSetKernelArg(response_kernel, 1, sizeof(cl_mem), &response_image);

            const cl_event response_prereqs[] = { structure_complete };
            error = clEnqueueNDRangeKernel(command_queue, response_kernel, 2, offset, size, nullptr, 1, response_prereqs, &response_complete);
            CheckError(error, "Failed to start Response kernel");

            row_max_kernel = clCreateKernel(program_, "RowMax", &error);
            CheckError(error, "Failed to create RowMax kernel");

            cl_mem row_max_buffer = clCreateBuffer(
                context_,
                CL_MEM_READ_WRITE,
                sizeof(float) * image.height(),
                nullptr, 
                &error);
            CheckError(error, "Failed to create row max buffer");

            clSetKernelArg(row_max_kernel, 0, sizeof(cl_mem), &response_image);
            clSetKernelArg(row_max_kernel, 1, sizeof(cl_mem), &row_max_buffer);

            std::size_t row_max_offset[3] = { 0 };
            std::size_t row_max_size[3] = { static_cast<size_t>(image.height()), 1, 1 };
            const cl_event row_max_prereqs[] = { response_complete };
            error = clEnqueueNDRangeKernel(command_queue, row_max_kernel, 2, row_max_offset, row_max_size, nullptr, 1, row_max_prereqs, &row_max_complete);
            CheckError(error, "Failed to start RowMax kernel");

            auto max_kernel = clCreateKernel(program_, "Max", &error);
            CheckError(error, "Failed to create Max kernel");

            int height = image.height();
            clSetKernelArg(max_kernel, 0, sizeof(int), &height);
            clSetKernelArg(max_kernel, 1, sizeof(cl_mem), &row_max_buffer);

            const cl_event max_prereqs[] = { row_max_complete };
            error = clEnqueueTask (command_queue, max_kernel, 1, max_prereqs, &max_complete);
            CheckError(error, "Failed to start Max kernel");

            suppression_kernel = clCreateKernel(program_, "NonMaxSuppression", &error);
            CheckError(error, "Failed to create NonMaxSuppression kernel");

            corner_image = clCreateImage(context_, CL_MEM_WRITE_ONLY, &internal_image_format, &internal_image_desc, nullptr, &error);
            CheckError(error, "Failed to create suppression image");

            clSetKernelArg(suppression_kernel, 0, sizeof(cl_mem), &response_image);
            clSetKernelArg(suppression_kernel, 1, sizeof(cl_mem), &row_max_buffer);
            clSetKernelArg(suppression_kernel, 2, sizeof(cl_mem), &corner_image);

            const cl_event suppression_prereqs[] = { response_complete };
            clEnqueueNDRangeKernel(command_queue, suppression_kernel, 2, offset, size, nullptr, 1, suppression_prereqs, &suppression_complete);

            Image<float> corners(image.width(), image.height());
            std::size_t origin[3] = { 0 };
            std::size_t region[3] = { static_cast<size_t>(corners.width()), static_cast<size_t>(corners.height()), 1 };
            const cl_event read_corners_prereqs[] = { suppression_complete };
            error = clEnqueueReadImage(command_queue, corner_image, CL_TRUE, origin, region, corners.stride(), 0, corners.data(), 1, read_corners_prereqs, nullptr);
            CheckError(error, "Failed to read corner image");

            clReleaseCommandQueue(command_queue);
            clReleaseKernel(smoothing_kernel);
            clReleaseKernel(diff_x_kernel);
            clReleaseKernel(diff_y_kernel);
            clReleaseKernel(structure_kernel);
            clReleaseKernel(response_kernel);
            clReleaseKernel(row_max_kernel);
            clReleaseKernel(max_kernel);
            clReleaseKernel(suppression_kernel);
            clReleaseMemObject(src_image);
            clReleaseMemObject(gaussian_buffer);
            clReleaseMemObject(smooth_image);
            clReleaseMemObject(i_x_image);
            clReleaseMemObject(i_y_image);
            clReleaseMemObject(structure_image);
            clReleaseMemObject(response_image);
            clReleaseMemObject(row_max_buffer);
            clReleaseMemObject(corner_image);
            clReleaseEvent(smoothing_complete);
            clReleaseEvent(diff_x_complete);
            clReleaseEvent(diff_y_complete);
            clReleaseEvent(structure_complete);
            clReleaseEvent(response_complete);
            clReleaseEvent(row_max_complete);
            clReleaseEvent(max_complete);
            clReleaseEvent(suppression_complete);

            return corners;
        }
        catch(const std::exception& e)
        {
            std::cerr << "An exception was throw while finding corners: " << e.what() << '\n';
        }
        catch(...)
        {
            std::cerr  << "An unknown exception type was throw while finding corners" << '\n';
        }

        clReleaseCommandQueue(command_queue);
        clReleaseKernel(smoothing_kernel);
        clReleaseKernel(diff_x_kernel);
        clReleaseKernel(diff_y_kernel);
        clReleaseKernel(structure_kernel);
        clReleaseKernel(response_kernel);
        clReleaseKernel(max_kernel);
        clReleaseKernel(suppression_kernel);
        clReleaseMemObject(src_image);
        clReleaseMemObject(gaussian_buffer);
        clReleaseMemObject(smooth_image);
        clReleaseMemObject(i_x_image);
        clReleaseMemObject(i_y_image);
        clReleaseMemObject(structure_image);
        clReleaseMemObject(response_image);
        clReleaseMemObject(corner_image);
        clReleaseEvent(smoothing_complete);
        clReleaseEvent(diff_x_complete);
        clReleaseEvent(diff_y_complete);
        clReleaseEvent(structure_complete);
        clReleaseEvent(response_complete);
        clReleaseEvent(suppression_complete);

        std::rethrow_exception(std::current_exception());
    }

private:
    std::vector<cl_device_id> device_ids_;
    std::vector<cl_platform_id> platform_ids_;
    cl_context context_;
    cl_program program_;
    FilterKernel gaussian_;

    cl_context CreateContext(cl_platform_id platform_id, const std::vector<cl_device_id>& device_ids) {
        const cl_context_properties context_properties [] =
        {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platform_id),
            0, 0
        };

        cl_int error = CL_SUCCESS;
        cl_context context = clCreateContext(context_properties, device_ids.size(), device_ids.data(), nullptr, nullptr, &error);
        CheckError(error, "Failed to create context");

        return context;
    }

    std::vector<cl_platform_id> GetPlatformIds() {
        cl_uint platform_id_count = 0;
        CheckError(
            clGetPlatformIDs (0, nullptr, &platform_id_count),
            "Failed to get platform ids length");

        if (platform_id_count == 0) throw std::runtime_error("No OpenCL platform found");

        std::vector<cl_platform_id> platform_ids(platform_id_count);
        CheckError(
            clGetPlatformIDs(platform_id_count, platform_ids.data(), nullptr),
            "Failed to get platform ids");

        return platform_ids;
    }

    std::vector<cl_device_id> GetDeviceIds(cl_platform_id platform_id) {
        cl_uint device_id_count = 0;
        CheckError(
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_id_count),
            "Failed to get device ids length");

        if (device_id_count == 0) throw std::runtime_error("No OpenCL devices found");

        std::vector<cl_device_id> device_ids(device_id_count);
        CheckError(
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_id_count, device_ids.data(), nullptr),
            "Failed to get device ids");

        return device_ids;
    }

    std::string GetPlatformName (cl_platform_id id)
    {
        size_t size = 0;
        CheckError(
            clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size),
            "Failed to get platform name length");

        std::string result;
        result.resize (size);
        CheckError(
            clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*> (result.data()), nullptr),
            "Failed to get platform name");

        return result;
    }

    std::string GetDeviceName(cl_device_id id)
    {
        size_t size = 0;
        clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

        std::string result;
        result.resize(size);
        CheckError(
            clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*> (result.data()), nullptr),
            "Failed to get device name");

        return result;
    }

    cl_program CreateProgram(const std::string& source_file, cl_context context)
    {
        std::ifstream in(source_file);
        std::stringstream source_stream;
        source_stream << in.rdbuf();
        auto source = source_stream.str();
        auto length = source.size();
        auto source_c_str = source.c_str();

        cl_int error = 0;
        cl_program program = clCreateProgramWithSource(context, 1, &source_c_str, &length, &error);
        CheckError(error, "Failed to create program");

        return program;
    }

    void BuildProgram(cl_program program, const std::vector<cl_device_id> device_ids) {
        std::stringstream options_stream;
        options_stream << " -D HALF_SMOOTHING=" << smoothing_size_ / 2; 
        options_stream << " -D HALF_STRUCTURE=" << structure_size_ / 2; 
        options_stream << " -D HALF_SUPPRESSION=" << suppression_size_ / 2; 
        options_stream << " -D HARRIS_K=" << k_;
        options_stream << " -D THRESHOLD_RATIO=" << threshold_ratio_;
        const auto options = options_stream.str();

        auto error = clBuildProgram (program, device_ids.size(), device_ids.data(), options.c_str(), nullptr, nullptr);

        if (error == CL_BUILD_PROGRAM_FAILURE) {
            // If program failed to build list the build log on stderror.
            std::cerr << "Program build failed. Build log:\n\n";

            for(auto device_id : device_ids) {
                std::cerr << GetDeviceName(device_id) << "\n\n";

                size_t log_size;
                CheckError(
                    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size),
                    "Failed to get program build log length");

                std::string log;
                log.resize(log_size);
                CheckError(
                    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, const_cast<char*>(log.data()), &log_size),
                    "Failed to get program build log");

                std::cerr << log << "\n";
            }

            throw std::runtime_error("Program build failed. See error log for details");
        }

        CheckError(error, "Failed to build program");
    }

    void CheckError(cl_int error, std::string message)
    {
        if (error != CL_SUCCESS) throw std::runtime_error(message + ": " + std::to_string(error));
    }

};
}

