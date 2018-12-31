#pragma once
// Harris corner detection algorithm implemented using OpenCL

#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef __APPLE__
    #include "OpenCL/cl.h"
#else
    #include "CL/cl.h"
#endif

#include "harris_base.h"

namespace harris {

class HarrisOpenCL : public HarrisBase {
public:

    HarrisOpenCL(int smoothing_size = 5, int structure_size = 5, float harris_k = 0.04, float threshold_ratio = 0.5, int suppression_size = 9) :
        HarrisBase(smoothing_size, structure_size, harris_k, threshold_ratio, suppression_size) {

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

        cl_program program = CreateProgram("harris.cl", context_);
        std::cout << "Program created" << std::endl;

        BuildProgram(program, device_ids_);
        std::cout << "Program built" << std::endl;
    }

    // Rule of five: Neither movable nor copyable
    HarrisOpenCL(const HarrisOpenCL&) = delete;
    HarrisOpenCL(HarrisOpenCL&&) = delete;
    HarrisOpenCL& operator=(const HarrisOpenCL&) = delete;
    HarrisOpenCL& operator=(HarrisOpenCL&&) = delete;
    ~HarrisOpenCL() override = default;

    // Runs the OpenCL Harris corner detector
    Image<float> FindCorners(const Image<float>& image) override {
        throw std::runtime_error("OpenCL is not implemented yet");
    }
private:
    std::vector<cl_device_id> device_ids_;
    std::vector<cl_platform_id> platform_ids_;
    cl_context context_;

    cl_context CreateContext(cl_platform_id platform_id, const std::vector<cl_device_id>& device_ids) {
        const cl_context_properties context_properties [] =
        {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platform_id),
            0, 0
        };

        cl_int error = CL_SUCCESS;
        cl_context context = clCreateContext(context_properties, device_ids.size(), device_ids.data(), nullptr, nullptr, &error);
        CheckError(error);

        return context;
    }

    std::vector<cl_platform_id> GetPlatformIds() {
        cl_uint platform_id_count = 0;
        clGetPlatformIDs (0, nullptr, &platform_id_count);

        if (platform_id_count == 0) throw std::runtime_error("No OpenCL platform found");

        std::vector<cl_platform_id> platform_ids(platform_id_count);
        clGetPlatformIDs(platform_id_count, platform_ids.data(), nullptr);

        return platform_ids;
    }

    std::vector<cl_device_id> GetDeviceIds(cl_platform_id platform_id) {
        cl_uint device_id_count = 0;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_id_count);

        if (device_id_count == 0) throw std::runtime_error("No OpenCL devices found");

        std::vector<cl_device_id> device_ids(device_id_count);
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_id_count, device_ids.data(), nullptr);

        return device_ids;
    }

    std::string GetPlatformName (cl_platform_id id)
    {
        size_t size = 0;
        clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

        std::string result;
        result.resize (size);
        clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*> (result.data()), nullptr);

        return result;
    }

    std::string GetDeviceName (cl_device_id id)
    {
        size_t size = 0;
        clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

        std::string result;
        result.resize(size);
        clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*> (result.data()), nullptr);

        return result;
    }

    cl_program CreateProgram (const std::string& source_file, cl_context context)
    {
        std::ifstream in(source_file);
        std::string source(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
        auto length = source.size();
        auto source_c_str = source.c_str();

        cl_int error = 0;
        cl_program program = clCreateProgramWithSource(context, 1, &source_c_str, &length, &error);
        CheckError (error);

        return program;
    }

    void BuildProgram(cl_program program, const std::vector<cl_device_id> device_ids) {
        std::stringstream options_stream;
        options_stream << " -D HALF_SMOOTHING=" << smoothing_size_ / 2; 
        options_stream << " -D HALF_STRUCTURE=" << structure_size_ / 2; 
        options_stream << " -D HALF_SUPPRESSION=" << suppression_size_ / 2; 
        options_stream << " -D HARRIS_K=" << k_ / 2;
        const auto options = options_stream.str();

        auto error = clBuildProgram (program, device_ids.size(), device_ids.data(), options.c_str(), nullptr, nullptr);

        if (error == CL_BUILD_PROGRAM_FAILURE) {
            // If program failed to build list the build log in the error.
            std::cerr << "Program build failed. Build log:\n";

            for(auto device_id : device_ids) {
                std::cerr << GetDeviceName(device_id) << "\n";

                size_t log_size;
                CheckError(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));

                std::string log;
                log.resize(log_size);
                CheckError(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, const_cast<char*>(log.data()), &log_size));

                std::cerr << log << "\n";
            }

            throw std::runtime_error("Program build failed. See error log for details");
        }

        CheckError(error);
    }

    void CheckError(cl_int error)
    {
        if (error != CL_SUCCESS) throw std::runtime_error("OpenCL call failed with error " + std::to_string(error));
    }

};
}

