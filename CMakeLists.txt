# Cmake 3.11 for proper OpenCL and Googletest support
cmake_minimum_required(VERSION 3.11)

# Project level properties
project(harris)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# Find Dependencies
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Main Target
set(MAIN_SOURCE main.cc)
add_executable(${PROJECT_NAME} ${MAIN_SOURCE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Testing
enable_testing()
include(Googletest)
add_executable(unitTests unit_tests.cpp "extern/googletest/googletest/src/gtest-all.cc" "extern/googletest/googletest/src/gtest_main.cc")
target_include_directories(unitTests PRIVATE "extern/googletest/googletest/include")
target_include_directories(unitTests PRIVATE "extern/googletest/googletest/src")
target_include_directories(unitTests PRIVATE "extern/googletest/googletest")
target_link_libraries(unitTests PRIVATE ${OpenCV_LIBS})
target_link_libraries(unitTests PRIVATE ${OpenCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(unitTests PRIVATE OpenMP::OpenMP_CXX)
endif()

gtest_add_tests(TARGET unitTests)
