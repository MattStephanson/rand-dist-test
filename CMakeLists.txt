cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project ("rand-dist-test")

# Add source to this project's executable.
add_executable (rand-dist-test "rand-dist-test.cpp")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rand-dist-test PUBLIC OpenMP::OpenMP_CXX)
endif()
