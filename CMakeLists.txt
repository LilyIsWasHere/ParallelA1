cmake_minimum_required(VERSION 3.28)
project(a1)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)

add_executable(a1 main.cpp)

target_link_libraries(a1 PUBLIC MPI::MPI_CXX)