cmake_minimum_required(VERSION 3.16)

project(GestureAmp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")

include_directories(include third-party/tensorflow)

find_package(OpenCV REQUIRED)

add_executable(camera_feed ${SOURCES})

target_link_libraries(camera_feed
    ${OpenCV_LIBS}
    tensorflowlite
)

link_directories(third-party/tensorflow)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wno-deprecated-enum-enum-conversion)
endif()

