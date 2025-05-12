#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <format>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"




int main() {
    // Open the default camera (usually the first camera device)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Webcam Feed", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // Capture a new frame
        auto t_cap_0 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto t_cap_1 = std::chrono::high_resolution_clock::now();

        // Display the frame in the window
        if (frame.empty()) {
            std::cerr << "Warning: Captured empty frame." << std::endl;
            break;
        }

        // Display the frame in the window
        auto t_disp_0 = std::chrono::high_resolution_clock::now();
        cv::imshow("Webcam Feed", frame);
        auto t_disp_1 = std::chrono::high_resolution_clock::now();

        // Wait for 25 ms and check if the 'Esc' key is pressed
        if (cv::waitKey(25) == 27) {
            std::cout << "'Esc' key pressed, exiting."<< std::endl;
            break;
        }

        std::chrono::duration<double, std::milli> duration_cap = t_cap_1 - t_cap_0;
        std::chrono::duration<double, std::milli> duration_disp = t_disp_1 - t_disp_0;
        std::cout << std::format("capture: {}, display: {}\n", duration_cap.count(), duration_disp.count());
    }

    // Release the camera and destroy the window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
