#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>

bool getAllFrames(std::string filename, std::vector<cv::Mat>& frames) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::VideoCapture cap(filename);
    auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Init capture Elapsed time: " << elapsed.count() << " ms" << std::endl;
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return false;
    }

    start = std::chrono::high_resolution_clock::now();
    int index = 0;
    while (true) {
        cv::Mat frame, frame8;
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::cvtColor(frame, frame8, cv::COLOR_BGR2GRAY);
        frames.push_back(frame8.clone());
        index++;
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Read frame Average Elapsed time: " << elapsed.count() / float(index) << " ms" << std::endl;
    std::cout << "Read frame All Elapsed time: " << elapsed.count() << " ms" << std::endl;

    return true;
}

bool getAllFramesNew(std::string filename, int vector_start_index, std::vector<cv::Mat>& frames) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return false;
    }
    int index = vector_start_index;
    while (true) {
        cv::Mat frame, frame8;
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::cvtColor(frame, frame8, cv::COLOR_BGR2GRAY);
        // frames.push_back(frame8.clone());
        frames[index] = frame8.clone();
        index++;
    }

    return true;
}