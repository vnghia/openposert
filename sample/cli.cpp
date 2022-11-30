#include <iostream>
#include <opencv2/imgproc.hpp>

#include "cxxopts.hpp"
#include "opencv2/opencv.hpp"
#include "openposert/openposert.hpp"
#include "spdlog/spdlog.h"

using namespace openposert;

int main(int argc, char* argv[]) {
  cxxopts::Options options("openposert", "Openpose TensorRT Engine");
  auto options_adder = options.add_options();
  options_adder("help", "Print help");
  options_adder("e,engine", "Engine path", cxxopts::value<fs::path>());
  options_adder("i,input", "Video input path", cxxopts::value<fs::path>());
  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  cv::VideoCapture cap(args["input"].as<fs::path>());

  auto input_width =
      static_cast<std::size_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  auto input_height =
      static_cast<std::size_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  auto input_channels = static_cast<std::size_t>(cap.get(cv::CAP_PROP_CHANNEL));
  input_channels = (input_channels > 0) ? input_channels : 3;

  auto openposert = OpenPoseRT(args["engine"].as<fs::path>(), input_width,
                               input_height, input_channels);

  cv::Mat frame(input_height, input_width, CV_8UC3,
                openposert.get_input_data());

  while (cap.read(frame)) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    openposert.forward();
    cv::imshow("OpenPoseRT", frame);
    if (cv::waitKey(5) >= 0) break;
    spdlog::info("result ndim: {}", openposert.get_pose_keypoints().get_size());
  }

  return 0;
}
