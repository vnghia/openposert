#include <cstddef>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "cxxopts.hpp"
#include "minrt/utils.hpp"
#include "opencv2/opencv.hpp"
#include "openposert/openposert.hpp"
#include "openposert/utilities/pose_model.hpp"
#include "spdlog/spdlog.h"

using namespace openposert;
using std::size_t;

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

  auto input_width = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  auto input_height = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  auto input_channels = static_cast<size_t>(cap.get(cv::CAP_PROP_CHANNEL));
  input_channels = (input_channels > 0) ? input_channels : 3;

  auto openposert = OpenPoseRT(
      args["engine"].as<fs::path>(), input_width, input_height, input_channels,
      PoseModel::BODY_25B, false, 0, 25, 5, 0, 0, 0, 8, 0);

  cv::Mat frame(input_height, input_width, CV_8UC3,
                openposert.get_input_data());

  const auto pose_key_points = openposert.get_pose_keypoints();

  unsigned long frame_count = 0;
  while (cap.read(frame)) {
    if (++frame_count % 2 == 0) continue;

    openposert.forward();

    for (size_t i = 0; i < openposert.get_pose_keypoints_size() / 3; i++) {
      cv::circle(frame,
                 cv::Point(pose_key_points[i * 3] * input_width /
                               openposert.net_output_width,
                           pose_key_points[i * 3 + 1] * input_width /
                               openposert.net_output_width),
                 5, cv::Scalar(0, 0, 255), -1);
    }
    cv::imshow("OpenPoseRT", frame);
    if (cv::waitKey(1) >= 0) break;
  }

  return 0;
}
