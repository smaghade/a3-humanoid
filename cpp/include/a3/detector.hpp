// Day 10: find objects in the robot's camera picture.
//   1. YOLO11n (pretrained, no training), run with OpenCV DNN from models/yolo11n.onnx
//      (export once:  yolo export model=yolo11n.pt format=onnx opset=12  -> models/yolo11n.onnx)
//   2. Colour check (HSV): names WHICH bottle is red/blue/yellow, and finds tall coloured
//      shapes YOLO misses (the simulated bottles are plain cylinders).
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace a3 {
struct Detection {
  std::string label, colour, source, name;          // name e.g. "red bottle"; colour "" if none
  double conf = 0;
  cv::Rect box;                                     // x, y, width, height (pixels)
  cv::Point centre;
  int x2() const { return box.x + box.width; }
  int y2() const { return box.y + box.height; }
};

cv::Mat colour_mask(const cv::Mat& bgr, const std::string& colour);
std::vector<cv::Rect> find_colour_blobs(const cv::Mat& bgr, const std::string& colour, double min_area = 80);
double colour_fraction(const cv::Mat& bgr, const cv::Rect& box, const std::string& colour);
extern const std::vector<std::string> COLOUR_NAMES;   // red, blue, yellow, green

class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& model_path = "models/yolo11n.onnx", double conf = 0.35,
                          bool use_yolo = true);
  std::vector<Detection> detect(const cv::Mat& bgr);
  static std::vector<const Detection*> find_all(const std::vector<Detection>& dets, const std::string& name);
  static const Detection* find(const std::vector<Detection>& dets, const std::string& name);
  bool has_yolo() const { return yolo_ok_; }

 private:
  std::vector<Detection> yolo(const cv::Mat& bgr);
  double conf_;
  bool yolo_ok_ = false;
  cv::dnn::Net net_;
};

cv::Mat& draw(cv::Mat& bgr, const std::vector<Detection>& dets, const std::string& target = "");
}  // namespace a3
