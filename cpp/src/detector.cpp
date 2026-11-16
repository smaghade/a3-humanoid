#include "a3/detector.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <opencv2/imgproc.hpp>

namespace a3 {
const std::vector<std::string> COLOUR_NAMES = {"red", "blue", "yellow", "green"};
// HSV ranges (OpenCV hue 0-180). Saturation >= 170 = strong colour: the blue-grey floor (~128)
// and the brown table (~153) are never mistaken for a bottle; the bottles are about 226.
static const std::map<std::string, std::vector<std::pair<cv::Scalar, cv::Scalar>>> COLOURS = {
    {"red", {{{0, 170, 70}, {10, 255, 255}}, {{170, 170, 70}, {180, 255, 255}}}},
    {"blue", {{{100, 170, 70}, {130, 255, 255}}}},
    {"yellow", {{{20, 170, 70}, {35, 255, 255}}}},
    {"green", {{{40, 170, 70}, {80, 255, 255}}}},
};
static const std::map<std::string, cv::Scalar> BOX_COLOURS = {
    {"red", {0, 0, 255}}, {"blue", {255, 80, 0}}, {"yellow", {0, 220, 255}}, {"green", {0, 200, 0}}};
static const char* COCO[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

cv::Mat colour_mask(const cv::Mat& bgr, const std::string& colour) {
  cv::Mat hsv, mask = cv::Mat::zeros(bgr.size(), CV_8U), part;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  for (const auto& r : COLOURS.at(colour)) {
    cv::inRange(hsv, r.first, r.second, part);
    mask |= part;
  }
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
  return mask;
}

std::vector<cv::Rect> find_colour_blobs(const cv::Mat& bgr, const std::string& colour, double min_area) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(colour_mask(bgr, colour), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Rect> boxes;
  for (const auto& c : contours)
    if (cv::contourArea(c) >= min_area) boxes.push_back(cv::boundingRect(c));
  std::stable_sort(boxes.begin(), boxes.end(), [](const cv::Rect& a, const cv::Rect& b) { return a.area() > b.area(); });
  return boxes;
}

double colour_fraction(const cv::Mat& bgr, const cv::Rect& box, const std::string& colour) {
  cv::Rect r = box & cv::Rect(0, 0, bgr.cols, bgr.rows);
  if (r.area() <= 0) return 0.0;
  return cv::mean(colour_mask(bgr(r), colour))[0] / 255.0;
}

static double overlap(const cv::Rect& a, const cv::Rect& b) {
  double inter = (a & b).area();
  double area = std::min(a.area(), b.area());
  return area ? inter / area : 0.0;
}

ObjectDetector::ObjectDetector(const std::string& model_path, double conf, bool use_yolo) : conf_(conf) {
  if (!use_yolo) return;
  std::string path = model_path;
  if (!std::ifstream(path).good() && std::ifstream("yolo11n.onnx").good()) path = "yolo11n.onnx";
  if (!std::ifstream(path).good()) {
    std::printf("[detector] YOLO not available (%s not found); using colour detection only\n", model_path.c_str());
    return;
  }
  try {
    net_ = cv::dnn::readNetFromONNX(path);
    yolo_ok_ = true;
    std::printf("[detector] YOLO loaded from %s (OpenCV DNN, CPU)\n", path.c_str());
  } catch (const cv::Exception& e) {
    std::printf("[detector] YOLO not available (%s); using colour detection only\n", e.what());
  }
}

std::vector<Detection> ObjectDetector::yolo(const cv::Mat& bgr) {
  std::vector<Detection> found;
  if (!yolo_ok_) return found;
  // letterbox to 640x640 (keep shape, grey padding), like Ultralytics
  const int S = 640;
  double r = std::min(S / (double)bgr.cols, S / (double)bgr.rows);
  int nw = (int)std::round(bgr.cols * r), nh = (int)std::round(bgr.rows * r);
  int px = (S - nw) / 2, py = (S - nh) / 2;
  cv::Mat resized, canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr, resized, cv::Size(nw, nh));
  resized.copyTo(canvas(cv::Rect(px, py, nw, nh)));
  cv::Mat out;
  try {
    net_.setInput(cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(S, S), cv::Scalar(), true, false));
    out = net_.forward();                            // [1, 84, 8400]: cx, cy, w, h, 80 class scores
  } catch (const cv::Exception& e) {
    std::printf("[detector] YOLO could not run (%s); using colour detection only\n", e.what());
    yolo_ok_ = false;
    return found;
  }
  if (out.dims != 3 || out.size[1] < 5) {
    std::printf("[detector] unexpected YOLO output shape; using colour detection only\n");
    yolo_ok_ = false;
    return found;
  }
  int rows = out.size[1], cols = out.size[2];
  cv::Mat m(rows, cols, CV_32F, out.ptr<float>());
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> classes;
  for (int i = 0; i < cols; i++) {
    int best = -1;
    float bs = 0;
    for (int c = 4; c < rows; c++)
      if (m.at<float>(c, i) > bs) { bs = m.at<float>(c, i); best = c - 4; }
    if (bs < conf_) continue;
    float cx = m.at<float>(0, i), cy = m.at<float>(1, i), w = m.at<float>(2, i), h = m.at<float>(3, i);
    int x1 = (int)((cx - w / 2 - px) / r), y1 = (int)((cy - h / 2 - py) / r);
    int x2 = (int)((cx + w / 2 - px) / r), y2 = (int)((cy + h / 2 - py) / r);
    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
    scores.push_back(bs);
    classes.push_back(best);
  }
  std::vector<int> keep;                             // non-maximum suppression per class (like Ultralytics)
  for (int cls = 0; cls < 80; cls++) {
    std::vector<int> idx;
    std::vector<cv::Rect> cb;
    std::vector<float> cs;
    for (size_t i = 0; i < boxes.size(); i++)
      if (classes[i] == cls) { idx.push_back((int)i); cb.push_back(boxes[i]); cs.push_back(scores[i]); }
    if (idx.empty()) continue;
    std::vector<int> k;
    cv::dnn::NMSBoxes(cb, cs, (float)conf_, 0.7f, k);
    for (int i : k) keep.push_back(idx[i]);
  }
  std::stable_sort(keep.begin(), keep.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  for (int k : keep) {
    Detection d;
    d.label = COCO[classes[k]];
    d.conf = scores[k];
    d.box = boxes[k];
    d.source = "yolo";
    found.push_back(d);
  }
  return found;
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat& bgr) {
  std::vector<Detection> dets = yolo(bgr);
  for (auto& d : dets)                              // give each YOLO object a colour
    for (const auto& c : COLOUR_NAMES)
      if (colour_fraction(bgr, d.box, c) > 0.25) { d.colour = c; break; }
  for (const char* colour : {"red", "blue", "yellow"})   // tall coloured blobs YOLO missed
    for (const auto& box : find_colour_blobs(bgr, colour)) {
      bool covered = false;
      for (const auto& d : dets)
        if (overlap(box, d.box) > 0.3) { covered = true; break; }
      if (covered) continue;
      Detection d;
      d.label = box.height > 1.4 * box.width ? "bottle" : "object";
      d.colour = colour;
      d.conf = 1.0;
      d.box = box;
      d.source = "colour";
      dets.push_back(d);
    }
  for (auto& d : dets) {
    d.centre = cv::Point((d.box.x + d.x2()) / 2, (d.box.y + d.y2()) / 2);
    d.name = d.colour.empty() ? d.label : d.colour + " " + d.label;
  }
  return dets;
}

std::vector<const Detection*> ObjectDetector::find_all(const std::vector<Detection>& dets, const std::string& name) {
  std::vector<const Detection*> m;
  for (const auto& d : dets)
    if (d.name == name) m.push_back(&d);
  std::stable_sort(m.begin(), m.end(), [](const Detection* a, const Detection* b) { return a->box.area() > b->box.area(); });
  return m;
}
const Detection* ObjectDetector::find(const std::vector<Detection>& dets, const std::string& name) {
  auto m = find_all(dets, name);
  return m.empty() ? nullptr : m[0];
}

cv::Mat& draw(cv::Mat& bgr, const std::vector<Detection>& dets, const std::string& target) {
  for (const auto& d : dets) {
    auto it = BOX_COLOURS.find(d.colour);
    cv::Scalar colour = it != BOX_COLOURS.end() ? it->second : cv::Scalar(0, 255, 255);
    int thick = (!target.empty() && d.name == target) ? 3 : 1;
    cv::rectangle(bgr, d.box, colour, thick);
    char text[160];
    std::snprintf(text, sizeof(text), "%s %.2f [%s]", d.name.c_str(), d.conf, d.source.c_str());
    cv::Point org(d.box.x, std::max(15, d.box.y - 6));
    cv::putText(bgr, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(bgr, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv::LINE_AA);
    cv::circle(bgr, d.centre, 3, colour, -1);
  }
  return bgr;
}
}  // namespace a3
