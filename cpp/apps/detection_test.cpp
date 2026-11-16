// Day 10 measurement: run the detector on saved camera pictures and count what it finds.
// C++ port of tools/detection_test.py.  Usage (from the project folder):
//   ./cpp/build/detection_test                       (pictures in data/snapshots/)
//   ./cpp/build/detection_test --target "blue bottle"
// Writes pictures with boxes to data/detections/.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <utility>
#include <vector>
#include "a3/detector.hpp"

using namespace a3;
namespace fs = std::filesystem;
using Counter = std::vector<std::pair<std::string, int>>;   // keeps first-seen order, like Python's Counter
static void bump(Counter& c, const std::string& k) {
  for (auto& kv : c)
    if (kv.first == k) { kv.second++; return; }
  c.push_back({k, 1});
}
static std::string as_dict(const Counter& c) {
  std::string s = "{";
  for (size_t i = 0; i < c.size(); i++) s += (i ? ", '" : "'") + c[i].first + "': " + std::to_string(c[i].second);
  return s + "}";
}

int main(int argc, char** argv) {
  std::string folder = "data/snapshots", target = "red bottle";
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--images" && i + 1 < argc) folder = argv[++i];     // a folder of .png pictures
    else if (s == "--target" && i + 1 < argc) target = argv[++i];
    else { std::fprintf(stderr, "unrecognized argument: %s\n", s.c_str()); return 1; }
  }
  std::vector<std::string> files;
  if (fs::is_directory(folder))
    for (const auto& e : fs::directory_iterator(folder))
      if (e.path().extension() == ".png") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::fprintf(stderr, "No pictures found in %s/*.png. Save some with C in the EYES window first.\n", folder.c_str());
    return 1;
  }
  ObjectDetector detector;
  fs::create_directories("data/detections");
  Counter counts, by_source;
  int target_hits = 0;
  std::printf("%-32s found\n", "picture");
  for (const auto& path : files) {
    cv::Mat img = cv::imread(path);
    auto found = detector.detect(img);
    std::vector<std::string> names, unique;
    for (const auto& d : found) {
      names.push_back(d.name);
      if (std::find(unique.begin(), unique.end(), d.name) == unique.end()) unique.push_back(d.name);
      bump(by_source, d.source);
    }
    for (const auto& n : unique) bump(counts, n);
    if (ObjectDetector::find(found, target)) target_hits++;
    cv::imwrite("data/detections/" + fs::path(path).filename().string(), draw(img, found, target));
    std::string list;
    for (size_t i = 0; i < names.size(); i++) list += (i ? ", " : "") + names[i];
    std::printf("%-32s %s\n", fs::path(path).filename().string().c_str(), list.empty() ? "-" : list.c_str());
  }
  int n = (int)files.size();
  std::printf("\n'%s' found in %d of %d pictures (%.0f%%)\n", target.c_str(), target_hits, n, 100.0 * target_hits / n);
  std::printf("Seen in how many pictures: %s\n", as_dict(counts).c_str());
  std::printf("Found by: %s (yolo = AI model, colour = colour check)\n", as_dict(by_source).c_str());
  std::printf("Pictures with boxes saved in data/detections/\n");
}
