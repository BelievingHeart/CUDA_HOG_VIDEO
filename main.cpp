#include <fmt/printf.h>
#include <opencv2/opencv.hpp>

namespace db {
void imshow(const cv::String &winName_, const cv::Mat &img_) {
#ifdef DEBUG
  if (img_.depth() == CV_8U) {
    cv::imshow(winName_, img_);
  } else {
    double min_ = 0, max_ = 0;
    cv::Point2i min_pt, max_pt;
    cv::Mat temp;
    cv::minMaxLoc(img_, &min_, &max_, &min_pt, &max_pt, cv::noArray());
    cv::normalize(img_, temp, 0, 255, cv::NORM_MINMAX, CV_8U, {});

    cv::imshow(winName_, temp);
    if (min_ < 0 || max_ > 1) {
      fmt::print("[DEBUG] {} is not of any showing formats, a makeshift image "
                 "is create.\nOriginal states:\n",
                 winName_);
      fmt::print("minVal: {} at point ({},{})\n", min_, min_pt.x, min_pt.y);
      fmt::print("maxVal: {} at point ({},{})\n\n", max_, max_pt.x, max_pt.y);
    }
  }
#endif // DEBUG
}
}; // namespace db

using namespace cv;
int main(const int argc, const char *argv[]) {
  const CommandLineParser parser(
      argc, argv,
      "{help ? h ||}"
      "{@video | /home/afterburner/Downloads/street_view.mp4 | input image}"
      "{svm_weights| ../SVM_HOG.xml | image scale factor}"
      );
  if (parser.has("help")) {
    parser.printMessage();
    return 2;
  }
  const auto video_path = parser.get<std::string>("@video");
  const auto svm_weights_path = parser.get<std::string>("svm_weights");
}
