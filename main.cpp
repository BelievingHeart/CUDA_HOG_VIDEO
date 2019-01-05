#include <fmt/printf.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace fmt;
int main(const int argc, const char *argv[]) {
    const CommandLineParser parser(argc, argv,
                                   "{help ? h ||}"
                                   "{@video | /home/afterburner/Downloads/street_view.mp4 | input image}"
                                   "{svm_weights| ../SVM_HOG.xml | image scale factor}");
    if (parser.has("help")) {
        parser.printMessage();
        return 2;
    }
    const auto video_path = parser.get<String>("@video");
    const auto svm_weights_path = parser.get<String>("svm_weights");

    // weights
    // NOTE: do not create and load. That will be a great trouble
    const Ptr<ml::SVM> svm_ptr = ml::SVM::load(svm_weights_path);
    // Get the compressed support vectors.
    // The classifier used to detected pedestrian is a binary classifier
    // Thus there is only 1 compressed support vector
    // To consult the original support vectors for each consider using getUncompressedSupportVectors()
    const Mat support_vectors = svm_ptr->getSupportVectors(); // 1 x descriptor_dims
    print("Number of compressed support vectors: {}\n", support_vectors.rows);

    // Get alphas and b
    std::vector<float> alphas, _; // The size of alphas equals the number of compressed support vectors. Because OpenCV throws away all the zeros and only left with 1s.
    const double bias = svm_ptr->getDecisionFunction(0, alphas, _); // There is only 1 boundary, the 0th boundary, in binary classification
    // Multiply each support vectors with corresponding alpha
    Mat alphas_mat(alphas);
    // Get the weighted compressed support vectors
    Mat w = alphas_mat * support_vectors; // 1 x NUM_SV  *  NUM_SV x descriptor_dims  ==> 1 x descriptor_dims
    w = -1 * w; // NOTE: don't forget -1

    // We now have all the parameters required to describe the only one decision boundary. Namely, the weighted compressed support vector and the bias
    std::vector<float> boundary_params(w.begin<float>(), w.end<float>());
    boundary_params.push_back(static_cast<float>(bias));

    // HOG
    const auto hog_ptr = cuda::HOG::create();
    hog_ptr->setSVMDetector(boundary_params);

    // CAP
    VideoCapture cap(video_path);
    cap.set(CAP_PROP_POS_FRAMES, 200);

    // Helper function
    std::vector<Rect2i> boxes;
    const auto draw_boxes = [](const auto &boxes, auto &canvas) {
        for (const auto b : boxes) {
            rectangle(canvas, b, {0, 255, 0});
        }
    };

    // Detect loops
    int key = 0;
    Mat frame, resized_downloaded;
    cuda::GpuMat gpu_frame, gpu_frame_resized, gpu_frame_grayscale;
    try {
        while (key != 27 && cap.read(frame)) {
            const auto start_count = getTickCount();
            gpu_frame.upload(frame);
            cuda::resize(gpu_frame, gpu_frame_resized, {480, 320});
            cuda::cvtColor(gpu_frame_resized, gpu_frame_grayscale, COLOR_BGR2GRAY);
            hog_ptr->detectMultiScale(gpu_frame_grayscale, boxes, nullptr);
            gpu_frame_resized.download(resized_downloaded);
            draw_boxes(boxes, resized_downloaded);
            const auto time_elapsed = 1.f / ((getTickCount() - start_count) / getTickFrequency());
            putText(resized_downloaded, String("FPS: ") + std::to_string(time_elapsed), {20, 20}, FONT_HERSHEY_SIMPLEX, 0.5, {255, 0, 0});
            imshow("Frames", resized_downloaded);
            key = waitKey(1);
        }
    }

    catch (const Exception &e) {
        return std::cout << "error: " << e.what() << '\n', 1;
    } catch (const std::exception &e) {
        return std::cout << "error: " << e.what() << '\n', 1;
    } catch (...) {
        return std::cout << "unknown exception" << '\n', 1;
    }
}
