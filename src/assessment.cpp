#include <opencv2/highgui/highgui_c.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

using namespace cv;
using namespace std;

float getHistogramBinValue(Mat hist, int binNum) {
    return hist.at<float>(binNum);
}

float getFrequencyOfBin(Mat channel) {
    float frequency = 0.0;
    int histSize = 256;
    for (int i = 1; i < histSize; i++) {
        float Hc = abs(getHistogramBinValue(channel, i));
        frequency += Hc;
    }
    return frequency;
}

float computeShannonEntropy(Mat r, Mat g, Mat b) {
    float entropy = 0.0;
    int histSize = 256;
    float frequency = getFrequencyOfBin(r);
    for (int i = 1; i < histSize; i++) {
        float Hc = abs(getHistogramBinValue(r, i));
        entropy += -(Hc / frequency) * log10((Hc / frequency));
    }
    frequency = getFrequencyOfBin(g);
    for (int i = 1; i < histSize; i++) {
        float Hc = abs(getHistogramBinValue(g, i));
        entropy += -(Hc / frequency) * log10((Hc / frequency));
    }
    frequency = getFrequencyOfBin(b);
    for (int i = 1; i < histSize; i++) {
        float Hc = abs(getHistogramBinValue(b, i));
        entropy += -(Hc / frequency) * log10((Hc / frequency));
    }
    entropy = entropy;
    cout << entropy << endl;
    return entropy;
}

// int main() {
//     std::vector<cv::String> fn;
//     std::string folder(
//         "D:"
//         "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
//         "on\\underwaterImages\\assessment\\entropy\\chosen\\*jpg");
//     glob(folder, fn, false);

//     vector<Mat> images;
//     size_t count = fn.size();  // number of jpg files in images folder
//     for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));
//     // cout << count << endl;

//     for (size_t i = 0; i < count; i++) {
//         // Mat rgb[3];
//         // split(images[i], rgb);
//         // computeShannonEntropy(rgb[0], rgb[1], rgb[2]);

//         if (images[i].channels() == 3)
//             cvtColor(images[i], images[i], CV_BGR2GRAY);
//         /// Establish the number of bins
//         int histSize = 256;
//         /// Set the ranges ( for B,G,R) )
//         float range[] = {0, 256};
//         const float* histRange = {range};
//         bool uniform = true;
//         bool accumulate = false;
//         Mat hist;
//         /// Compute the histograms:
//         calcHist(&images[i], 1, 0, Mat(), hist, 1, &histSize, &histRange,
//                  uniform, accumulate);
//         hist /= images[i].total();
//         hist += 1e-4;  // prevent 0

//         Mat logP;
//         cv::log(hist, logP);

//         float entropy = -1 * sum(hist.mul(logP)).val[0];

//         MyExcelFile << entropy << endl;
//     }
//     // end = clock();
//     // cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
//     MyExcelFile.close();

//     return 0;
// }

// canny
int main() {
    std::vector<cv::String> fn;
    std::string folder(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\entropy\\chosen\\*jpg");
    glob(folder, fn, false);

    vector<Mat> images;
    size_t count = fn.size();  // number of jpg files in images folder
    for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));
    // cout << count << endl;
    for (size_t i = 0; i < count; i++) {
        Mat imgGrayscale;
        Mat imgBlurred;
        Mat imgCanny;
        cv::cvtColor(images[i], imgGrayscale,
                     CV_BGR2GRAY);  // convert to grayscale

        cv::GaussianBlur(
            imgGrayscale,    // input image
            imgBlurred,      // output image
            cv::Size(5, 5),  // smoothing window width and height in pixels
            1.5);  // sigma value, determines how much the image will be blurred

        cv::Canny(imgBlurred,  // input image
                  imgCanny,    // output image
                  100,         // low threshold
                  200);        // high threshold

        stringstream ss1, ss2;

        string name =
            "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\entropy\\canny\\chosen_";
        string type = ".jpg";

        ss1 << name << i << type;

        string filename = ss1.str();
        ss1.str("");

        imwrite(filename, imgCanny);
        images[i] = imgCanny;
    }
    return 0;
}