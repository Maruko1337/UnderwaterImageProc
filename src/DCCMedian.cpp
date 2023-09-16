//#include <opencv2/highgui/highgui_c.h>
#include <time.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;
using namespace std;

// median filtered dark channel
Mat getMedianDarkChannel(Mat src, int patch) {
    Mat rgbmin = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat MDCP;
    Vec3b intensity;

    for (int m = 0; m < src.rows; m++) {
        for (int n = 0; n < src.cols; n++) {
            intensity = src.at<Vec3b>(m, n);
            rgbmin.at<uchar>(m, n) =
                min(min(intensity.val[0], intensity.val[1]), intensity.val[2]);
        }
    }
    medianBlur(rgbmin, MDCP, patch);
    return MDCP;
}

// estimate airlight by the brightest pixel in dark channel (proposed by He et
// al.)
int estimateA(Mat DC) {
    double minDC, maxDC;
    minMaxLoc(DC, &minDC, &maxDC);
    // cout << "estimated airlight is:" << maxDC << endl;
    return maxDC;
}

// estimate transmission map
Mat estimateTransmission(Mat DCP, int ac) {
    double w = 0.75;
    Mat transmission = Mat::zeros(DCP.rows, DCP.cols, CV_8UC1);
    Scalar intensity;

    for (int m = 0; m < DCP.rows; m++) {
        for (int n = 0; n < DCP.cols; n++) {
            intensity = DCP.at<uchar>(m, n);
            transmission.at<uchar>(m, n) =
                (1 - w * intensity.val[0] / ac) * 255;
        }
    }
    return transmission;
}

// dehazing foggy image
Mat getDehazed(Mat source, Mat t, int al) {
    double tmin = 0.1;
    double tmax;

    Scalar inttran;
    Vec3b intsrc;
    Mat dehazed = Mat::zeros(source.rows, source.cols, CV_8UC3);

    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            inttran = t.at<uchar>(i, j);
            intsrc = source.at<Vec3b>(i, j);
            tmax =
                (inttran.val[0] / 255) < tmin ? tmin : (inttran.val[0] / 255);
            for (int k = 0; k < 3; k++) {
                dehazed.at<Vec3b>(i, j)[k] =
                    abs((intsrc.val[k] - al) / tmax + al) > 255
                        ? 255
                        : abs((intsrc.val[k] - al) / tmax + al);
            }
        }
    }
    return dehazed;
}

int main(int argc, char** argv) {
    // clock_t start, end;
    // start = clock();

    std::vector<cv::String> fn;
    std::string folder(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\entropy\\raw-890\\*png");
    glob(folder, fn, false);
    vector<Mat> images;
    size_t count = fn.size();  // number of jpg files in images folder
    for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));

    cout << count << endl;
    for (size_t i = 0; i < count; i++) {
        Mat fog = images[i];
        Mat darkChannel;
        Mat T;
        Mat fogfree;
        Mat beforeafter = Mat::zeros(fog.rows, 2 * fog.cols, CV_8UC3);
        Rect roil(0, 0, fog.cols, fog.rows);
        Rect roir(fog.cols, 0, fog.cols, fog.rows);
        int Airlight;
        // namedWindow("before and after");

        int patch = 5;
        darkChannel = getMedianDarkChannel(fog, patch);
        Airlight = estimateA(darkChannel);
        T = estimateTransmission(darkChannel, Airlight);
        fogfree = getDehazed(fog, T, Airlight);

        fog.copyTo(beforeafter(roil));
        fogfree.copyTo(beforeafter(roir));

        stringstream ss1, ss2;

        string name =
            "D:"
            "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplemen"
            "tati"
            "on\\underwaterImages\\assessment\\entropy\\dehazed_MDCP\\MDCP_";
        //         "D:"
        // "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        // "on\\underwaterImages\\assessment\\dehazed\\compared";
        string type = ".jpg";

        ss1 << name << i << type;
        // ss2 << name2 << i << type;

        string filename = ss1.str();
        ss1.str("");

        // string resultName = ss2.str();
        // ss2.str("");

        imwrite(filename, fogfree);
        // imwrite(resultName, beforeafter);
        // imshow("before and after", beforeafter);
        waitKey();
    }
    // end = clock();
    // cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
    return 0;
}