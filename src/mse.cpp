#include <opencv2/highgui/highgui_c.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

using namespace cv;
using namespace std;

double getMSE(Mat& I1, Mat& I2) {
    Mat s1;
    // save the I1 and I2 type before converting to float
    int im1type = I1.type();
    int im2type = I2.type();
    // convert to float to avoid producing zero for negative numbers
    I1.convertTo(I1, CV_32F);
    I2.convertTo(I2, CV_32F);
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);  // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2];  // sum channels

    if (sse <= 1e-10)  // for small values return zero
        return 0;
    else {
        double mse = sse / (double)(I1.channels() * I1.total());
        return mse;
        // Instead of returning MSE, the tutorial code returned PSNR (below).
        double psnr = 10.0 * log10((255 * 255) / mse);
        // return psnr;
    }
    // return I1 and I2 to their initial types
    I1.convertTo(I1, im1type);
    I2.convertTo(I2, im2type);
}

void printDiffRGB(Mat& I1, Mat& I2) {
    Mat s1;
    // save the I1 and I2 type before converting to float
    int im1type = I1.type();
    int im2type = I2.type();
    // convert to float to avoid producing zero for negative numbers
    I1.convertTo(I1, CV_32F);
    I2.convertTo(I2, CV_32F);
    int previous = 0;
    int output;
    int columns = 0;
    for (int i = 0; i < I1.cols; ++i) {
        for (int j = 0; j < I1.rows; ++j) {
            output =
                sqrt((I2.at<Vec3b>(i, j)[0] - I1.at<Vec3b>(i, j)[0]) ^
                     2 + (I2.at<Vec3b>(i, j)[1] - I1.at<Vec3b>(i, j)[1]) ^
                     2 + (I2.at<Vec3b>(i, j)[2] - I1.at<Vec3b>(i, j)[2]) ^ 2);
            if (previous == output || output == 0) {
                previous = 0;
                continue;
            } else {
                previous = output;
                cout << output << "   ";
                if (columns == 5) {
                    cout << endl;
                    columns = 0;
                } else {
                    columns++;
                }
            }
        }
    }
}

void diffImg(Mat& I1, Mat& I2) {
    Mat s1;
    // save the I1 and I2 type before converting to float
    int im1type = I1.type();
    int im2type = I2.type();
    // convert to float to avoid producing zero for negative numbers
    I1.convertTo(I1, CV_32F);
    I2.convertTo(I2, CV_32F);
    absdiff(I1, I2, s1);  // |I1 - I2|
    imwrite(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\colorCompare\\Pur_diff.jpg",
        s1);
}

int main() {
    Mat Original = imread(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\colorCompare\\OriginalGrid.png");
    Mat hazed = imread(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\colorCompare\\Hazed_1.png");
    Mat purposed = imread(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\colorCompare\\Proposed1.png");
    // diffImg(Original, purposed);
    cout << "mse of purposed: " << getMSE(Original, purposed) << endl;
    cout << "mse of hazed: " << getMSE(Original, hazed) << endl;
    cout << "mse of original: " << getMSE(Original, Original) << endl;
}