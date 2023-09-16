#include <opencv2/highgui/highgui_c.h>
#include <time.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

using namespace cv;
using namespace std;

// int getRedMinChannel(Mat src) {
//     Mat redmin = Mat::zeros(src.rows, src.cols, CV_8UC1);
//     Vec3b intensity;
//     for (int m = 0; m < src.rows; m++) {
//         for (int n = 0; n < src.cols; n++) {
//             intensity = src.at<Vec3b>(m, n);
//             redmin.at<uchar>(m, n) = intensity.val[2];
//         }
//     }
//     redmin = filter_darkest(redmin, int(redmin.rows * redmin.cols * 0.1));
//     double max, min;
//     minMaxLoc(redmin, &min, &max);
//     return min;
// }

// median filtered dark channel
Mat getMedianDarkChannel(Mat src, int patch) {
    Mat rgbmin = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat MDCP;
    Vec3b intensity;

    for (int m = 0; m < src.rows; m++) {
        for (int n = 0; n < src.cols; n++) {
            intensity = src.at<Vec3b>(m, n);
            rgbmin.at<uchar>(m, n) = min(intensity.val[0], intensity.val[1]);
        }
    }
    medianBlur(rgbmin, MDCP, patch);
    return MDCP;
}

double getMean(Mat color) {
    if (color.type() == CV_8U) {
        int intensity = 0;
        for (int m = 0; m < color.rows; m++) {
            for (int n = 0; n < color.cols; n++) {
                intensity += (int)color.at<uchar>(m, n);
            }
        }
        if (color.rows * color.cols == 0) {
            // cout << "denominator cannot be zero" << endl;
            return 0;
        } else {
            return (double)(intensity / (color.rows * color.cols));
        }
    } else {
        // cout << "cannot find mean from a multi channel image" << endl;
        return 0;
    }
}

// estimate airlight by the brightest pixel in dark channel (proposed by He
// et al.)
int estimateA2(Mat DC, Mat origin) {
    // determine whether the numberator to be green or blue
    // blue - 0, green - 1
    Mat rgb[3];
    split(origin, rgb);
    // find the max mean between blue and green channel as the numerator
    int num = getMean(rgb[0]) > getMean(rgb[1]) ? 0 : 1;
    Mat ratio = Mat::zeros(origin.rows, origin.cols, CV_8UC1);

    for (int m = 0; m < origin.rows; m++) {
        for (int n = 0; n < origin.cols; n++) {
            if (ratio.type() == CV_8U && rgb[num].type() == CV_8U &&
                rgb[2].type() == CV_8U) {
                if ((int)rgb[2].at<uchar>(m, n) != 0) {
                    ratio.at<uchar>(m, n) = (int)rgb[num].at<uchar>(m, n) /
                                            (int)rgb[2].at<uchar>(m, n);
                }
            } else {
                cout << "ratio should be single channel" << endl;
            }
        }
    }
    double min, max;
    Point2i minLoc, maxLoc;
    minMaxLoc(ratio, &min, &max, &minLoc, &maxLoc);
    // cout << maxLoc.x << "  " << maxLoc.y << endl;
    // cout << "the estimate A is " << (int)origin.at<uchar>(maxLoc.x, maxLoc.y)
        //  << endl;
    return (int)DC.at<uchar>(maxLoc.x, maxLoc.y) * 1.5 > 255
               ? 255
               : (int)DC.at<uchar>(maxLoc.x, maxLoc.y) * 1.5;
}

int estimateA(Mat DC, Mat origin) {
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
            for (int k = 0; k < 2; k++) {
                dehazed.at<Vec3b>(i, j)[k] =
                    abs((intsrc.val[k] - al) / tmax + al) > 255
                        ? 255
                        : abs((intsrc.val[k] - al) / tmax + al);
            }
            dehazed.at<Vec3b>(i, j)[2] = intsrc.val[2];
        }
    }
    return dehazed;
}

// Mat getSecDehazed(Mat src) {
//     Mat redmin = Mat::zeros(src.rows, src.cols, CV_8UC1);
//     Vec3b intensity;
//     for (int m = 0; m < src.rows; m++) {
//         for (int n = 0; n < src.cols; n++) {
//             intensity = src.at<Vec3b>(m, n);
//             redmin.at<uchar>(m, n) = intensity.val[2];
//         }
//     }
//     redmin = filter_darkest(redmin, int(redmin.rows * redmin.cols * 0.1));
//     imshow("darkest red", redmin);
//     for (size_t i = 0; i < redmin.rows; ++i) {
//         for (size_t j = 0; j < redmin.cols; ++j) {
//             if (redmin.at<int>(i, j) == 0) {
//                 src.at<Vec3b>(i, j).val[0] *= 0.5;  // blue channel
//                 src.at<Vec3b>(i, j).val[2] *= 1.5;  // red channel
//             }
//         }
//     }
//     return src;
// }

double getPSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);  // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2];  // sum channels

    if (sse <= 1e-10)  // for small values return zero
        return 0;
    else {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

// get psnr
// int main(int argc, char** argv) {
//     // read all images in file ..\\images
//     std::vector<cv::String> fn;
//     std::string folder(
//         "D:"
//         "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
//         "on\\underwaterImages\\RGBFilter\\Green\\*.jpg");
//     glob(folder, fn, false);
//     Mat standard = imread(
//         "D:"
//         "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
//         "on\\underwaterImages\\RGBFilter\\Original\\Original.jpg");
//     vector<Mat> images;
//     size_t count = fn.size();  // number of jpg files in images folder
//     for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));
//     for (size_t i = 0; i < count; i++) {
//         cout << (double)getPSNR(standard, images[i]) << endl;
//     }
//     return 0;
// }

// resize
// int main(int argc, char** argv) {
//     // read all images in file ..\\images
//     std::vector<cv::String> fn;
//     std::string folder(
//         "D:"
//         "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
//         "on\\underwaterImages\\Scale\\Green\\*.png");
//     glob(folder, fn, false);
//     int width = 156;
//     int height = 215;
//     vector<Mat> images;
//     size_t count = fn.size();  // number of jpg files in images folder
//     for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));
//     for (size_t i = 0; i < count; i++) {
//         Mat fog = images[i];
//         Mat newSize;
//         resize(images[i], newSize, Size(width, height), INTER_LINEAR);
//         stringstream ss1;
//         string name =
//             "D:\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementation\\underwaterImages\\Scale\\Green\\Resized\\";
//         string type = ".jpg";
//         ss1 << name << i << type;
//         string filename = ss1.str();
//         ss1.str("");
//         imwrite(filename, newSize);
//     }
//     return 0;
// }

// output all ratios
// int main(int argc, char** argv) {
//     // read all images in file ..\\images
//     std::vector<cv::String> fn;
//     std::string folder(
//         "D:"
//         "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
//         "on\\underwaterImages\\RGBFilter\\*.jpg");
//     glob(folder, fn, false);
//     vector<Mat> images;
//     size_t count = fn.size();  // number of jpg files in images folder
//     for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));
//     cout << count << endl;
//     for (size_t i = 0; i < count; i++) {
//         Mat fog = images[i];
//         Mat rgb[3];
//         split(fog, rgb);
//         double redMean = getMean(rgb[2]);
//         double greemMean = getMean(rgb[1]);
//         double blueMean = getMean(rgb[0]);
//         string dominate = "blue";
//         int maxMean = max(redMean, max(greemMean, blueMean));
//         if (maxMean == redMean) {
//             dominate = "red";
//         } else if (maxMean == greemMean) {
//             dominate = "green";
//         }
//         cout << "the dominate color of image " << i << " is " << dominate
//              << endl;
//         Mat darkChannel;
//         Mat allCom = Mat::zeros(fog.rows, 6 * fog.cols, CV_8UC3);
//         // namedWindow("all combinations", CV_WINDOW_FULLSCREEN);
//         int patch = 5;
//         darkChannel = getMedianDarkChannel(images[i], patch);
//         int col = 0;
//         for (int num = 0; num < 3; ++num) {
//             for (int dem = 0; dem < 3; ++dem) {
//                 if (num != dem) {
//                     int Airlight = estimateA2(darkChannel, images[i], num,
//                     dem); Mat T = estimateTransmission(darkChannel,
//                     Airlight); Mat fogfree = getDehazed(images[i], T,
//                     Airlight); Rect roi(col * fog.cols, 0, fog.cols,
//                     fog.rows);
//                     stringstream ss1;
//                     string name =
//                         "D:"
//                         "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\"
//                         "codeImplemen"
//                         "tation\\underwaterImages\\RGBFilter\\Blue\\dehazed_";
//                     string type = ".jpg";
//                     ss1 << name << i << "_" << num << dem << type;
//                     string filename = ss1.str();
//                     ss1.str("");
//                     imwrite(filename, fogfree);
//                     col++;
//                 }
//             }
//         }
//         // imshow("all combinations", allCom);
//         // waitKey();
//     }
//     return 0;
// }

int main(int argc, char** argv) {
    // read all images in file ..\\images
    // clock_t start, end;
    // start = clock();
    std::vector<cv::String> fn;
    std::string folder(
        "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\entropy\\raw-890\\*.png");
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
        // namedWindow("before and after", CV_WINDOW_FULLSCREEN);
        // cout << "break point 1" << endl;

        int patch = 5;
        darkChannel = getMedianDarkChannel(images[i], patch);
        // cout << "break point 2" << endl;

        Airlight = estimateA(darkChannel, images[i]);
        // cout << "break point 3" << endl;

        T = estimateTransmission(darkChannel, Airlight);
        // cout << "break point 4" << endl;

        fogfree = getDehazed(images[i], T, Airlight);

        fog.copyTo(beforeafter(roil));
        fogfree.copyTo(beforeafter(roir));

        stringstream ss1, ss2;

        string name =
            "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\assessment\\entropy\\dehazed_P\\P_";
        string name2 =
            "D:"
        "\\mine\\Waterloo\\F21coop\\UnderWaterImageProcessing\\codeImplementati"
        "on\\underwaterImages\\dehazed_ref\\compared_";
        string type = ".jpg";

        ss1 << name << i << type;
        ss2 << name2 << i << type;

        string filename = ss1.str();
        ss1.str("");

        string resultName = ss2.str();
        ss2.str("");

        imwrite(filename, fogfree);
        imwrite(resultName, beforeafter);
        images[i] = fogfree;
        // imshow("before and after", beforeafter);
        // waitKey();
    }
    // end = clock();
    // cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
    return 0;
}

// for testing
// int main(int argc, char** argv) {
//     vector<String> fn;
//     glob("..\\underwaterImages\\*.jpg", fn, false);

//     vector<Mat> images;
//     size_t count = fn.size();  // number of jpg files in images folder
//     for (size_t i = 0; i < count; i++) images.push_back(imread(fn[i]));

//     Point2i loc{1, 0};
//     BPTrans(images[2], 5);
// }