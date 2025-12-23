#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

void rgb2yuv_opencv(const Mat& bgr, Mat& yuv)
{
    cvtColor(bgr, yuv, COLOR_BGR2YUV);
}

void yuv2rgb_opencv(const Mat& yuv, Mat& bgr)
{
    cvtColor(yuv, bgr, COLOR_YUV2BGR);
}

int main()
{
    string img_path = "/home/cat/RGB_YUV/test_picture.jpg";

    Mat bgr = imread(img_path, IMREAD_COLOR);
    if (bgr.empty())
    {
        cerr << "Failed to load image: " << img_path << endl;
        return -1;
    }

    cout << "Image loaded: "
         << bgr.cols << " x " << bgr.rows << endl;

    //BGR -> YUV
    Mat yuv;
    auto t1 = high_resolution_clock::now();
    rgb2yuv_opencv(bgr, yuv);
    auto t2 = high_resolution_clock::now();

    imwrite("yuv_picture_cv.png", yuv);

    ////YUV -> BGR
    //Mat bgr_recover;
    //yuv2rgb_opencv(yuv, bgr_recover);
    //imwrite("bgr_recover.png", bgr_recover);

    auto time = duration_cast<microseconds>(t2 - t1).count();
    cout << "time using : " << time << "us" << endl;

    cout << "Conversion finished." << endl;
    cout << "Generated files:" << endl;
    cout << "  yuv_image.png" << endl;


    return 0;
}


//g++ rgb_yuv_cv.cpp -O3 -o rgb_yuv_cv `pkg-config --cflags --libs opencv4`
//./rgb_yuv_cv

