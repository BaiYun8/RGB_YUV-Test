//该版本是把opencv代码改成cpp代码转化
#include <opencv2/opencv.hpp>   
#include <iostream>             
#include <chrono>               
#include <algorithm>            
#include <cstdint>              

using namespace std;
using namespace cv;
using namespace std::chrono;

void rgb2yuv_scalar(const Mat& bgr, Mat& yuv)
{
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    int rows = bgr.rows;  
    int cols = bgr.cols;  

    // 逐行遍历
    for(int i = 0; i < rows; ++i)
    {
        // 获取当前行的输入/输出内存首地址
        const uchar* src = bgr.ptr<uchar>(i);  
        uchar* dst = yuv.ptr<uchar>(i);        
        
        for (int j = 0; j < cols; ++j)
        {
            // 提取当前像素的 B/G/R 
            int B = src[3*j + 0];  
            int G = src[3*j + 1];  
            int R = src[3*j + 2];  

            // BT.601 
            int Y = (77*R + 150*G + 29*B) >> 8;
            int U = (-43*R - 85*G + 128*B + 32768) >> 8;
            int V = (128*R - 107*G - 21*B + 32768) >> 8;

            dst[3*j + 0] = (uchar)std::clamp(Y, 0, 255);  
            dst[3*j + 1] = (uchar)std::clamp(U, 0, 255);  
            dst[3*j + 2] = (uchar)std::clamp(V, 0, 255);  
        }
    }
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
    cout << "Image loaded: " << bgr.cols << " x " << bgr.rows << endl;

    Mat yuv;

    auto t1 = high_resolution_clock::now();  
    rgb2yuv_scalar(bgr, yuv);                
    auto t2 = high_resolution_clock::now();  

    auto duration = duration_cast<microseconds>(t2 - t1).count();
    cout << "RGB2YUV scalar conversion time: " << duration << " us" << endl;

    bool save_ok = imwrite("yuv_scalar.png", yuv);
    if (!save_ok)
    {
        cerr << "Failed to save YUV image!" << endl;
        return -1;
    }

    cout << "Conversion finished." << endl;
    cout << "Generated file: " << endl;
    cout << " - yuv_scalar.png (YUV 444 格式)" << endl;

    return 0;
}

//g++ rgb_yuv_c.cpp -O3 -o rgb_yuv_c `pkg-config --cflags --libs opencv4`
//./rgb_yuv_c

