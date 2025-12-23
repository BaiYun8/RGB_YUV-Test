#include <opencv2/opencv.hpp>   
#include <iostream>             
#include <chrono>               
#include <algorithm>            
#include <cstdint>              
#include <arm_neon.h>

using namespace std;
using namespace cv;
using namespace std::chrono;


//按行访问序列
void rgb2yuv_8(const cv::Mat& bgr, cv::Mat& yuv)
{
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    const int rows = bgr.rows;
    const int cols = bgr.cols;

    // Y 系数无符号 
    const uint16x8_t cYR = vdupq_n_u16(77);
    const uint16x8_t cYG = vdupq_n_u16(150);
    const uint16x8_t cYB = vdupq_n_u16(29);

    // U/V 系数有符号 
    const int16x8_t cUR = vdupq_n_s16(-43);
    const int16x8_t cUG = vdupq_n_s16(-85);
    const int16x8_t cUB = vdupq_n_s16(128);

    const int16x8_t cVR = vdupq_n_s16(128);
    const int16x8_t cVG = vdupq_n_s16(-107);
    const int16x8_t cVB = vdupq_n_s16(-21);

    const int32x4_t offset32 = vdupq_n_s32(128 << 8);

    for (int i = 0; i < rows; ++i)
    {
        const uchar* src = bgr.ptr<uchar>(i);
        uchar* dst = yuv.ptr<uchar>(i);

        int j = 0;

        // NEON：8 像素 
        for (; j + 8 <= cols; j += 8)
        {
            uint8x8x3_t bgr8 = vld3_u8(src + 3 * j);

            uint16x8_t Bu = vmovl_u8(bgr8.val[0]);
            uint16x8_t Gu = vmovl_u8(bgr8.val[1]);
            uint16x8_t Ru = vmovl_u8(bgr8.val[2]);

            // Y 
            uint16x8_t Y = vmulq_u16(Ru, cYR);
            Y = vmlaq_u16(Y, Gu, cYG);
            Y = vmlaq_u16(Y, Bu, cYB);
            uint8x8_t y = vshrn_n_u16(Y, 8);

            // 转为有符号
            int16x8_t Rs = vreinterpretq_s16_u16(Ru);
            int16x8_t Gs = vreinterpretq_s16_u16(Gu);
            int16x8_t Bs = vreinterpretq_s16_u16(Bu);

            // U 转为int32 累加
            int32x4_t U0 = vmull_s16(vget_low_s16(Rs), vget_low_s16(cUR));
            U0 = vmlal_s16(U0, vget_low_s16(Gs), vget_low_s16(cUG));
            U0 = vmlal_s16(U0, vget_low_s16(Bs), vget_low_s16(cUB));
            U0 = vaddq_s32(U0, offset32);
            U0 = vshrq_n_s32(U0, 8);

            int32x4_t U1 = vmull_s16(vget_high_s16(Rs), vget_high_s16(cUR));
            U1 = vmlal_s16(U1, vget_high_s16(Gs), vget_high_s16(cUG));
            U1 = vmlal_s16(U1, vget_high_s16(Bs), vget_high_s16(cUB));
            U1 = vaddq_s32(U1, offset32);
            U1 = vshrq_n_s32(U1, 8);

            int16x8_t U16 = vcombine_s16(vqmovn_s32(U0), vqmovn_s32(U1));
            uint8x8_t u = vqmovun_s16(U16);

            // V 转为int32 累加
            int32x4_t V0 = vmull_s16(vget_low_s16(Rs), vget_low_s16(cVR));
            V0 = vmlal_s16(V0, vget_low_s16(Gs), vget_low_s16(cVG));
            V0 = vmlal_s16(V0, vget_low_s16(Bs), vget_low_s16(cVB));
            V0 = vaddq_s32(V0, offset32);
            V0 = vshrq_n_s32(V0, 8);

            int32x4_t V1 = vmull_s16(vget_high_s16(Rs), vget_high_s16(cVR));
            V1 = vmlal_s16(V1, vget_high_s16(Gs), vget_high_s16(cVG));
            V1 = vmlal_s16(V1, vget_high_s16(Bs), vget_high_s16(cVB));
            V1 = vaddq_s32(V1, offset32);
            V1 = vshrq_n_s32(V1, 8);

            int16x8_t V16 = vcombine_s16(vqmovn_s32(V0), vqmovn_s32(V1));
            uint8x8_t v = vqmovun_s16(V16);

            // 存储
            uint8x8x3_t yuv8;
            yuv8.val[0] = y;
            yuv8.val[1] = u;
            yuv8.val[2] = v;

            vst3_u8(dst + 3 * j, yuv8);
        }

        // 标量尾部
        for (; j < cols; ++j)
        {
            int B = src[3*j + 0];
            int G = src[3*j + 1];
            int R = src[3*j + 2];

            int Y = (77*R + 150*G + 29*B) >> 8;
            int U = (-43*R - 85*G + 128*B + 32768) >> 8;
            int V = (128*R - 107*G - 21*B + 32768) >> 8;

            dst[3*j + 0] = (uchar)std::clamp(Y, 0, 255);
            dst[3*j + 1] = (uchar)std::clamp(U, 0, 255);
            dst[3*j + 2] = (uchar)std::clamp(V, 0, 255);
        }
    }
}

//按列访问
void rgb2yuv_8_col(const cv::Mat& bgr, cv::Mat& yuv)
{
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    const int rows = bgr.rows;
    const int cols = bgr.cols;

    // Y 系数（无符号）
    const uint16x8_t cYR = vdupq_n_u16(77);
    const uint16x8_t cYG = vdupq_n_u16(150);
    const uint16x8_t cYB = vdupq_n_u16(29);

    // U/V 系数（有符号）
    const int16x8_t cUR = vdupq_n_s16(-43);
    const int16x8_t cUG = vdupq_n_s16(-85);
    const int16x8_t cUB = vdupq_n_s16(128);

    const int16x8_t cVR = vdupq_n_s16(128);
    const int16x8_t cVG = vdupq_n_s16(-107);
    const int16x8_t cVB = vdupq_n_s16(-21);

    const int32x4_t offset32 = vdupq_n_s32(128 << 8);

    // 外层：列
    for (int j = 0; j < cols; ++j)
    {
        int i = 0;

        // 每次处理 8 行
        for (; i + 8 <= rows; i += 8)
        {
            uint8_t Btmp[8], Gtmp[8], Rtmp[8];

            // gather：同一列，8 行
            for (int k = 0; k < 8; ++k)
            {
                const uchar* src = bgr.ptr<uchar>(i + k);
                Btmp[k] = src[3*j + 0];
                Gtmp[k] = src[3*j + 1];
                Rtmp[k] = src[3*j + 2];
            }

            uint8x8_t B8 = vld1_u8(Btmp);
            uint8x8_t G8 = vld1_u8(Gtmp);
            uint8x8_t R8 = vld1_u8(Rtmp);

            uint16x8_t Bu = vmovl_u8(B8);
            uint16x8_t Gu = vmovl_u8(G8);
            uint16x8_t Ru = vmovl_u8(R8);

            // Y
            uint16x8_t Y = vmulq_u16(Ru, cYR);
            Y = vmlaq_u16(Y, Gu, cYG);
            Y = vmlaq_u16(Y, Bu, cYB);
            uint8x8_t y = vshrn_n_u16(Y, 8);

            int16x8_t Rs = vreinterpretq_s16_u16(Ru);
            int16x8_t Gs = vreinterpretq_s16_u16(Gu);
            int16x8_t Bs = vreinterpretq_s16_u16(Bu);

            // U
            int32x4_t U0 = vmull_s16(vget_low_s16(Rs), vget_low_s16(cUR));
            U0 = vmlal_s16(U0, vget_low_s16(Gs), vget_low_s16(cUG));
            U0 = vmlal_s16(U0, vget_low_s16(Bs), vget_low_s16(cUB));
            U0 = vaddq_s32(U0, offset32);
            U0 = vshrq_n_s32(U0, 8);

            int32x4_t U1 = vmull_s16(vget_high_s16(Rs), vget_high_s16(cUR));
            U1 = vmlal_s16(U1, vget_high_s16(Gs), vget_high_s16(cUG));
            U1 = vmlal_s16(U1, vget_high_s16(Bs), vget_high_s16(cUB));
            U1 = vaddq_s32(U1, offset32);
            U1 = vshrq_n_s32(U1, 8);

            uint8x8_t u = vqmovun_s16(vcombine_s16(vqmovn_s32(U0), vqmovn_s32(U1)));

            // V
            int32x4_t V0 = vmull_s16(vget_low_s16(Rs), vget_low_s16(cVR));
            V0 = vmlal_s16(V0, vget_low_s16(Gs), vget_low_s16(cVG));
            V0 = vmlal_s16(V0, vget_low_s16(Bs), vget_low_s16(cVB));
            V0 = vaddq_s32(V0, offset32);
            V0 = vshrq_n_s32(V0, 8);

            int32x4_t V1 = vmull_s16(vget_high_s16(Rs), vget_high_s16(cVR));
            V1 = vmlal_s16(V1, vget_high_s16(Gs), vget_high_s16(cVG));
            V1 = vmlal_s16(V1, vget_high_s16(Bs), vget_high_s16(cVB));
            V1 = vaddq_s32(V1, offset32);
            V1 = vshrq_n_s32(V1, 8);

            uint8x8_t v = vqmovun_s16(vcombine_s16(vqmovn_s32(V0), vqmovn_s32(V1)));

            // 写回
            for (int k = 0; k < 8; ++k)
            {
                uchar* dst = yuv.ptr<uchar>(i + k);
                dst[3*j + 0] = vget_lane_u8(y, k);
                dst[3*j + 1] = vget_lane_u8(u, k);
                dst[3*j + 2] = vget_lane_u8(v, k);
            }
        }

        // 尾部标量
        for (; i < rows; ++i)
        {
            const uchar* src = bgr.ptr<uchar>(i);
            uchar* dst = yuv.ptr<uchar>(i);

            int B = src[3*j + 0];
            int G = src[3*j + 1];
            int R = src[3*j + 2];

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
    
    //读取 BGR 图像
    Mat bgr = imread(img_path, IMREAD_COLOR);
    if (bgr.empty())
    {
        cerr << "Failed to load image: " << img_path << endl;
        return -1;
    }
    cout << "Image loaded: " << bgr.cols << " x " << bgr.rows << endl;

    Mat yuv;

    auto t1 = high_resolution_clock::now();  
    rgb2yuv_8(bgr, yuv);
    //rgb2yuv_8_col(bgr, yuv);                
    auto t2 = high_resolution_clock::now();  
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    cout << "RGB2YUV conversion time: " << duration << " us" << endl;

    bool save_ok = imwrite("yuv_8.png", yuv);
    if (!save_ok)
    {
        cerr << "Failed to save YUV image!" << endl;
        return -1;
    }

    cout << "Conversion finished." << endl;
    cout << "Generated file: " << endl;
    cout << " - yuv_scalar.png (YUV 444 格式)" << endl;

    //测试信息
    //Mat yuv_cv;
    //cvtColor(bgr, yuv_cv, COLOR_BGR2YUV);
    //Mat yuv_neon;
    //rgb2yuv_8(bgr, yuv_neon); 
    //for (int i = 0; i < 5; i++) {
    //    int x = 100 + i * 50;
    //    int y = 100;
    //    
    //    Vec3b cv_p = yuv_cv.at<Vec3b>(y, x);
    //    Vec3b neon_p = yuv_neon.at<Vec3b>(y, x);
    //    Vec3b bgr_p = bgr.at<Vec3b>(y, x);
    //    
    //    cout << "像素(" << x << "," << y << ") BGR=(" 
    //         << (int)bgr_p[0] << "," << (int)bgr_p[1] << "," << (int)bgr_p[2] << ")" << endl;
    //    cout << "  OpenCV: Y=" << (int)cv_p[0] << " U=" << (int)cv_p[1] << " V=" << (int)cv_p[2] << endl;
    //    cout << "  NEON:   Y=" << (int)neon_p[0] << " U=" << (int)neon_p[1] << " V=" << (int)neon_p[2] << endl;
    //    cout << "  差值:   Y=" << (int)neon_p[0] - (int)cv_p[0] 
    //         << " U=" << (int)neon_p[1] - (int)cv_p[1]
    //         << " V=" << (int)neon_p[2] - (int)cv_p[2] << endl;
    //    cout << endl;
    //}

    return 0;
}

//g++ -O3 -march=armv8-a -o rgb_yuv_neon_8 rgb_yuv_neon_8.cpp `pkg-config --cflags --libs opencv4`
//./rgb_yuv_neon_8