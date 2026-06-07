读取一张 BGR 格式图片，将其转换为 YUV444 格式，并保存转换结果。三者的区别主要在于 RGB/BGR 到 YUV 的实现方式不同：
rgb_yuv_cv.cpp 使用 OpenCV 官方接口完成 BGR -> YUV 转换;
rgb_yuv_c.cpp 手写标量版本，用 C/C++ 循环逐像素计算 YUV;
rgb_yuv_neon_8.cpp 使用 ARM NEON SIMD 指令优化转换速度


rgb_yuv_cv.cpp核心功能：
使用 imread 读取 /home/cat/RGB_YUV/test_picture.jpg。
调用 cvtColor(bgr, yuv, COLOR_BGR2YUV) 将 BGR 图像转换为 YUV 图像。
使用 chrono 统计转换耗时，单位为微秒。
将转换后的结果保存为 yuv_picture_cv.png。
预留了yuv2rgb_opencv 函数，可以将 YUV 转回 BGR。
该文件用来验证手写 C 版本和 NEON 优化版本的转换结果是否正确。



rgb_yuv_c.cpp 核心函数是：
void rgb2yuv_scalar(const Mat& bgr, Mat& yuv)，主要流程：
检查输入图像格式必须是 CV_8UC3，即 8 位三通道图像。
按行遍历图像，每次读取一个像素的 B/G/R 三个通道。
使用 BT.601 近似整数公式计算 Y、U、V：
Y = (77*R + 150*G + 29*B) >> 8;
U = (-43*R - 85*G + 128*B + 32768) >> 8;
V = (128*R - 107*G - 21*B + 32768) >> 8;
使用 std::clamp 将结果限制在 [0, 255] 范围内。
输出结果保存为 yuv_scalar.png。
这个文件适合作为基础版本或性能对照版本，用来展示最直接的逐像素 RGB/YUV 转换逻辑。

rgb_yuv_neon_8.cpp 是 ARM NEON 优化版本，用于提升 BGR 到 YUV 的转换性能。它包含两个 NEON 实现函数：
void rgb2yuv_8(const cv::Mat& bgr, cv::Mat& yuv)按行访问版本：
每次从连续内存中读取 8 个 BGR 像素。
使用 vld3_u8 将 B、G、R 三个通道拆分加载到 NEON 向量。
使用 NEON 向量乘法、累加、移位等指令并行计算 8 个像素的 Y/U/V。
使用 vst3_u8 将 YUV 三通道结果写回输出图像。
对不能被 8 整除的尾部像素，回退到普通标量计算。


void rgb2yuv_8_col(const cv::Mat& bgr, cv::Mat& yuv)是按列访问版本：
外层按列遍历，内层每次处理同一列上的 8 行像素。
由于同一列的 8 个像素在内存中不是连续的，所以先用临时数组 Btmp/Gtmp/Rtmp 收集数据，再加载到 NEON 向量中计算。
计算完成后逐行写回结果。
同样对不足 8 行的尾部数据使用标量方式处理。

性能展示：
1.使用cvtColor 接口，
<img width="915" height="441" alt="ef8e476135942ca354b99aa0f67fc338" src="https://github.com/user-attachments/assets/676e58de-7c78-4d78-84f1-9bfa21c15890" />  
2.NEON指令按行访问：perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./rgb_yuv_row
<img width="915" height="438" alt="1a44041fd5ffb70b5d29731d60092ccd" src="https://github.com/user-attachments/assets/aafdec62-2e1c-43f4-8319-262ab8f385f3" />  
3.NEON指令按列访问：perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./rgb_yuv_col
<img width="915" height="435" alt="c21666717bf2ea31b20aeb2df4b3ac4b" src="https://github.com/user-attachments/assets/651f3095-d231-45c3-a107-5c659b31e371" />  

获得cache-references 和 cache-misses后
cache hit = cache-references - cache-misses



