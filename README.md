本实验项目为学习NEON优化技巧，首先使用opencv函数进行转化，其次写一个C++对应的版本来理清处理思路，最后开始写NEON版本：

rgb_yuv_cv.cpp为使用 OpenCV 内置颜色空间转换函数实现的 RGB–YUV 转换版本，主要基于cvtColor 接口。  
rgb_yuv_c.cpp 为使用 C++ 标量每次处理一个像素。  
rgb_yuv_neon_8.cpp 分为两个版本：按列访问和按行访问，每次处理八个像素点，利用NEON 的128位寄存器进行多个数据的同时运算。  

性能展示：
1.使用cvtColor 接口，
<img width="915" height="441" alt="ef8e476135942ca354b99aa0f67fc338" src="https://github.com/user-attachments/assets/676e58de-7c78-4d78-84f1-9bfa21c15890" />  
2.NEON指令按行访问，
<img width="915" height="438" alt="1a44041fd5ffb70b5d29731d60092ccd" src="https://github.com/user-attachments/assets/aafdec62-2e1c-43f4-8319-262ab8f385f3" />  
3.NEON指令按列访问，
<img width="915" height="435" alt="c21666717bf2ea31b20aeb2df4b3ac4b" src="https://github.com/user-attachments/assets/651f3095-d231-45c3-a107-5c659b31e371" />  




