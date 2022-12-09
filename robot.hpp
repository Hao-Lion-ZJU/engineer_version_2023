#pragma once
#include <memory>
#include <iostream>
#include <mutex>

#include "./Match/Match.h"
#include "./alogrithm/alogrithm.hpp"
#include "./pose/slove.hpp"

using namespace std;
using namespace cv;






class Rmversion{
public:
    /**
     * @brief Construct a new Rmversion object
     */
    Rmversion();
    /**
     * @brief Destroy the Rmversion object
     */
    ~Rmversion();
    /**
     * @brief 图像处理函数
     */
    void Imgprocess();
    /**
     * @brief 读取图像到缓冲区
     */
    void Graber();
    /**
     * @brief 初始化
     */
    void Init();
    /**
 	* @brief 重新设置窗口位置
 	*/
	void resize_window();
private:
    uint8_t flags;
    size_t Buffersize = 6;
    std::unique_ptr<arrow_Imgprocess> arrow_Imgprocess_Ptr;
    std::unique_ptr<PoseSlover> PoseSlover_Ptr;
};



