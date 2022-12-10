#pragma once
#include <memory>
#include <iostream>
#include <mutex>

#include "params.hpp"
#include "./Match/Match.hpp"
#include "./alogrithm/alogrithm.hpp"
#include "./pose/slove.hpp"
#include "./device/camera.hpp"
#include "./device/communicator.hpp"
#include "record.hpp"

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
    [[noreturn]] void Graber();
    /**
     * @brief 
     */
    [[noreturn]] void Receiver();
    /**
     * @brief 初始化
     */
    void Init();
    /**
 	* @brief 重新设置窗口位置
 	*/
	void resize_window();
/**********************data**********************/

    // 后处理线程同步包
    struct Pkg2Post{
        cv::Mat src_img;
        std::chrono::time_point<std::chrono::system_clock> time_stamp;
        Robot robot;
        bool updated = false;
    } pkg2post;
private:
    uint8_t flags;
    size_t Buffersize = 6;
    std::unique_ptr<arrow_Imgprocess> arrow_Imgprocess_Ptr;
    std::unique_ptr<PoseSlover> PoseSlover_Ptr;
    std::unique_ptr<Camera> Camera_Ptr;
    std::unique_ptr<Communicator> Communicator_Ptr;
    std::unique_ptr<Record>Record_Ptr;
};



