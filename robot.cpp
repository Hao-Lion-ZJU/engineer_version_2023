/**
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 * @file   robot.cpp
 * @brief 线程函数
 * @author Hao Lion(郝亮亮)    Email:(haolion_zju@foxmail.com)
 * @version 1.0
 * @date 2022-10-01
 * 
 * @verbatim:
 * ==============================================================================
 *                                                                               
 * ==============================================================================
 * @endverbatim
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 */

#include "robot.hpp"
#include <glog/logging.h>
#include "params.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>

#define picture
// #define Video


bool RUNNING = true;

// ------------ 线 程 同 步 锁 ------------
//                  pkg2post
// pre(get frame)  ----------> post(to uart)
static mutex pkg2post_mtx;
static condition_variable cond2post;

/**
 * @brief Construct a new Rmversion object
 */
Rmversion::Rmversion()
{
    arrow_Imgprocess_Ptr = make_unique<arrow_Imgprocess>();
    PoseSlover_Ptr = make_unique<PoseSlover>();
    Communicator_Ptr = make_unique<Communicator>(param.enable_serial);
    Camera_Ptr = make_unique<MVCamera>(param.mv_camera_name);
    Record_Ptr = make_unique<Record>(param.save_video);
}

/**
 * @brief 释放智能指针
 */
Rmversion::~Rmversion()
{
    arrow_Imgprocess_Ptr.release();
    PoseSlover_Ptr.release();
    Communicator_Ptr.release();
    Camera_Ptr.release();
    Record_Ptr.release();
}

/**
 * @brief 初始化一些标志位
 */
void Rmversion::Init()
{
    flags = CANNOT_SOLVE;
    /*调整下窗口位置，方便DEBUG*/
    this->resize_window();

    
}

/**
 * @brief 图像处理线程
 */
void Rmversion::Imgprocess()
{
    Mat src;
#ifdef Video
    VideoCapture cap("/home/haolion/test/rm/engineer_version/demo/robot.mp4");
    do {
        //时间消耗测试
        double t, tc;
        t = getTickCount();
        cap >> src;
        // resize(src, src, Size(640, 480));
        arrow_Imgprocess_Ptr->pretreatment(src);
        arrow_Imgprocess_Ptr->Arrow_detector();
        //时间消耗测试
        tc = (getTickCount() - t) / getTickFrequency();
        printf("time consume %.5f   fps %.2f\n", tc, 1 / tc);    //显示出耗费时间的多少
        #ifdef DEBUG
        arrow_Imgprocess_Ptr->Draw_bound();
        arrow_Imgprocess_Ptr->debug_show();
        waitKey(1);
        #endif//DEBUG
    } while (true);
#endif
#ifdef picture
    uint8_t match_flag = 0;
    std::vector<Point2f> PointOfSlot;
    char filename[100];
    for (size_t i = 0; i < 6; i++)
    {
        /* code */
        sprintf(filename, "/home/haolion/test/rm/engineer_version/Image/%d.jpg", i);
        src = imread(filename);
         arrow_Imgprocess_Ptr->pretreatment(src);
        match_flag = arrow_Imgprocess_Ptr->Arrow_detector(); 
        PointOfSlot = arrow_Imgprocess_Ptr->get_PointOfPnp();
        if(match_flag == PNP_SOLVE)
        {
            PoseSlover_Ptr->setTarget(PointOfSlot);
            PoseSlover_Ptr->solve();
        }
        if(param.show_debug)
        {
            arrow_Imgprocess_Ptr->Draw_bound();
            arrow_Imgprocess_Ptr->debug_show();
        }
        if(param.show_pose)
        {
            PoseSlover_Ptr->showPoseInfo();
        }
    }
    
#endif
}


/**
 * @brief 读取图像到缓冲区
 */
[[noreturn]] void Rmversion::Graber()
{

    Mat src_img;

    while( RUNNING ) {

        // TIME_BEGIN()
        bool status = Camera_Ptr->getFrame(src_img);
        // TIME_END("get Frame")
        if ( false == status ) {
            LOG(WARNING) << "Get Image Failed !";
            this_thread::sleep_for(chrono::milliseconds(10));
            continue;
        }
        if ( param.save_video )
            Record_Ptr->push(src_img);

        { // 将数据发送至推理
            unique_lock<mutex> lk(pkg2post_mtx);

            pkg2post.time_stamp = chrono::system_clock::now();
            pkg2post.robot = g_robot;
            src_img.copyTo(pkg2post.src_img);

            pkg2post.updated = true;
            lk.unlock();
            // 唤醒推理线程
            cond2post.notify_one();
        }
    }
    this_thread::sleep_for(chrono::milliseconds(20));
    cond2post.notify_all();

}


/**
 * @brief 
 */
[[noreturn]] void Rmversion::Receiver()
{
    uint8_t buffer[12] = {0};
    while ( RUNNING ) {
        /// receive
        int num_bytes = Communicator_Ptr->receive(buffer, 7);
        if (num_bytes <= 0) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        if(buffer[0] == 0x7f && buffer[1] == 0x7f && buffer[2] == 0x7f && buffer[3] == 0x7f && buffer[4] == 0x7f && buffer[5] == 0x7f)
        {
            LOG(FATAL) << "-------------------------------------shutdown!!!";
            system("sudo shutdown now");
        }
        else{
            int16_t pitch_angle = buffer[0] << 8 | buffer[1];
            int16_t yaw_angle   = buffer[2] << 8 | buffer[3];
            int16_t move_speed  = buffer[4] << 8 | buffer[5];
            int     mode        = buffer[6];

            g_robot.yaw         = yaw_angle / 100.f;
            g_robot.pitch       = pitch_angle / 100.f;

            // 串口中的比特位（敌方颜色） 蓝0 红1
            param.target_color  = (mode & 0x80) ? COLOR::RED : COLOR::BLUE;

        }
    }
}

/**
 * @brief 重新设置窗口位置
 */
#define SHOW_HEIGHT  450
#define SHOW_WIDTH	600
void Rmversion::resize_window()
{
	cv::namedWindow("gray", cv::WINDOW_NORMAL);
	cv::namedWindow("Pre_bin", cv::WINDOW_NORMAL);
	cv::namedWindow("thin", cv::WINDOW_NORMAL);
	cv::namedWindow("corner", cv::WINDOW_NORMAL);
	cv::namedWindow("PoseSlover", cv::WINDOW_NORMAL);
	cv::namedWindow("result", cv::WINDOW_NORMAL);

    cv::resizeWindow("gray", SHOW_WIDTH,SHOW_HEIGHT);
	cv::resizeWindow("Pre_bin", SHOW_WIDTH,SHOW_HEIGHT);
	cv::resizeWindow("thin", SHOW_WIDTH,SHOW_HEIGHT);
	cv::resizeWindow("corner", SHOW_WIDTH,SHOW_HEIGHT);
	cv::resizeWindow("PoseSlover", SHOW_WIDTH,SHOW_HEIGHT);
	cv::resizeWindow("result", SHOW_WIDTH,SHOW_HEIGHT);

    cv::moveWindow("gray", 0, 0);
    cv::moveWindow("Pre_bin", SHOW_WIDTH,0);
    cv::moveWindow("thin", 2*SHOW_WIDTH, 0);
    cv::moveWindow("corner", 0, SHOW_HEIGHT);
    cv::moveWindow("PoseSlover", SHOW_WIDTH, SHOW_HEIGHT);
    cv::moveWindow("result", 2*SHOW_WIDTH,SHOW_HEIGHT);
}
