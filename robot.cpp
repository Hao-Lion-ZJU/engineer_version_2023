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

#define picture
// #define Video

/**
 * @brief Construct a new Rmversion object
 */
Rmversion::Rmversion()
{
    arrow_Imgprocess_Ptr = make_unique<arrow_Imgprocess>();
    PoseSlover_Ptr = make_unique<PoseSlover>();
}

/**
 * @brief 释放智能指针
 */
Rmversion::~Rmversion()
{
    arrow_Imgprocess_Ptr.release();
    PoseSlover_Ptr.release();
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
void Rmversion::Graber()
{

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
