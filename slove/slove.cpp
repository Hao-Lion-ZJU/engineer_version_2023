/**
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 * @file   slove.cpp
 * @brief  解算角度
 * @author Hao Lion(郝亮亮)    Email:(haolion_zju@foxmail.com)
 * @version 1.0
 * @date 2022-12-08
 * 
 * @verbatim:
 * ==============================================================================
 *                                                                               
 * ==============================================================================
 * @endverbatim
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 */


#include "slove.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "math.h"
#include "params.hpp"
using namespace cv;
using namespace std;



/*bool solvePnP(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs,
 OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int flags=ITERATIVE )

　　objectPoints：特征点的世界坐标，坐标值需为float型，不能为double型，可以为mat类型，也可以直接输入vector
　　imagePoints：特征点在图像中的像素坐标，可以输入mat类型，也可以直接输入vector，
	注意输入点的顺序要与前面的特征点的世界坐标一一对应
　　cameraMatrix：相机内参矩阵
　　distCoeffs：相机的畸变参数
　　rvec：输出的旋转向量
　　tvec：输出的平移向量
　　最后的输入参数有三个可选项：
　　CV_ITERATIVE，默认值，它通过迭代求出重投影误差最小的解作为问题的最优解。
　　CV_P3P则是使用非常经典的Gao的P3P问题求解算法。
　　CV_EPNP使用文章《EPnP: Efficient Perspective-n-Point Camera Pose Estimation》中的方法求解。*/
  //由于装甲板是一个平板，四个顶点的Z轴坐标可以设为0，x轴和y轴坐标可以分别设置为正负二分之一的长和宽，装甲板中心即为原点

static const vector<Point3d> POINT_3D_OF_SLOT = { // 单位：mm
    {-264, 264, 0.},
    {-264, -264, 0.},
    {264, -264, 0.},
    {264, 264, 0.}};

PoseSlover::PoseSlover()
{
	//将四个像素点坐标初始化为(0, 0)
	for(int ll = 0; ll <= 3; ll++)
		point_2d_of_slot.push_back(cv::Point2f(0.0, 0.0));
}




void PoseSlover::setTarget(const std::vector<cv::Point2f> objectPoints)
{
	point_2d_of_slot = objectPoints;
}



PoseSlover::AngleFlag PoseSlover::solve()
//类外定义PoseSlover类的成员函数solve，返回PoseSlover::AngleFlag类型的数据
{
    {
        //使用solvePNP求解相机姿态和位置信息  rvec：输出的旋转向量，tvec：输出的平移相量
        //平移向量就是以当前摄像头中心为原点时物体原点所在的位置。
        solvePnP(POINT_3D_OF_SLOT, point_2d_of_slot, param.camera.CI_MAT, param.camera.D_MAT, _rVec, _tVec, false, SOLVEPNP_ITERATIVE);

        pose.x = _tVec.at<double>(0, 0);
        pose.y = _tVec.at<double>(1, 0);
        pose.z = _tVec.at<double>(2, 0);
        pose.roll = atan2(_rVec.at<double>(2, 1),_rVec.at<double>(2, 2))/CV_PI*180;
        pose.pitch = atan2(-_rVec.at<double>(2,0),
        sqrt(_rVec.at<double>(2,0)*_rVec.at<double>(2,0)+_rVec.at<double>(2,2)*_rVec.at<double>(2,2)))/CV_PI*180;
        pose.yaw =  atan2(_rVec.at<double>(1, 0),_rVec.at<double>(0, 0))/CV_PI*180;
        
        return ANGLES_AND_DISTANCE;
    }
    return ANGLE_ERROR;
}





void PoseSlover::ITERATIVE_solve()
{
    slover_algorithm="ITERATIVE_Solve";
}

void PoseSlover::EPNP_solve()
{
    slover_algorithm="EPNP_Solve";
}



//获取偏转的角度向量
const cv::Vec3f PoseSlover::getPosition()
{
    return cv::Vec3f(pose.x, pose.y, pose.z);
}

const cv::Vec2f PoseSlover::getDev()
{
    return cv::Vec2f(_tVec.at<float>(0, 0), _tVec.at<float>(1, 0));
}



void PoseSlover::showPoseInfo()
{
    Mat angleImage = Mat::zeros(350,600,CV_8UC3);
    putText(angleImage, "Roll: " + to_string(pose.roll), Point(100, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    putText(angleImage, "Pitch: " + to_string(pose.pitch), Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    putText(angleImage, "Yaw: " + to_string(pose.yaw), Point(100, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    putText(angleImage, "X:" + to_string((int)pose.x), Point(50, 200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    putText(angleImage, "Y:" + to_string((int)pose.y), Point(200, 200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    putText(angleImage, "Z:" + to_string((int)pose.z), Point(350, 200), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    putText(angleImage, ("Algorithm:" + slover_algorithm),Point(100,250), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 1, 8, false);
    imshow("PoseSlover",angleImage);

}



