/**
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 * @File   Match.hpp
 * @brief  
 * @Author : Hao Lion(郝亮亮)    Email : haolion_zju@foxmail.com
 * @Version  1.0
 * @Creat Date : 2022-11-16
 * 
 * @verbatim
 * ==============================================================================
 *                                                                               
 * ==============================================================================
 * @endverbatim
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 */

#pragma once

#ifndef _MATCH_HPP_
#define _MATCH_HPP_

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


enum COLOR {
    GRAY   = 0,
    BLUE   = 1,
    RED    = 2,
    PURPLE = 3,
};

/**
 * @brief 角标类型
 */
enum arrow_type_e
{
	Small=0,
	Big=1
};

/**
 * @brief 检测结果
 */
enum Arrow_Flag_e
{
	CANNOT_SOLVE = 0,
	PNP_SOLVE = 1,
	ANGLE_SOLVE
};



/**
 * @brief 描述角标类
 */
class arrow_info
{
public:
	/**
	 * @brief Construct a new arrow::arrow object
	 */
	arrow_info();
	arrow_info(cv::RotatedRect rect)
	{
		rect.points(Points);
		width = rect.size.width;
		height = rect.size.height;
		center = rect.center;
		area = rect.size.area();
	}

	/**
	 * @brief 清除角标信息
	 */
	void clear()
	{
		score = 0;
		center = cv::Point2f(0, 0);
		corners.clear();
		for(int i = 0; i < 4; i++)
		{
			Points[i] = cv::Point2f(0, 0);
		}
	}

	/**
	 * @brief 返回顶点坐标
	 * @return const cv::Point2f* 
	 */
	const cv::Point2f* get_vertex() const
	{
		return Points;
	}

	/**
	 * @brief 返回角点坐标
	 * @return const vector<Point2f> 
	 */
	const vector<Point2f> get_corners() const
	{
		return corners;
	}

	/**
	 * @brief 返回装甲板中心坐标
	 * @return const cv::Point2f* 
	 */
	const cv::Point2f get_center() const
	{
		return this->center;
	}

	/**
	 * @brief Destroy the arrow info object
	 */
	~arrow_info() = default;

	/**
	 * @brief 返回角标矩形
	 * @return cv::RotatedRect 
	 */
	cv::RotatedRect rec() const
	{
		return cv::RotatedRect(center, cv::Size2f(width, height), angle);
	}
	/**
	 * @brief 计算两个角标中心点距离
	 * @param  pt               My Param doc
	 * @return float 
	 */
	float distance(const cv::Point2f & pt) const
	{
		return std::sqrt(std::pow((pt.x - this->center.x), 2) + std::pow((pt.y - this->center.y), 2));
	}

public:
	cv::Point2f Points[4];
	float height;
	float area;
	float width;
	float score;
	cv::Point2f center;
	float angle;
	vector<Point2f> corners;//检测出的角点
	Mat warpPerspective_img; //warpPerspective Image
	arrow_type_e arrow_type; //arrow_type
};


class arrow_classifier
{
public:
	/**
 	* @brief Construct a new arrow classifier object
 	*/
	arrow_classifier()
	{

	}
	/**
	 * @brief 载入模板图片
	 * @param  template_path    模板路径
	 * @param  arrowImgSize     模板图片尺寸
	 */
	void Load_Template(const char* big_template_path,const char* small_template_path,Size arrowImgSize= Size(40,40));
	/**
	 * @brief	将图像旋转一定角度
	 * @param  src              原图像
	 * @param  angle            旋转角度
	 * @return Mat 				返回旋转后的图像
	 */
	Mat ImageRotate(Mat src,float angle);
	/**
	 * @brief 多角度模板匹配得到箭头图像置信度
	 * @param  arrow         	箭头
	 * @return const float 		置信度分数
	 */
	const float Arrow_confidence(arrow_info& arrow);
	/**
	 * @brief 载入形态学操作后的图像
	 * @param  srcImg           My Param doc
	 */
	void PreLoad(const Mat & srcImg)
	{
		//copy srcImg as warpPerspective_src
    	(srcImg).copyTo(warpPerspective_src);
	}
	


private:
	Mat Big_arrow_template;		//大角标模板图片
	Mat Small_arrow_template;	//小角标模板图片
	Size arrowImgSize; 		//Template Img size 模板匹配的识别图片大小
	Mat warpPerspective_src; //warpPerspective srcImage  透射变换的原图
    Mat warpPerspective_dst; //warpPerspective dstImage   透射变换生成的目标图
    Mat warpPerspective_mat; //warpPerspective transform matrix 透射变换的变换矩阵
    Point2f srcPoints[4];   //warpPerspective srcPoints		透射变换的原图上的目标点 
    Point2f dstPoints[4];	//warpPerspective dstPoints     透射变换的目标图中的点   
};

class arrow_Imgprocess
{
public:
	/**
	 * @brief Construct a new arrow Imgprocess::arrow Imgprocess object
	 */
	arrow_Imgprocess();
	/**
	 * @brief Destroy the arrow Imgprocess object
	 */
	~arrow_Imgprocess() = default;
	/**
	 * @brief Init object
	 */
	void Init();
	/**
	 * @brief 可视化检测结果
	 */
	void Draw_bound();
	/**
	 * @brief 载入原图像并对其进行预处理
	 * @param  src              原图像
	 */
	void pretreatment(const Mat &src);
	/**
	 * @brief 识别角标主函数
	 */
	Arrow_Flag_e Arrow_detector();
	/**
	 * @brief 寻找角标最外面的角点
	 * @param  arrow            My Param doc
	 * @return vector<Point2f> 角点坐标
	 */
	vector<Point2f> Corner_detector(arrow_info& arrow);

	/**
	 * @brief 返回角点坐标
	 * @return const vector<Point2f> 
	 */
	std::vector<Point2f> get_PointOfPnp() const;

	/**
	 * @brief 展示过程图像
	 */
	void debug_show();

private:
	std::vector<arrow_info> arrows;
	Arrow_Flag_e Arrow_Flag;
	cv::Mat Gray_Img;
	cv::Mat Src_Img;
	cv::Mat Bin_Img;
	cv::Mat Pre_Img;
	cv::Mat Thin_Img;
	//cv::Mat Bin_Img_red;
	Mat arrowDisplay; //Image for the use of displaying arrow
	cv::Point2f ImgCenter;
	arrow_classifier classifier;
	
};


#endif
