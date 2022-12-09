/**
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 * @file   Match.cpp
 * @brief 
 * @author Hao Lion(郝亮亮)    Email:(haolion_zju@foxmail.com)
 * @version 1.0
 * @date 2022-12-09
 * 
 * @verbatim:
 * ==============================================================================
 *                                                                               
 * ==============================================================================
 * @endverbatim
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 */

#include "Match.hpp"
#include "opencv2/opencv.hpp"
#include "../alogrithm/alogrithm.hpp"
#include "params.hpp"
#include <glog/logging.h>
using namespace cv;
using namespace std;


/************************************箭头匹配************************************/
/**
 * @brief 载入模板图片
 * @param  template_path    模板路径
 * @param  arrowImgSize     模板图片尺寸
 */
void arrow_classifier::Load_Template(const char* big_template_path,const char* small_template_path,Size arrowImgSize)
{
    this->arrowImgSize = arrowImgSize;
    Big_arrow_template = imread(big_template_path);
    Small_arrow_template = imread(small_template_path);
    if(Big_arrow_template.empty() || Small_arrow_template.empty())
    {
        LOG(ERROR) << "模板读取失败";
        exit(0);
    }
    cvtColor(Big_arrow_template, Big_arrow_template, COLOR_RGB2GRAY);
    threshold(Big_arrow_template, Big_arrow_template,50, 255, cv::THRESH_BINARY);
    cvtColor(Small_arrow_template, Small_arrow_template, COLOR_RGB2GRAY);
    threshold(Small_arrow_template, Small_arrow_template,50, 255, cv::THRESH_BINARY);
    //set dstPoints (the same to arrowImgSize, as it can avoid resize arrowImg)
    dstPoints[0] = Point2f(0, 0);
    dstPoints[1] = Point2f(arrowImgSize.width, 0);
    dstPoints[2] = Point2f(arrowImgSize.width, arrowImgSize.height);
    dstPoints[3] = Point2f(0, arrowImgSize.height);
}

/**
 * @brief	将图像旋转一定角度
 * @param  src              原图像
 * @param  angle            旋转角度
 * @return Mat 				返回旋转后的图像
 */
Mat arrow_classifier::ImageRotate(Mat src,float angle)
{
    Mat newImg;
    Point2f pt = Point2f((float)src.cols / 2, (float)src.rows / 2);
    Mat M = getRotationMatrix2D(pt, angle, param.arrow_para.template_extend_ratio);
    warpAffine(src, newImg, M, src.size());
    return newImg;
}

/**
 * @brief 多角度模板匹配得到箭头图像置信度
 * @param  arrow         	箭头
 * @return const float 		置信度分数
 */
const float arrow_classifier::Arrow_confidence(arrow_info& arrow)
{

    //set the arrow vertex as srcPoints
    for (int i = 0; i < 4; i++)
        srcPoints[i] = arrow.Points[i];

    //get the arrow image using warpPerspective
    warpPerspective_mat = getPerspectiveTransform(srcPoints, dstPoints);  // get perspective transform matrix  透射变换矩阵
    warpPerspective(warpPerspective_src, warpPerspective_dst, warpPerspective_mat, arrowImgSize, INTER_NEAREST, BORDER_CONSTANT, Scalar(0)); //warpPerspective to get arrowImage
    warpPerspective_dst.copyTo(arrow.warpPerspective_img); //copyto arrowImg
    

    Mat Template;
    if(arrow.arrow_type == Big)
    {
        Template = Big_arrow_template;
    }
    else 
    {
        Template = Small_arrow_template;
    }
    auto step = param.arrow_para.mtach_step;
    double value = 0;
    Mat newImg;
    Mat result;
    for (int i = 0; i <= 360 / step; i++)
      {
        newImg = ImageRotate(Template, step * i);
        matchTemplate(warpPerspective_dst, newImg, result, TM_CCOEFF_NORMED);
        double minval, maxval;
        Point minloc, maxloc;
        minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
        if(maxval>value)
        {
            arrow.angle = i;
            value = maxval;
        }
      }
      return value;
}

/************************************************寻找箭头角点**********************************************/

/**
 * @brief Construct a new arrow Imgprocess::arrow Imgprocess object
 */
arrow_Imgprocess::arrow_Imgprocess()
{
    Init();
}

/**
 * @brief Init object
 */
void arrow_Imgprocess::Init()
{
    Arrow_Flag = CANNOT_SOLVE;
    classifier.Load_Template(param.big_template_path.data(), param.small_template_path.data());
}

/**
 * @brief 载入原图像并对其进行预处理
 * @param  src              原图像
 */
void arrow_Imgprocess::pretreatment(const Mat &src)
{
    src.copyTo(Src_Img);
    ImgCenter = Point2f(Src_Img.cols/2, Src_Img.rows/2);
    cvtColor(Src_Img, Gray_Img, COLOR_RGB2GRAY);
    
	GammaTransform(Gray_Img, Gray_Img, param.arrow_para.Gama);

	//阈值化
	threshold(Gray_Img, Bin_Img,param.arrow_para.brightness_threshold, 255, cv::THRESH_BINARY);

    

	//定义椭圆形结构元素，锚点为元素中心点，用于膨胀操作，结构大小为内切3×3矩形的椭圆
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	//开运算，去除白色噪点
	morphologyEx(Bin_Img, Pre_Img, MORPH_DILATE, element);
	element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	morphologyEx(Pre_Img, Pre_Img, MORPH_DILATE, element);
    classifier.PreLoad(Pre_Img);
    /*八邻域算法骨架提取*/
    Thin_Img = thinImage(Pre_Img);
    
}

/**
 * @brief 识别角标主函数
 */
Arrow_Flag_e arrow_Imgprocess::Arrow_detector()
{
    arrows.clear();
    {
        std::vector<std::vector<Point>> arrowContours;
		cv::findContours(Pre_Img.clone(), arrowContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		for (const auto& contour:arrowContours)
		{
            //计算整个轮廓的面积
			float arrowContourArea = contourArea(contour);
            if(contour.size()<=param.arrow_para.arrow_min_size || arrowContourArea<= param.arrow_para.rect_min_area)
                continue;
            cv::RotatedRect rect = minAreaRect(contour);
            float height = MAX(rect.size.height,rect.size.width);
            float width = MIN(rect.size.height,rect.size.width);
            if((height/width)>param.arrow_para.rect_max_ratio)
                continue;
            arrow_info arrow(rect);
            if(height>=param.arrow_para.arrow_max_lenth)
            {
                arrow.arrow_type = Big;
            }
            else
            {
                arrow.arrow_type = Small;
            }
            arrow.score = classifier.Arrow_confidence(arrow);
            if(arrow.score>param.arrow_para.score_threshold)
            {
                arrow.corners = Corner_detector(arrow);
                this->arrows.emplace_back(arrow);
            }  
        }
    }
    if(arrows.size()<3)
    {
        Arrow_Flag = CANNOT_SOLVE;
    }
    else if(arrows.size()==3)
    {
        Arrow_Flag = ANGLE_SOLVE;
    }
    else if(arrows.size()>=4)
    {
        Arrow_Flag = PNP_SOLVE;
    }
    return Arrow_Flag;
}

/**
 * @brief 寻找角标最外面的角点
 * @param  arrow            My Param doc
 * @return vector<Point2f> 角点坐标
 */
vector<Point2f> arrow_Imgprocess::Corner_detector(arrow_info& arrow)
{
    vector<Point2f> corners;
    Rect bbox = arrow.rec().boundingRect();
    Mat mask = Mat::zeros(Src_Img.size(), CV_8UC1); 
    mask(bbox).setTo(255);
    int maxcorners = 1;
    double qualityLevel = 0.1;  //角点检测可接受的最小特征值
	double minDistance = 10;	//角点之间最小距离
	int blockSize = 7;//计算导数自相关矩阵时指定的领域范围
	double  k = 0.04; //权重系数
    goodFeaturesToTrack(Thin_Img,corners,maxcorners,qualityLevel,minDistance,mask,blockSize,false,k);
    return corners;
}


/**
 * @brief 返回角点坐标
 * @return const vector<Point2f> 
 */
std::vector<Point2f> arrow_Imgprocess::get_PointOfPnp() const
{
    std::vector<Point2f> PointOfSlot;
    PointOfSlot.clear();
    for(auto arrow:arrows)
    {
        if(arrow.arrow_type == Small)
        {
            PointOfSlot.emplace_back(arrow.corners[0]);
        }

    }
    return PointOfSlot;

}

/**
 * @brief 可视化检测结果
 */
void arrow_Imgprocess::Draw_bound()
{
    arrowDisplay = Src_Img.clone();
    if(!arrows.empty())
    {
        
        string tmp_str;
        cv::Scalar color = CV_COLOR_YELLOW;
        putText(arrowDisplay, "FOUND"+std::to_string(arrows.size())+"ARROW", Point(100, 50), FONT_HERSHEY_SIMPLEX, 1, CV_COLOR_PINK, 2, 8, false);
	    for(auto arrow:arrows)
	    {
            if(arrow.arrow_type == Small)
            {
                tmp_str = "Small_Arrow";
                color = CV_COLOR_YELLOW;
            }
            else
            {
                tmp_str = "Big_Arrow";
                color = CV_COLOR_GREEN;
            }
            circle(arrowDisplay, arrow.corners[0], 2, color, 2);
            for (size_t i = 0; i < 4; i++)
            {
                line(arrowDisplay, arrow.get_vertex()[i], arrow.get_vertex()[(i + 1) % 4], color, 2, 8, 0);
            }
            //display its center point x,y value 显示中点坐标
            putText(arrowDisplay, to_string(int(arrow.corners[0].x)), arrow.corners[0], FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 255), 1, 8, false);
            putText(arrowDisplay, to_string(int(arrow.corners[0].y)), arrow.corners[0] + Point2f(0, 15), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 255), 1, 8, false);
            putText(arrowDisplay, tmp_str+to_string(float(arrow.score)), arrow.rec().boundingRect().tl() + Point2i(0, 30), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255), 1, 8, false);
	    }
    }
    
}

/**
 * @brief 展示过程图像
 */
void arrow_Imgprocess::debug_show()
{
    imshow("gray",Gray_Img);
    imshow("Pre_bin",Pre_Img);
    imshow("thin",Thin_Img);
    imshow("result",arrowDisplay);
    waitKey(0);
}