#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <math.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>

#define SURFACE 127
#define BACKGROUND 0
#define EDGE 255

//Parameter
#define F_GRAD_ALPHA 0.1 // 0.3 recommend
#define F_VAR_SIGMA  45 //40 - 45 recommend
#define F_MEAN_M 150 // 150 recommend
#define F_HIST_ALPHA_1 0.8 // 0.1 recommend
#define F_HIST_ALPHA_2 0.1 // 0.8 recommend
#define F_DIR_ALPHA 65
#define F_DIR_THITA_T 45
#define F_EDGE_T_MAX 0.7
#define F_EDGE_T_MIN 0.2

//ïWèÄïŒç∑åvéZópä÷êî
double mean(std::vector<double> vec);
double dev(std::vector<double> vec);
float inMaxValue(cv::Mat src_img);

cv::Mat create_gradientImg(cv::Mat src_img);
cv::Mat create_histImg(cv::Mat src_img, cv::Mat PIM);
std::vector<double> getSurfacePixels(cv::Mat raw_img, cv::Mat PIM);
std::vector<double> getEdgePixels(cv::Mat raw_img, cv::Mat PIM);
std::vector<double> getBackgroundPixels(cv::Mat raw_img, cv::Mat PIM);

double f_grad(cv::Mat input_img, cv::Mat PIM);
double f_edge(cv::Mat input_img, cv::Mat PIM);
double f_var(cv::Mat input_img, cv::Mat PIM);
double f_mean(cv::Mat input_img, cv::Mat PIM);
double f_hist(cv::Mat input_img, cv::Mat PIM);
double f_dir(cv::Mat input_img, cv::Mat PIM);

double f_Q(double f_grad,double f_edge,double f_var,double f_mean,double f_hist,double f_dir);