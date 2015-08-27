#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <math.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream> 

#include "2002_R_Shacked_ALD.h"

#define FOLDER_NAME "C:/Users/—I‘¾/Desktop/monkey"

int main()
{

	std::ofstream ofs( "C:/Users/—I‘¾/Desktop/monkey/result.csv");
	ofs  << "thita" << "," << "fay" << "," << "f_edge" << "," << "f_grad" << "," << "f_var" << "," << "f_mean" << "," << "f_hist" << "," << "f_dir" << "," 	<< "f_Q" << std::endl;
	for(int i = 0; i <= 9; ++i )
	{
		for(int j = 0; j < 36; ++j)
		{
			if(i==0)
				if(j!=0) break;
			std::stringstream    stream;
			stream << FOLDER_NAME << "/"  << "sita" << i*10 << "fay" << j*10 << ".png";
			std::string fileName = stream.str();

			cv::Mat raw_img = cv::imread(fileName,0);
			cv::Mat PIM = cv::imread("C:/Users/—I‘¾/Desktop/monkey/PIM.bmp",0);

			cv::Mat gradImg = create_gradientImg(raw_img);
			cv::Mat histImg = create_histImg(raw_img,PIM);

			ofs  << i*10 << ","	<< j*10 << "," << f_edge(raw_img, PIM) << ","<< f_grad(raw_img, PIM) << "," << f_var(raw_img, PIM) << ","<< 
				f_mean(raw_img, PIM) << ","	<< f_hist(raw_img, PIM) << "," << f_dir(raw_img, PIM) << ","<< f_Q(raw_img, PIM) << std::endl;

/*
			cv::imshow("tmhistp_img.jpg",histImg);
			cv::imshow("tmp_img.jpg",gradImg);
			cv::imwrite("C:/Users/—I‘¾/Desktop/grad.jpg", (gradImg));
			cv::imwrite("C:/Users/—I‘¾/Desktop/histImg.jpg", (histImg));
			cv::waitKey( 0 );*/
		}
	}
	return 0;
}