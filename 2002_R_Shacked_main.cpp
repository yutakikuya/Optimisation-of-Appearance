#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <math.h>
#include <sys/stat.h> // stat

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream> 

#include "2002_R_Shacked_ALD.h"

#define FOLDER_NAME "C:/Users/悠太/Pictures/monkey"

int main()
{
	cv::FileStorage fs("config.yml", cv::FileStorage::READ);

	std::string folder_name = static_cast<std::string>(fs["folder_name"]);
	double c = static_cast<double>(fs["f_edge_scale"]);
	std::cout << c << " a" << folder_name << std::endl;
	std::stringstream    stream_result;
	stream_result << folder_name << "/result.csv";
	std::string fileName_result = stream_result.str();
	std::ofstream ofs( fileName_result );
	ofs  << "thita" << "," << "fay" << "," << "f_edge" << "," << "f_grad" << "," << "f_var" << "," << "f_mean" << "," << "f_hist" << "," << "f_dir" << "," 	<< "f_Q" << std::endl;
	
	std::vector< std::vector <double> > f_e_all, f_g_all, f_v_all, f_m_all, f_h_all, f_q_all;
	double min[6] = {1000000,1000000,1000000,1000000,1000000,1000000};
	double max[6]= {-1000000,-1000000,-1000000,-1000000,-1000000,-1000000};
	std::cout << "|----------|" << std::endl;
	std::cout << " ";
	for(int i = 0; i <= 9; ++i )
	{
		std::vector<double> f_e_tmp, f_g_tmp, f_v_tmp, f_m_tmp, f_h_tmp, f_q_tmp;
		std::cout << "+";
		for(int j = 0; j < 36; ++j)
		{
			if(j!=0 && i==0) break;
			std::stringstream    stream;
			stream << folder_name << "/"  << "sita" << i*10 << "fay" << j*10 << ".png";
			std::string fileName = stream.str();
			std::stringstream    stream_pim;
			stream_pim << folder_name << "/PIM.bmp";
			std::string fileName_pim = stream_pim.str();
			cv::Mat raw_img = cv::imread(fileName,0);
			cv::Mat PIM = cv::imread(fileName_pim,0);

			double f_e = f_edge(raw_img, PIM);
			double f_g = f_grad(raw_img, PIM);
			double f_v = f_var(raw_img, PIM);
			double f_m = f_mean(raw_img, PIM);
			double f_h = f_hist(raw_img, PIM);
			double f_d = f_dir(raw_img, PIM);
			double f_q =  f_Q( f_g, f_e, f_v, f_m, f_h,f_d);

			f_e_tmp.push_back(f_e);
			f_g_tmp.push_back(f_g);
			f_v_tmp.push_back(f_v);
			f_m_tmp.push_back(f_m);
			f_h_tmp.push_back(f_h); 
			f_q_tmp.push_back(f_q);

			if(min[0] > f_e) min[0] = f_e;
			if(min[1] > f_g) min[1] = f_g;
			if(min[2] > f_v) min[2] = f_v;
			if(min[3] > f_m) min[3] = f_m;
			if(min[4] > f_h) min[4] = f_h;
			if(min[5] > f_q) min[5] = f_q;

			if(max[0] < f_e) max[0] = f_e;
			if(max[1] < f_g) max[1] = f_g;
			if(max[2] < f_v) max[2] = f_v;
			if(max[3] < f_m) max[3] = f_m;
			if(max[4] < f_h) max[4] = f_h;
			if(max[5] < f_q) max[5] = f_q;

			

			ofs  << i*10 << ","	<< j*10 << "," << f_e << ","<< f_g << "," << f_v << ","<< 
				f_m << ","	<< f_h << "," << f_d << ","<< f_q << std::endl;
		}
		f_e_all.push_back(f_e_tmp);
		f_g_all.push_back(f_g_tmp);
		f_v_all.push_back(f_v_tmp);
		f_m_all.push_back(f_m_tmp);
		f_h_all.push_back(f_h_tmp); 
		f_q_all.push_back(f_q_tmp);
	}
	std::cout << std::endl;

	std::vector<cv::Mat> results;
	for(int k = 0; k < 6; ++k)
	{
		results.push_back(cv::Mat::zeros(600, 600, CV_8UC3));
		int angle = 270;
		cv::ellipse(results[k], cv::Point(300, 300), cv::Size(250, 250), angle, angle, angle+360, cv::Scalar(255,255,255), 3, CV_AA);
		for( int i = 9; i >= 0; --i )
		{
			for( int j = 35; j >= 0; --j )
			{
				double color = 0.0;
				/*if      (k == 0) color = f_e_all[i][j] == 0 ? 0 : 255 - (255 * (max[k] - f_e_all[i][j]) / (max[k] - min[k]));
				else if (k == 1) color = f_g_all[i][j] == 0 ? 0 : 255 - (255 * (max[k] - f_g_all[i][j]) / (max[k] - min[k]));
				else if (k == 2) color = f_v_all[i][j] == 0 ? 0 : 255 - (255 * (max[k] - f_v_all[i][j]) / (max[k] - min[k]));
				else if (k == 3) color = f_m_all[i][j] == 0 ? 0 : 255 - (255 * (max[k] - f_m_all[i][j]) / (max[k] - min[k]));
				else if (k == 4) color = f_h_all[i][j] == 0 ? 0 : 255 - (255 * (max[k] - f_h_all[i][j]) / (max[k] - min[k]));
				else if (k == 5) color = f_q_all[i][j] == 0 ? 0 : 255 - (255 * (max[k] - f_q_all[i][j]) / (max[k] - min[k]));
*/
				if      (k == 0) color = f_e_all[i][j] == 0 ? 0 : 255 - (255 * (f_e_all[i][j] - min[k]) / (max[k] - min[k]));
				else if (k == 1) color = f_g_all[i][j] == 0 ? 0 : 255 - (255 * (f_g_all[i][j] - min[k]) / (max[k] - min[k]));
				else if (k == 2) color = f_v_all[i][j] == 0 ? 0 : 255 - (255 * (f_v_all[i][j] - min[k]) / (max[k] - min[k]));
				else if (k == 3) color = f_m_all[i][j] == 0 ? 0 : 255 - (255 * (f_m_all[i][j] - min[k]) / (max[k] - min[k]));
				else if (k == 4) color = f_m_all[i][j] == 0 ? 0 : 255 - (255 * (f_h_all[i][j] - min[k]) / (max[k] - min[k]));
				else if (k == 5) color = f_q_all[i][j] == 0 ? 0 : 255 - (255 * (f_q_all[i][j] - min[k]) / (max[k] - min[k]));

				/*if      (k == 0) color = f_e_all[i][j] == 0 ? 0 : (255 * (1 - f_e_all[i][j]) / (1 - 0));
				else if (k == 1) color = f_g_all[i][j] == 0 ? 0 : (255 * (1 - f_g_all[i][j]) / (1 - 0));
				else if (k == 2) color = f_v_all[i][j] == 0 ? 0 : (255 * (1 - f_v_all[i][j]) / (1 - 0));
				else if (k == 3) color = f_m_all[i][j] == 0 ? 0 : (255 * (1 - f_m_all[i][j]) / (1 - 0));
				else if (k == 4) color = f_m_all[i][j] == 0 ? 0 : (255 * (1 - f_h_all[i][j]) / (1 - 0));
				else if (k == 5) color = f_q_all[i][j] == 0 ? 0 : (255 * (max[k] - f_q_all[i][j]) / (max[k] - min[k]));

*/

				if(i==0 && j == 0){
					cv::ellipse(results[k], cv::Point(300, 300), cv::Size(25, 25), angle, angle-90, angle-90+360, cv::Scalar(color,color,color), -1, CV_AA);
				}
				else if(i==0 && j != 0) ;
				else		cv::ellipse(results[k], cv::Point(300, 300), cv::Size(250*(i+1)/10.0, 250*(i+1)/10.0), angle, angle-90-10*(j), angle-90-10*(j+1), cv::Scalar(color,color,color), -1, CV_AA);
			}
		}

		if(k==0) cv::imwrite("f_edge.jpg",results[k]);
		else if(k==1) cv::imwrite("f_grad.jpg",results[k]);
		else if(k==2) cv::imwrite("f_var.jpg",results[k]);
		else if(k==3) cv::imwrite("f_mean.jpg",results[k]);
		else if(k==4) cv::imwrite("f_hist.jpg",results[k]);
		else if(k==5) cv::imwrite("f_Q.jpg",results[k]);
		cv::waitKey( 0 );
	}

	return 0;
}