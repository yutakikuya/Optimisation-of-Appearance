#include "2002_R_Shacked_ALD.h"

//標準偏差と平均
double mean(std::vector<double> vec)
{
	double sum = 0.0;
	for (int i = 0; i < vec.size(); ++i)
	{
		sum += vec[i];
	}
	return sum / vec.size();
}

double dev(std::vector<double> vec)
{
	double var = 0.0;
	double m = mean(vec);
	for (int i = 0; i < vec.size(); ++i)
		var += (vec[i] - m) * (vec[i] - m);

	return sqrt(var / vec.size());
}

float inMaxValue(cv::Mat src_img)
{
	float max=0;
	for(int y = 0; y < src_img.rows; ++y){
		for(int x = 0; x < src_img.cols; ++x){
			if(max < src_img.at<float>(y, x)) max = src_img.at<float>(y, x); 
		}
	}
	return max;
}

//評価式で途中に使う関数群
cv::Mat create_gradientImg(cv::Mat src_img)
{
	cv::Mat sobel_x;
	cv::Mat sobel_y;
	cv::Mat normal_src_img;
	 // 0 - 255 を 0.0 - 1.0 に変換
	src_img.convertTo(normal_src_img, CV_32F, 1.0/255);

	cv::Sobel(normal_src_img, sobel_x, CV_32F, 1, 0);
	cv::Sobel(normal_src_img, sobel_y, CV_32F, 0, 1);

	cv::Mat gradientImg;
	cv::magnitude(sobel_x, sobel_y, gradientImg);

	return gradientImg;
}

cv::Mat create_histImg(cv::Mat src_img, cv::Mat PIM)
{
	const int ch_width = 260, ch_height=200;
	cv::Mat histImg(cv::Size(ch_width, ch_height), CV_8UC3, cv::Scalar::all(255));

	cv::Mat hist;
	const int hdims[] = {256}; // 次元毎のヒストグラムサイズ
	const float hranges[] = {0,256};
	const float* ranges[] = {hranges}; // 次元毎のビンの下限上限
	double max_val = .0;
	cv::calcHist(&src_img, 1, 0, PIM, hist, 1, hdims, ranges);

	cv::minMaxLoc(hist, 0, &max_val);
	// ヒストグラムのスケーリングと描画
	cv::Scalar color = cv::Scalar::all(100);
	// スケーリング
	hist = hist * (max_val? ch_height/max_val:0.);
	std::vector<int> a;

	for(int j=0; j<hdims[0]; ++j) {
		int bin_w = cv::saturate_cast<int>((double)ch_width/hdims[0]);
		a.push_back(ch_height);
		cv::rectangle(histImg, 
			cv::Point(j*bin_w, histImg.rows),
			cv::Point((j+1)*bin_w, histImg.rows-cv::saturate_cast<int>(hist.at<float>(j))),
			color, -1);
	}

	return histImg;
}

std::vector<double> getSurfacePixels(cv::Mat raw_img, cv::Mat PIM)
{
	std::vector<double> surface_pixels;
	for(int y = 0; y < PIM.rows; ++y)
	{
		for(int x = 0; x < PIM.cols; ++x)
		{
			if(PIM.at<uchar>(y, x) == SURFACE)
			{
				surface_pixels.push_back((double)raw_img.at<uchar>(y, x));
			}
		}
	}
	return surface_pixels;
}

std::vector<double> getEdgePixels(cv::Mat raw_img, cv::Mat PIM)
{
	std::vector<double> surface_pixels;
	for(int y = 0; y < PIM.rows; ++y)
	{
		for(int x = 0; x < PIM.cols; ++x)
		{
			if(PIM.at<uchar>(y, x) == EDGE)
			{
				surface_pixels.push_back((double)raw_img.at<uchar>(y, x));
			}
		}
	}
	return surface_pixels;
}

std::vector<double> getBackgroundPixels(cv::Mat raw_img, cv::Mat PIM)
{
	std::vector<double> surface_pixels;
	for(int y = 0; y < PIM.rows; ++y)
	{
		for(int x = 0; x < PIM.cols; ++x)
		{
			if(PIM.at<uchar>(y, x) == BACKGROUND)
			{
				surface_pixels.push_back((double)raw_img.at<uchar>(y, x));
			}
		}
	}
	return surface_pixels;
}

//評価項 と 評価式
double f_grad(cv::Mat raw_img, cv::Mat PIM)
{
	double result_f_grad;
	int Surface_pixels = 0;
	double Sum_gradient = 0;
	double g_t = 0;
	double g_I = 0;

	cv::Mat gradImg = create_gradientImg(raw_img);

	for(int y = 0; y < gradImg.rows; ++y)
	{
		for(int x = 0; x < gradImg.cols; ++x)
		{
			if(PIM.at<uchar>(y, x) == SURFACE)
			{
				Surface_pixels++;
				Sum_gradient += pow(fabs(gradImg.at<float>(y, x)),2); 
				if(g_t < gradImg.at<float>(y, x)) g_t = gradImg.at<float>(y, x); 
			}
		}
	}

	g_I = sqrt(Sum_gradient/Surface_pixels);

	if(g_I >= F_GRAD_ALPHA*g_t) result_f_grad = (g_t - g_I) / ((1-F_GRAD_ALPHA)*g_t);
	else                        result_f_grad = 1;
	return result_f_grad;
}

double f_var(cv::Mat raw_img, cv::Mat PIM)
{
	double result_f_var =0;
	std::vector<double> surface_pixels = getSurfacePixels(raw_img, PIM);

	double tmp_result = fabs(dev(surface_pixels) - (double)F_VAR_SIGMA) / (double)F_VAR_SIGMA;
	result_f_var = (tmp_result < 1) ? tmp_result : 1;
	return result_f_var;
}

double f_mean(cv::Mat raw_img, cv::Mat PIM)
{
	double result = 0.0;
	std::vector<double> surface_pixels = getSurfacePixels(raw_img, PIM);

	result = fabs(mean(surface_pixels) - F_MEAN_M) / ((F_MEAN_M > 255 - F_MEAN_M) ? F_MEAN_M : 255 - F_MEAN_M);
	return result;
}

double f_hist(cv::Mat raw_img, cv::Mat PIM)
{
	double result = 0.0;
	std::vector<double> Surf_p = getSurfacePixels(raw_img,PIM);
	double N = (double)Surf_p.size();
	std::sort(Surf_p.begin(),Surf_p.end());
	std::vector<int> hist(256);

	for(int i=0; i<(double)N; ++i) 	hist[Surf_p[i]] += 1;

	//ideal histgram element
	double n_t = N/255.0;
	
	//target_hist
	double sum = 0.0;
	for(int i=0; i<256; ++i) sum += pow(hist[i]- n_t, 2);
	double d = sqrt(sum/255);

	//worst case histgram
	double worst_d = sqrt((pow(N - n_t,2) + pow(n_t,2) * 255) / 255);
	
	double a1 = F_HIST_ALPHA_1*worst_d;
	double a2 = F_HIST_ALPHA_2*worst_d;
	if      ( d > a1 )               result = 1.0;
	else if ( (a2 < d) && (d < a1) ) result = (d - a2) / (a1 - a2);
	else if ( d < a2 )               result = 0.0;

	return result;
}

double f_dir(cv::Mat raw_img, cv::Mat PIM)
{
	double result = 0.0;
	double N = 256; //tekitooooo
	std::vector<double> thita(N);
	double sum = 0.0;
	for(int i = 1; i <= N; ++i) sum += pow(thita[i] - F_DIR_THITA_T,2);
	double d = sqrt(sum / N);
	//if(d < F_DIR_ALPHA) reuslt = d / F_DIR_ALPHA;
	//else                result = 1;
	return result;
}

double f_edge(cv::Mat raw_img, cv::Mat PIM)
{
	double result = 0.0;
	// laplacian image
	cv::Mat laplacian_img, tmp_img;
	cv::Mat normal_src_img;
	 // 0 - 255 を 0.0 - 1.0 に変換
	raw_img.convertTo(normal_src_img, CV_32F, 1.0/255);
	cv::Laplacian(normal_src_img, tmp_img, CV_32F);
	cv::convertScaleAbs(tmp_img, laplacian_img);

	//gradient image
	cv::Mat grad_img = create_gradientImg(raw_img);
	std::vector<double> Edge_p = getEdgePixels(raw_img, PIM);
	double sum = 0.0;

	for(int y = 0; y < laplacian_img.rows; ++y)
	{
		for(int x = 0; x < laplacian_img.cols; ++x)
		{
			bool z_filter;
			if(PIM.at<uchar>(y,x) == EDGE)
			{
				if(laplacian_img.at<float>(y,x) > 0)
				{
					if(laplacian_img.at<float>(y-1,x) == 0 && laplacian_img.at<float>(y,x-1) == 0 && 
						laplacian_img.at<float>(y,x+1) == 0 && laplacian_img.at<float>(y+1,x) == 0)
						z_filter = false;
					else
						z_filter = true;
				}
				if(laplacian_img.at<float>(y,x) == 0)
				{
					if(laplacian_img.at<float>(y-1,x) > 0 && laplacian_img.at<float>(y,x-1) > 0 && 
						laplacian_img.at<float>(y,x+1) > 0 && laplacian_img.at<float>(y+1,x) > 0)
						z_filter = false;
					else
						z_filter = true;
				}
				if      (F_EDGE_T_MAX < fabs(grad_img.at<float>(y,x)) && z_filter == true)
				{
					sum += 1;
				}
				else if (F_EDGE_T_MIN < fabs(grad_img.at<float>(y,x)) 
					      && fabs(grad_img.at<float>(y,x)) < F_EDGE_T_MAX && z_filter == 1)
			    {
					sum += (fabs(grad_img.at<float>(y,x)) - F_EDGE_T_MIN) / (F_EDGE_T_MAX - F_EDGE_T_MIN);
				}
				else if (abs(grad_img.at<float>(y,x)) < F_EDGE_T_MIN || z_filter == 0)
				{
					sum += 0;
				}
			}
		}
	}

	result = (Edge_p.size() - sum) / Edge_p.size();
	return result;
}

double f_Q(double f_grad,double f_edge,double f_var,double f_mean,double f_hist,double f_dir)
{
	cv::FileStorage fs("config.yml", cv::FileStorage::READ);
	double f_e_c = static_cast<double>(fs["f_edge_scale"]);
	double f_g_c = static_cast<double>(fs["f_grad_scale"]);
	double f_v_c = static_cast<double>(fs["f_var_scale"]);
	double f_m_c = static_cast<double>(fs["f_mean_scale"]);
	double f_h_c = static_cast<double>(fs["f_hist_scale"]);
	double f_d_c = static_cast<double>(fs["f_dir_scale"]);
	return f_g_c*f_grad+f_e_c*f_edge+f_v_c*f_var+f_m_c*f_mean+f_h_c*f_hist+f_d_c*f_dir;
}
