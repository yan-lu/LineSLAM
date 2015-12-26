
/////////////////////////////////////////////////////////////////////////////////
//
// LineSLAM, version 1.0
// Copyright (C) 2013-2015 Yan Lu, Dezhen Song
// Netbot Laboratory, Texas A&M University, USA
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//
/////////////////////////////////////////////////////////////////////////////////

#include "lineslam.h"
#include "utils.h"


#include "../node.h"

#define EXTRACTLINE_USE_MAHDIST


extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

void Map3d::slam() {}

Frame::Frame (string rgbName, string depName, cv::Mat K, cv::Mat dc) {}


void Node::detectFrameLines(const cv::Mat& gray_uchar, double lineLenThresh, int method)
	// detect line segments from rgb image
	// input: gray image
	// ouput: lines
{
	if(method == 0) {

		IplImage pImg = gray_uchar;
		ntuple_list  lsdOut;	
	
		lsdOut = callLsd(&pImg);// use LSD method
	
		int dim = lsdOut->dim;
		double a,b,c,d;
		lines.reserve(lsdOut->size);
		for(int i=0; i<lsdOut->size; i++) {// store LSD output to lineSegments 
			a = lsdOut->values[i*dim];
			b = lsdOut->values[i*dim+1];
			c = lsdOut->values[i*dim+2];
			d = lsdOut->values[i*dim+3];
			if ( sqrt((a-c)*(a-c)+(b-d)*(b-d)) > lineLenThresh) {
				lines.push_back(FrameLine(cv::Point2d(a,b), cv::Point2d(c,d)));
			}
		}		
	}

	if(method == 1) {
		int n;
		LS* ls = callEDLines(gray_uchar, &n);
		lines.reserve(n);
		for(int i=0; i<n; i++) {// store output to lineSegments 
			if ((ls[i].sx-ls[i].ex)*(ls[i].sx-ls[i].ex) +(ls[i].sy-ls[i].ey)*(ls[i].sy-ls[i].ey) 
				> lineLenThresh*lineLenThresh) {
				lines.push_back(FrameLine(cv::Point2d(ls[i].sx,ls[i].sy), cv::Point2d(ls[i].ex,ls[i].ey)));
			}
		}
	}


	// assign id and equation
	for (int i=0; i<lines.size(); ++i){
		lines[i].lid = i;
		lines[i].complineEq2d();
	}
  
	// compute msld line descriptors
	cv::Mat xGradImg, yGradImg;
	int ddepth = CV_64F;	
	cv::Sobel(gray_uchar, xGradImg, ddepth, 1, 0, 3); // Gradient X
	cv::Sobel(gray_uchar, yGradImg, ddepth, 0, 1, 3); // Gradient Y
	double	*xG, *yG;
	if(xGradImg.isContinuous() && yGradImg.isContinuous()) {
		xG = (double*) xGradImg.data;
		yG = (double*) yGradImg.data;
	} else { 
		xG = new double[xGradImg.rows*xGradImg.cols];
 		yG = new double[yGradImg.rows*yGradImg.cols];
 		int idx = 0;
    	for(int i=0; i<xGradImg.rows; ++i) {
    		for(int j=0; j<xGradImg.cols; ++j) {
    			xG[idx] = xGradImg.at<double>(i,j);
    			yG[idx] = yGradImg.at<double>(i,j);
    			++idx;
    		}
    	}
	}

	for (int i=0; i<lines.size(); ++i)  {
	//	computeMSLD(lines[i], &xGradImg, &yGradImg);
		lines[i].r = lines[i].getGradient(&xGradImg, &yGradImg);	
		computeMSLD(lines[i], xG, yG, gray_uchar.cols, gray_uchar.rows);
	}
	if(!xGradImg.isContinuous() || !yGradImg.isContinuous()) { // dynamic allocation
		delete[] xG; 
		delete[] yG;
	}
}


void Node::extractLineDepth(const cv::Mat& depth_float, const cv::Mat& K, 
	double ratio_of_collinear_pts, double line_3d_len_thres_m, double depth_scaling)
	// extract the 3d info of an frame line if availabe from the depth image
	// input: depth, lines
	// output: lines with 3d info
{		
	int depth_CVMatDepth = depth_float.depth();
	for(int i=0; i<lines.size(); ++i)	{ // each line
		double len = cv::norm(lines[i].p - lines[i].q);		
		vector<cv::Point3d> pts3d;
		// iterate through a line
   		double numSmp = min(max(len/sysPara.line_sample_interval, (double)sysPara.line_sample_min_num), (double)sysPara.line_sample_max_num);  // smaller numSmp for fast speed, larger numSmp should be more accurate
		pts3d.reserve(numSmp);
		for(int j=0; j<=numSmp; ++j) {
			// use nearest neighbor to querry depth value
			// assuming position (0,0) is the top-left corner of image, then the
			// top-left pixel's center would be (0.5,0.5)
			cv::Point2d pt = lines[i].p * (1-j/numSmp) + lines[i].q * (j/numSmp);
			if(pt.x<0 || pt.y<0 || pt.x >= depth_float.cols || pt.y >= depth_float.rows ) continue;
			int row, col; // nearest pixel for pt
			if((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) {// boundary issue
				col = max(int(pt.x-1),0);
				row = max(int(pt.y-1),0);
			} else {
				col = int(pt.x);
				row = int(pt.y);
			}
			double zval = -1;
			double depval = depth_float.at<float>(row,col);
			if(depth_CVMatDepth == CV_32F) 
				depval = depth_float.at<float>(row,col);
			else if (depth_CVMatDepth == CV_64F) 
				depval = depth_float.at<double>(row,col);
			else {
				cerr<<"Node::extractLineDepth: depth image matrix type is not float/double\n";
				exit(0);	
			}	
			if(depval < EPS) { // no depth info

			} else {
				zval = depval/depth_scaling; // in meter, z-value
			}

			// export 3d points to file
			if (zval > 0 ) {
				cv::Point2d xy3d = mat2cvpt(K.inv()*cvpt2mat(pt))*zval;	
				pts3d.push_back(cv::Point3d(xy3d.x, xy3d.y, zval));
			}
		}
		//if (pts3d.size() < max(10.0,len*ratio_of_collinear_pts))
		if (pts3d.size() < max(5.0, numSmp *ratio_of_collinear_pts))
			continue;

		RandomLine3d tmpLine;		

#ifdef EXTRACTLINE_USE_MAHDIST
		vector<RandomPoint3d> rndpts3d;
		rndpts3d.reserve(pts3d.size());
		// compute uncertainty of 3d points
		for(int j=0; j<pts3d.size();++j) {
			rndpts3d.push_back(compPt3dCov(pts3d[j], K, asynch_time_diff_sec_));
		}
		// using ransac to extract a 3d line from 3d pts
		tmpLine = extract3dline_mahdist(rndpts3d);
		
#else
		tmpLine = extract3dline(pts3d);
#endif
		if(tmpLine.pts.size()/numSmp > ratio_of_collinear_pts	&&
			cv::norm(tmpLine.A - tmpLine.B) > line_3d_len_thres_m) {
				MLEstimateLine3d(tmpLine, sysPara.line3d_mle_iter_num);//expensive, postpone to after line matching
				lines[i].haveDepth = true;
				lines[i].line3d = tmpLine;
				
		}		
		lines[i].line3d.pts.clear();
	}	

}

void Node::detect3DLines(const cv::Mat& gray_uchar, const cv::Mat& depth_float, double line2d_len_thres, 
                      const cv::Mat& K,double ratio_of_collinear_pts, double line_3d_len_thres_m, double depth_scaling, string algorithm)
{
//	MyTimer tm; tm.start();	
	
	vector<FrameLine> allLines; // 2d and 3d lines
	if(algorithm == "LSD") { // using LSD, 100+ ms
		IplImage pImg = gray_uchar;
		ntuple_list  lsdOut;	
		lsdOut = callLsd(&pImg);// use LSD method
		int dim = lsdOut->dim;
		double a,b,c,d;
		allLines.reserve(lsdOut->size);
		for(int i=0; i<lsdOut->size; i++) {// store LSD output to lineSegments 
			a = lsdOut->values[i*dim];
			b = lsdOut->values[i*dim+1];
			c = lsdOut->values[i*dim+2];
			d = lsdOut->values[i*dim+3];
			if ( sqrt((a-c)*(a-c)+(b-d)*(b-d)) > line2d_len_thres) {
				allLines.push_back(FrameLine(cv::Point2d(a,b), cv::Point2d(c,d)));
			}
		}	
		
	}
    
	if(algorithm == "EDLINES") { // using EDlines, 15 ms
		int n;
		LS* ls = callEDLines(gray_uchar, &n);
		allLines.reserve(n);
		for(int i=0; i<n; i++) {// store output to lineSegments 
			if ((ls[i].sx-ls[i].ex)*(ls[i].sx-ls[i].ex) +(ls[i].sy-ls[i].ey)*(ls[i].sy-ls[i].ey) 
				> line2d_len_thres*line2d_len_thres) {
				allLines.push_back(FrameLine(cv::Point2d(ls[i].sx,ls[i].sy), cv::Point2d(ls[i].ex,ls[i].ey)));
			}
		}
	}

	Eigen::Matrix3d eK;
	for(int i=0; i<3; ++i)
		for(int j=0; j<3; ++j)
			eK(i,j) = K.at<double>(i,j);
	Eigen::Matrix3d Kinv = eK.inverse();	
 

 	
 	int depth_CVMatDepth = depth_float.depth();
  	#pragma omp parallel for
	for(int i=0; i<allLines.size(); ++i)	{ // 20 -30 ms
		double len = cv::norm(allLines[i].p - allLines[i].q);		
		// iterate through a line
		double numSmp = min(max(len/sysPara.line_sample_interval, (double)sysPara.line_sample_min_num), (double)sysPara.line_sample_max_num);  // smaller numSmp for fast speed, larger numSmp should be more accurate	
   		vector<cv::Point3d> pts3d; pts3d.reserve(numSmp);
		for(int j=0; j<=numSmp; ++j) {
			// use nearest neighbor to querry depth value
			// assuming position (0,0) is the top-left corner of image, then the
			// top-left pixel's center would be (0.5,0.5)
			cv::Point2d pt = allLines[i].p * (1-j/numSmp) + allLines[i].q * (j/numSmp);
			if(pt.x<0 || pt.y<0 || pt.x >= depth_float.cols || pt.y >= depth_float.rows ) continue;
			int row, col; // nearest pixel for pt
			if((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) {// boundary issue
				col = max(int(pt.x-1),0);
				row = max(int(pt.y-1),0);
			} else {
				col = int(pt.x);
				row = int(pt.y);
			}
			double zval = -1;
			double depval;
			if(depth_CVMatDepth == CV_32F) 
				depval = depth_float.at<float>(row,col);
			else if (depth_CVMatDepth == CV_64F) 
				depval = depth_float.at<double>(row,col);
			else {
				cerr<<"Node::extractLineDepth: depth image matrix type is not float/double\n";
				exit(0);	
			}	
			if(depval < EPS || isnan((float)depval)) { // no depth info

			} else {
				zval = depval/depth_scaling; // in meter, z-value
			}

			if (zval > 0 ) {
				Eigen::Vector3d ept(pt.x, pt.y, 1);
				Eigen::Vector3d xy3d = Kinv * ept;
				xy3d = xy3d/xy3d(2);
				pts3d.push_back(cv::Point3d(xy3d(0)*zval, xy3d(1)*zval, zval));								
			}
		}
		if (pts3d.size() < max(10.0, numSmp *ratio_of_collinear_pts))
			continue;

		RandomLine3d tmpLine;		
		vector<RandomPoint3d> rndpts3d;
		rndpts3d.reserve(pts3d.size());
		// compute uncertainty of 3d points
		for(int j=0; j<pts3d.size();++j) {
			rndpts3d.push_back(compPt3dCov(pts3d[j], K, asynch_time_diff_sec_));
		}
		// using ransac to extract a 3d line from 3d pts
		tmpLine = extract3dline_mahdist(rndpts3d);
		
		if(tmpLine.pts.size()/numSmp > ratio_of_collinear_pts	&&
			cv::norm(tmpLine.A - tmpLine.B) > line_3d_len_thres_m) {
				allLines[i].haveDepth = true;
				allLines[i].line3d = tmpLine;
				
		}		
	}	

//// prepare for compute msld line descriptors
	cv::Mat xGradImg, yGradImg;
	int ddepth = CV_64F;	
	cv::Sobel(gray_uchar, xGradImg, ddepth, 1, 0, 5); // Gradient X
	cv::Sobel(gray_uchar, yGradImg, ddepth, 0, 1, 5); // Gradient Y
	double	*xG, *yG;
	if(xGradImg.isContinuous() && yGradImg.isContinuous()) {
		xG = (double*) xGradImg.data;
		yG = (double*) yGradImg.data;
	} else { 
		xG = new double[xGradImg.rows*xGradImg.cols];
 		yG = new double[yGradImg.rows*yGradImg.cols];
 		int idx = 0;
    	for(int i=0; i<xGradImg.rows; ++i) {
    		for(int j=0; j<xGradImg.cols; ++j) {
    			xG[idx] = xGradImg.at<double>(i,j);
    			yG[idx] = yGradImg.at<double>(i,j);
    			++idx;
    		}
    	}
	}

	lines.reserve(allLines.size());
	// NOTE this for-loop mustnot be parallized
	for(int i=0; i<allLines.size(); ++i) {
		if(allLines[i].haveDepth) { 
			lines.push_back(allLines[i]);
			lines.back().lid = lines.size()-1;
			lines.back().complineEq2d();				
			lines.back().r = lines.back().getGradient(&xGradImg, &yGradImg);
		}
	}
	

	#pragma omp parallel for
	for(int i=0; i<lines.size(); ++i) {		// 60 ms, 0.6ms/line
		computeMSLD(lines[i], xG, yG, gray_uchar.cols, gray_uchar.rows); // 0.1 ms/line
		MLEstimateLine3d(lines[i].line3d, sysPara.line3d_mle_iter_num);
		vector<RandomPoint3d> ().swap(lines[i].line3d.pts); // swap with a empty vector effectively free up the memory, clear() doesn't.
	}
//	tm.end(); cout<<"detect 3d lines "<<tm.time_ms<<" ms\n";

	if(!xGradImg.isContinuous() || !yGradImg.isContinuous()) { // dynamic allocation
		delete[] xG; 
		delete[] yG;
	}

}


Frame::Frame (cv::Mat rgbImg_d, cv::Mat depImg_f, cv::Mat K)
{
	
	rgb = rgbImg_d.clone();

	if(rgb.channels() ==3)
		cv::cvtColor(rgb, gray, CV_RGB2GRAY); //  
	else
		gray = rgb;
//	cv::equalizeHist(gray, gray);
	oriDepth = depImg_f; // scale 1 

	oriDepth.convertTo(depth, CV_64F);

	lineLenThresh = sysPara.line_segment_len_thresh;

	detectFrameLines();	
	extractLineDepth();

	rgb.release();
	gray.release();
	depth.release();
	oriDepth.release();

	int n3=0;
	for(int i=0; i<lines.size(); ++i) {		
		if(lines[i].haveDepth) {
			n3++;
		}
	}

	cout<<" LSD "<<lines.size()<<", 3D "<<n3<<endl;

}

void Frame::detectFrameLines()
	// detect line segments from rgb image
	// input: gray image
	// ouput: lines
{
///	static int cn=0; static double at1=0, at2=0; MyTimer t1,t2; 

	IplImage pImg = gray;
	ntuple_list  lsdOut;	

	lsdOut = callLsd(&pImg);// use LSD method
 
	int dim = lsdOut->dim;
	double a,b,c,d;
	double minv=10;
	lines.reserve(lsdOut->size);
	for(int i=0; i<lsdOut->size; i++) {// store LSD output to lineSegments 
		a = lsdOut->values[i*dim];
		b = lsdOut->values[i*dim+1];
		c = lsdOut->values[i*dim+2];
		d = lsdOut->values[i*dim+3];
		if ( sqrt((a-c)*(a-c)+(b-d)*(b-d)) > lineLenThresh) {
			lines.push_back(FrameLine(cv::Point2d(a,b), cv::Point2d(c,d)));
		}
	}
	// assign id and equation
	for (int i=0; i<lines.size(); ++i){
		lines[i].lid = i;
		lines[i].complineEq2d();
	}
	// compute msld line descriptors
	cv::Mat xGradImg, yGradImg;
	int ddepth = CV_64F;	
	cv::Sobel(gray, xGradImg, ddepth, 1, 0, 5); // Gradient X
	cv::Sobel(gray, yGradImg, ddepth, 0, 1, 5); // Gradient Y
	#pragma omp  parallel for
	for (int i=0; i<lines.size(); ++i)  {
		computeMSLD(lines[i], &xGradImg, &yGradImg);
	}	
}

void Frame::extractLineDepth()
	// extract the 3d info of an frame line if availabe from the depth image
	// input: depth, lines
	// output: lines with 3d info
{	double depth_scaling = 1;
	int n_3dln = 0;
    #pragma omp  parallel for
	for(int i=0; i<lines.size(); ++i)	{ // each line
		double len = cv::norm(lines[i].p - lines[i].q);		
		vector<cv::Point3d> pts3d;
		// iterate through a line
		double numSmp = (double) min((int)len, 100); // number of line points sampled
		pts3d.reserve(numSmp);
		for(int j=0; j<=numSmp; ++j) {
			// use nearest neighbor to querry depth value
			// assuming position (0,0) is the top-left corner of image, then the
			// top-left pixel's center would be (0.5,0.5)
			cv::Point2d pt = lines[i].p * (1-j/numSmp) + lines[i].q * (j/numSmp);
			if(pt.x<0 || pt.y<0 || pt.x >= depth.cols || pt.y >= depth.rows ) continue;
			int row, col; // nearest pixel for pt
			if((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) {// boundary issue
				col = max(int(pt.x-1),0);
				row = max(int(pt.y-1),0);
			} else {
				col = int(pt.x);
				row = int(pt.y);
			}
			double zval = -1;
			if(depth.at<double>(row,col) < EPS) { // no depth info

			} else {
				zval = depth.at<double>(row,col)/depth_scaling; // in meter, z-value
			}

			// export 3d points to file
			if (zval > 0 ) {
				cv::Point2d xy3d = mat2cvpt(K.inv()*cvpt2mat(pt))*zval;	
				pts3d.push_back(cv::Point3d(xy3d.x, xy3d.y, zval));
			}
		}
		if (pts3d.size() < max(10.0,len*sysPara.ratio_of_collinear_pts))
			continue;

		RandomLine3d tmpLine;		

		vector<RandomPoint3d> rndpts3d;
		rndpts3d.reserve(pts3d.size());
		// compute uncertainty of 3d points
		for(int j=0; j<pts3d.size();++j) {
			rndpts3d.push_back(compPt3dCov(pts3d[j], K, 0));
		}
		// using ransac to extract a 3d line from 3d pts
		tmpLine = extract3dline_mahdist(rndpts3d);
	

		if(tmpLine.pts.size()/len > sysPara.ratio_of_collinear_pts	&&
			cv::norm(tmpLine.A - tmpLine.B) > sysPara.line3d_length_thresh) {
				MLEstimateLine3d (tmpLine, 100);
				vector<RandomPoint3d>().swap(tmpLine.pts);
				lines[i].haveDepth = true;
				lines[i].line3d = tmpLine;
				n_3dln++;
		}
	}	
 
}


void Frame::write2file(string fname)
{
	fname = fname + num2str(id) + ".txt";
	ofstream file(fname.c_str());
	for(int i=0; i<lines.size(); ++i) {
		if(lines[i].haveDepth) {
			file<<lines[i].line3d.A.x<<'\t'<<lines[i].line3d.A.y<<'\t'<<lines[i].line3d.A.z<<'\t'
				<<lines[i].line3d.B.x<<'\t'<<lines[i].line3d.B.y<<'\t'<<lines[i].line3d.B.z<<'\n';
		}
	}
	{}
	file.close();
}

FrameLine::FrameLine(cv::Point2d p_, cv::Point2d q_)
{
	p = p_;
	q = q_;
	l = cvpt2mat(p).cross(cvpt2mat(q));
	haveDepth = false;
	gid = -1;
}

cv::Point2d FrameLine::getGradient(cv::Mat* xGradient, cv::Mat* yGradient)
{	
	cv::LineIterator iter(*xGradient, p, q, 8);
	double xSum=0, ySum=0;
	for (int i=0; i<iter.count; ++i, ++iter) {
		xSum += xGradient->at<double>(iter.pos());
		ySum += yGradient->at<double>(iter.pos());
	}
	double len = sqrt(xSum*xSum+ySum*ySum);
	return cv::Point2d(xSum/len, ySum/len);
}



void Frame::clear()
{
	if(this->isKeyFrame) {
		rgb.release();
		gray.release();
		depth.release();
		oriDepth.release();
		for(int i=0; i<lines.size(); ++i) {
			if(lines[i].haveDepth && lines[i].line3d.covA.rows != 0) {
				lines[i].line3d.pts.clear();
			}
		/*	lines[i].line3d.covA.release();
			lines[i].line3d.covB.release();
			lines[i].line3d.rndA.cov.release();
			lines[i].line3d.rndB.cov.release();
			lines[i].line3d.rndA.U.release();
			lines[i].line3d.rndB.U.release();
			lines[i].line3d.rndA.W.release();
			lines[i].line3d.rndB.W.release();
		*/}
	} else {
		lines.clear();	
		rgb.release();
		gray.release();
		depth.release();
		oriDepth.release();
	}
//	cout<<" <<<<<< frame "<<this->id << " cleared >>>>>>> \n ";
}

#ifdef SLAM_LBA
void Map3d::loopclose() {}
#endif



void SystemParameters::init()
{
	ParameterServer* ps = ParameterServer::instance();
	// ----- 2d-line -----
	line_segment_len_thresh		= ps->get<double>("line_2d_len_thres");		// pixels, min lenght of image line segment to use 
	
	
	msld_sample_interval		= ps->get<double>("msld_sample_interval");

	// ----- 3d-line measurement ----
	line_sample_max_num			= ps->get<int>("line_sample_max_num");
	line_sample_min_num			= ps->get<int>("line_sample_min_num");
	line_sample_interval		= ps->get<double>("line_sample_interval");
	line3d_mle_iter_num			= ps->get<int>("line3d_mle_iter_num");
	ratio_of_collinear_pts		= ps->get<double>("collin_pts_ratio");		// ratio, decide if a frameline has enough collinear pts
	pt2line_dist_extractline	= ps->get<double>("3d_pt2line_dst_extline_m");		// meter, threshold pt to line distance when detect lines from pts
	pt2line_mahdist_extractline	= ps->get<double>("3d_pt2line_mahdst_extline");		// NA,	  as above
	ransac_iters_extract_line	= ps->get<int>("ransac_iters_extline");		// 1, max ransac iters when detect lines from pts
	num_cells_lineseg_range		= ps->get<int>("n_cell_lineseg_range");		// 1, 
	ratio_support_pts_on_line	= ps->get<double>("pts_ratio_verify_3dline");		// ratio, when verifying a detected 3d line
	line3d_length_thresh		= ps->get<double>("line_3d_len_thres_m");		// in meter, ignore too short 3d line segments

	// ----- camera model -----
	stdev_sample_pt_imgline		= ps->get<double>("sample_pt_std_imgline");		// in pixel, std dev of sample point from an image line
	depth_stdev_coeff_c1		= ps->get<double>("depth_std_coeff_c1");	// c1,c2,c3: coefficients of depth noise quadratic function
	depth_stdev_coeff_c2		= ps->get<double>("depth_std_coeff_c2");
	depth_stdev_coeff_c3		= ps->get<double>("depth_std_coeff_c3");

	// ----- key frame -----
	num_raw_frame_skip			= 1;
	window_length_keyframe		= 1;
	num_2dlinematch_keyframe	= 30;		// detect keyframe, minmum number of 2d line matches left
	num_3dlinematch_keyframe	= 40;

	// ----- relative motion -----
	pt2line3d_dist_relmotion	= ps->get<double>("3d_pt2line_dst_relmot_m");		// in meter, 
	line3d_angle_relmotion		= ps->get<double>("3d_line_angle_relmot_deg");
	fast_motion					= 1;
	inlier_ratio_constvel		= 0.4;
	dark_ligthing				= false;
	max_img_brightness			= 0;
	ransac_iters_line_motion	= ps->get<int>("ransac_iters_line_motion");
	adjacent_linematch_window 	= ps->get<int>("adjacent_linematch_window");
	line_match_number_weight    = ps->get<int>("line_match_number_weight");
	min_feature_matches 		= ps->get<int>("min_matches");
	max_mah_dist_for_inliers 	= ps->get<double>("max_mah_dist_for_inliers");
	g2o_line_error_weight 		= ps->get<double>("g2o_line_error_weight");
	min_matches_loopclose 		= ps->get<int>("min_matches_loopclose");
	
	// ----- lba -----
	num_pos_lba	= 5;
	num_frm_lba	= 7;
	g2o_BA_use_kernel			= true;
	g2o_BA_kernel_delta			= 10;

	// ----- loop closing -----
	loopclose_interval			= 50;  // frames, check loop closure
	loopclose_min_3dmatch		= 30;  // min_num for 3d line matches between two frames

	// ----- lsd setting -----
	lsd_angle_th 				= ps->get<double>("lsd_angle_thres");						// default: 22.5 deg
	lsd_density_th 				= ps->get<double>("lsd_density_thres");					// default: 0.7

}



