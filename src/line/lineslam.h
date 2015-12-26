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

#ifndef LINESLAM_H
#define LINESLAM_H
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <fstream>
#ifdef QTPROJECT
#include <QThread>
#endif
#include "lsd/lsd.h"
#include "levmar-2.6/levmar.h"

using namespace std;

#define EPS	(1e-10)
#define PI (3.14159265)
//#define	SLAM_LBA 

class RandomPoint3d 
{
public:
	cv::Point3d pos;	
	cv::Mat		cov;
	cv::Mat		U, W; // cov = U*D*U.t, D = diag(W); W is vector
	double      W_sqrt[3]; // used for mah-dist from pt to ln
	double 		DU[9];
	double      dux[3];

	RandomPoint3d(){}
	RandomPoint3d(cv::Point3d _pos) 
	{
		pos = _pos;
		cov = cv::Mat::eye(3,3,CV_64F);
		U = cv::Mat::eye(3,3,CV_64F);
		W = cv::Mat::ones(3,1,CV_64F);
	}
	RandomPoint3d(cv::Point3d _pos, cv::Mat _cov)
	{
		pos = _pos;
		cov = _cov.clone();
		cv::SVD svd(cov);
		U = svd.u.clone();
		W = svd.w.clone();
		W_sqrt[0] = sqrt(svd.w.at<double>(0));
		W_sqrt[1] = sqrt(svd.w.at<double>(1));
		W_sqrt[2] = sqrt(svd.w.at<double>(2));

		cv::Mat D = (cv::Mat_<double>(3,3)<<1/W_sqrt[0], 0, 0, 
			0, 1/W_sqrt[1], 0,
			0, 0, 1/W_sqrt[2]);
		cv::Mat du = D*U.t();
		DU[0] = du.at<double>(0,0); DU[1] = du.at<double>(0,1); DU[2] = du.at<double>(0,2);
		DU[3] = du.at<double>(1,0); DU[4] = du.at<double>(1,1); DU[5] = du.at<double>(1,2);
		DU[6] = du.at<double>(2,0); DU[7] = du.at<double>(2,1); DU[8] = du.at<double>(2,2);
		dux[0] = DU[0]*pos.x + DU[1]*pos.y + DU[2]*pos.z;
		dux[1] = DU[3]*pos.x + DU[4]*pos.y + DU[5]*pos.z;
		dux[2] = DU[6]*pos.x + DU[7]*pos.y + DU[8]*pos.z;

	}
};

class RandomLine3d 
{
public:
	vector<RandomPoint3d> pts;  //supporting collinear points
	cv::Point3d A, B;
	cv::Mat covA, covB;
	RandomPoint3d rndA, rndB;
	cv::Point3d u, d; // following the representation of Zhang's paper 'determining motion from...'
	RandomLine3d () {}
	RandomLine3d (cv::Point3d _A, cv::Point3d _B, cv::Mat _covA, cv::Mat _covB) 
	{
		A = _A;
		B = _B;
		covA = _covA.clone();
		covB = _covB.clone();
	}
	
};

class LmkLine
{
public:
	cv::Point3d			A, B;
	int					gid;
	vector<vector<int> >	frmId_lnLid;

	LmkLine(){}
};

class FrameLine 
// FrameLine represents a line segment detected from a rgb-d frame.
// It contains 2d image position (endpoints, line equation), and 3d info (if 
// observable from depth image).
{
public:
	cv::Point2d  p, q;				// image endpoints p and q
	cv::Mat		 l;					// 3-vector of image line equation,
	double		 lineEq2d[3];
	
	bool		 haveDepth;			// whether have depth
	RandomLine3d line3d;

	cv::Point2d	 r;					// image line gradient direction (polarity);
	cv::Mat		 des;				// image line descriptor;
//	double*		 desc;				

	int			 lid;				// local id in frame
	int			 gid;				// global id;

	int			 lid_prvKfrm;		// correspondence's lid in previous keyframe

	FrameLine() {gid = -1;}
	FrameLine(cv::Point2d p_, cv::Point2d q_);

	cv::Point2d getGradient(cv::Mat* xGradient, cv::Mat* yGradient);
	void complineEq2d() 
	{
		cv::Mat pt1 = (cv::Mat_<double>(3,1)<<p.x, p.y, 1);
		cv::Mat pt2 = (cv::Mat_<double>(3,1)<<q.x, q.y, 1);
		cv::Mat lnEq = pt1.cross(pt2); // lnEq = pt1 x pt2		
		lnEq = lnEq/sqrt(lnEq.at<double>(0)*lnEq.at<double>(0)
			+lnEq.at<double>(1)*lnEq.at<double>(1)); // normalize, optional
		lineEq2d[0] = lnEq.at<double>(0);
		lineEq2d[1] = lnEq.at<double>(1);
		lineEq2d[2] = lnEq.at<double>(2);

	}
};

class Frame
// Frame represents a rgb-d frame, including its feature info
{
public:
	int					id;
	double				timestamp;
	bool				isKeyFrame;
	vector<FrameLine>	lines;
	cv::Mat				R;
	cv::Mat				t;
	cv::Mat				rgb, gray;
	cv::Mat				depth,oriDepth;	// 
	double				lineLenThresh;


	Frame () {}
	Frame (string rgbName, string depName, cv::Mat K, cv::Mat dc);
	Frame (cv::Mat rgbImg_d, cv::Mat depImg_f, cv::Mat K);

	void detectFrameLines();
	void extractLineDepth();
	void clear();
	void write2file(string);
};

class PoseConstraint
{
public:
	int from, to; // keyframe ids
	cv::Mat R, t;
	int numMatches;
};

class Map3d
{
public:
	string			datapath;
	vector<Frame>	frames;
	vector<int>		keyframeIdx;
	vector<LmkLine>	lmklines;
	vector<string>  paths;
	cv::Mat			vel, w; // linear vol in world coord
	int				lastLoopCloseFrame;
	bool			useConstVel;

	Map3d(){}
	Map3d(string path):datapath(path){}
	Map3d(vector<string> paths_):paths(paths_){}
	void slam();	
	void compGT();
#ifdef SLAM_LBA	
	void lba(int numPos=3, int numFrm=5);
	void lba_g2o(int numPos=3, int numFrm=5, int mode = 0);
	void loopclose();
	void correctPose(vector<PoseConstraint> pcs);
	void correctAll(vector<vector<int> > lmkGidMatches);
#endif	

//void draw3d();
};


class SystemParameters 
{
public:
	double	ratio_of_collinear_pts;		// decide if a frameline has enough collinear pts
	double	pt2line_dist_extractline;	// threshold pt to line distance when detect lines from pts
	double	pt2line_mahdist_extractline;// threshold for pt to line mahalanobis distance
	int		ransac_iters_extract_line;	// max ransac iters when detect lines from pts
	double	line_segment_len_thresh;		// min lenght of image line segment to use 
	double	ratio_support_pts_on_line;	// the ratio of the number of filled cells over total cell number along a line
										// to check if a line has enough points covering the whole range
	int		num_cells_lineseg_range;	// divide a linesegment into multiple cells 
	double	line3d_length_thresh;		// frameline length threshold in 3d
	double	stdev_sample_pt_imgline;	// std dev of sample point from an image line
	double  depth_stdev_coeff_c1;		// c1,c2,c3: coefficients of depth noise quadratic function
	double  depth_stdev_coeff_c2;
	double  depth_stdev_coeff_c3;

	int 	line_sample_max_num;
	int 	line_sample_min_num;
	double 	line_sample_interval;
	int 	line3d_mle_iter_num;
	int 	line_detect_algorithm;
	double 	msld_sample_interval;
	int 	ransac_iters_line_motion;
	int 	adjacent_linematch_window;
	int 	line_match_number_weight;
	int 	min_feature_matches;
	double 	max_mah_dist_for_inliers;
	double  g2o_line_error_weight;
	int 	min_matches_loopclose;



	int		num_2dlinematch_keyframe;	// detect keyframe, minmum number of 2d line matches left
	int		num_3dlinematch_keyframe;
	double	pt2line3d_dist_relmotion;	// in meter, 
	double  line3d_angle_relmotion;		// in degree
	int		num_raw_frame_skip;			// number of raw frame to skip when tracking lines
	int		window_length_keyframe;		
	bool	fast_motion;
	double	inlier_ratio_constvel;
	int		num_pos_lba;
	int		num_frm_lba;
	// ----- lsd setting -----
	double lsd_angle_th;
	double lsd_density_th;
	// ----- loop closing -----
	double loopclose_interval;  // frames, check loop closure
	int	   loopclose_min_3dmatch;  // min_num for 3d line matches between two frames

	bool	g2o_BA_use_kernel;
	double  g2o_BA_kernel_delta;

	bool 	dark_ligthing;
	double	max_img_brightness;


	void init();
	SystemParameters(){}

};



#endif //LINESLAM_H