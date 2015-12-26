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


#ifndef LINESLAM_UTILS_H
#define LINESLAM_UTILS_H
#ifdef _WIN32
	#include <Windows.h>
#else
	#include <time.h>
#endif

#include "lineslam.h"
#include "../node.h"

void getConfigration (string* datapath, cv::Mat& K, cv::Mat& distCoeffs) ;
void getConfigration (vector<string>& datapaths, cv::Mat& K, cv::Mat& distCoeffs, int n_data=1);
ntuple_list callLsd (IplImage* src);
cv::Point2d mat2cvpt (const cv::Mat& m);
cv::Point3d mat2cvpt3d (cv::Mat m);
cv::Mat cvpt2mat( cv::Point2d p, bool homo=true);
cv::Mat cvpt2mat(const cv::Point3d& p, bool homo=true);
cv::Mat array2mat(double a[], int n);

double getMonoSubpix(const cv::Mat& img, cv::Point2d pt);

void showImage(string name, cv::Mat *img, int width=640) ;
string num2str(double i);

template<class bidiiter> //Fisher-Yates shuffle
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
	size_t left = std::distance(begin, end);
	while (num_random--) {
		bidiiter r = begin;
		std::advance(r, rand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}    
	return begin;
}

void computeLine3d_svd (vector<cv::Point3d> pts, cv::Point3d& mean, cv::Point3d& drct);
void computeLine3d_svd (vector<RandomPoint3d> pts, cv::Point3d& mean, cv::Point3d& drct);
void computeLine3d_svd (const vector<RandomPoint3d>& pts, const vector<int>& idx, cv::Point3d& mean, cv::Point3d& drct);
RandomLine3d extract3dline(const vector<cv::Point3d>& pts);
RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d>& pts);
cv::Point3d projectPt3d2Ln3d (const cv::Point3d& P, const cv::Point3d& mid, const cv::Point3d& drct);
cv::Point3d projectPt3d2Ln3d_2 (const cv::Point3d& P, const cv::Point3d& A, const cv::Point3d& B);
bool verify3dLine(vector<cv::Point3d> pts, cv::Point3d A, cv::Point3d B);
bool verify3dLine(const vector<RandomPoint3d>& pts, const cv::Point3d& A,  const cv::Point3d& B);
double dist3d_pt_line (cv::Point3d X, cv::Point3d A, cv::Point3d B);
double dist3d_pt_line (cv::Mat x, cv::Point3d A, cv::Point3d B);
double dist3d_pt_line (Eigen::Vector4f X, Eigen::Vector4f A, Eigen::Vector4f B);

double depthStdDev (double d) ;
double depthStdDev (double d, double time_diff_sec) ;
RandomPoint3d compPt3dCov (cv::Point3d pt, cv::Mat K, double);
double mah_dist3d_pt_line (cv::Point3d p, cv::Mat C, cv::Point3d q1, cv::Point3d q2);
double mah_dist3d_pt_line (cv::Point3d p, cv::Mat R, cv::Mat s, cv::Point3d q1, cv::Point3d q2);
double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Point3d& q1, const cv::Point3d& q2);
double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Mat& q1, const cv::Mat& q2);
cv::Point3d mahvec_3d_pt_line (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
cv::Point3d mahvec_3d_pt_line(const RandomPoint3d& pt, cv::Mat q1, cv::Mat q2);
cv::Point3d closest_3dpt_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
double closest_3dpt_ratio_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
void termReason(int info);
void MLEstimateLine3d(RandomLine3d& line,	int maxIter);
void MLEstimateLine3d_perpdist (vector<cv::Point3d> pts, cv::Point3d midpt, cv::Point3d dirct, RandomLine3d& line);
void MlestimateLine3dCov (double* p, const int n, int i1, int i2, const cv::Mat& cov_meas_inv,
						  cv::Mat& cov_i1, cv::Mat& cov_i2);
void write_linepairs_tofile(vector<RandomLine3d> a, vector<RandomLine3d> b, string fname, double timestamp);
class MyTimer
{
#ifdef _WIN32
public:
	MyTimer() {	QueryPerformanceFrequency(&TickPerSec);	}

	LARGE_INTEGER TickPerSec;        // ticks per second
	LARGE_INTEGER Tstart, Tend;           // ticks

	double time_ms;
	double time_s;
	void start()  {	QueryPerformanceCounter(&Tstart);}
	void end() 	{
		QueryPerformanceCounter(&Tend);
		time_ms = (Tend.QuadPart-Tstart.QuadPart)*1000.0/TickPerSec.QuadPart;
		time_s = time_ms/1000.0;
	}
#else
public:
	timespec t0, t1; 
	MyTimer() {}
	double time_ms;
	double time_s;
	void start() {
		clock_gettime(CLOCK_REALTIME, &t0);
	}
	void end() {
		clock_gettime(CLOCK_REALTIME, &t1);
		time_ms = t1.tv_sec * 1000 + t1.tv_nsec/1000000.0 - (t0.tv_sec * 1000 + t0.tv_nsec/1000000.0);
		time_s = time_ms/1000.0;			
	}
#endif	
};

int computeMSLD (FrameLine& l, cv::Mat* xGradient, cv::Mat* yGradient) ;
int computeMSLD (FrameLine& l, double* xGradient, double* yGradient, int width, int height);
double line_to_line_dist2d(const FrameLine& a, const FrameLine& b);
void trackLine (const vector<FrameLine>& f1, const vector<FrameLine>& f2, vector<vector<int> >& matches);
double lineSegmentOverlap(const FrameLine& a, const FrameLine& b);
bool computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);
vector<int> computeRelativeMotion_Ransac (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);
void optimizeRelmotion(vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);
cv::Mat q2r(cv::Mat q);
cv::Mat q2r (double* q);
cv::Mat r2q(cv::Mat R);
Eigen::Vector4d r2q(Eigen::Matrix3d R);
cv::Mat vec2SkewMat (cv::Point3d vec);
void write2file (Map3d& m, string suffix);
double pesudoHuber(double e, double band);
double rotAngle (cv::Mat R);
void matchLine (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches);
double ave_img_bright(cv::Mat img);
bool get_pt_3d (cv::Point2d p2, cv::Point3d& p3, const cv::Mat& depth);


bool getTransform_PtsLines_ransac (const Node* trainNode, const Node* queryNode, 
									const std::vector<cv::DMatch> all_point_matches,
									const std::vector<cv::DMatch> all_line_matches,
									std::vector<cv::DMatch>& output_point_inlier_matches,
									std::vector<cv::DMatch>& output_line_inlier_matches,
									Eigen::Matrix4f& ransac_tf, 
									float& inlier_rmse);
void optimize_motion_g2o(vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);

//// line segment struct for EDline detection
struct LS {
  double sx, sy, ex, ey; // Start & end coordinates of the line segment
};

LS* callEDLines (const cv::Mat& im_uchar, int* numLines);

void fivepoint_stewnister (const cv::Mat& P1, const cv::Mat& P2, vector<cv::Mat> &Es);
bool isEssnMatSimilar(cv::Mat E1, cv::Mat E2);
double fund_samperr (cv::Mat x1_d, cv::Mat x2_d, cv::Mat F_d) ;
void optimizeEmat (cv::Mat p1, cv::Mat p2, cv::Mat K, cv::Mat *E);
void decEssential (cv::Mat *E, cv::Mat *R1, cv::Mat *R2, cv::Mat *t);
cv::Mat findTrueRt(cv::Mat R1,cv::Mat R2, cv::Mat t,cv::Point2d q1,cv::Point2d q2) ;
double rotateAngleDeg(cv::Mat R);
cv::Mat triangulatePoint (const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& K,
	cv::Point2d pt1, cv::Point2d pt2);
int essn_ransac (cv::Mat* pts1, cv::Mat* pts2, cv::Mat* E, cv::Mat K, 
	cv::Mat* inlierMask, int imsize);
void essn_ransac (cv::Mat* pts1, cv::Mat* pts2, vector<cv::Mat>& bestEs, cv::Mat K, 
					vector<cv::Mat>& inlierMasks, int imsize, bool usePrior, cv::Mat t_prior);

cv::Mat getRotationFromPoints( const Node* trainNode, const Node* queryNode, 
							   const std::vector<cv::DMatch> all_matches,
								std::vector<cv::DMatch>& output_inlier_matches,
								bool& valid);



ntuple_list callLsd (IplImage* src, bool bShow);
Eigen::Matrix3f compPt3dCov (Eigen::Vector3f pt, double, double, double, double);
cv::Mat MleLine3dCov(const vector<RandomPoint3d>& pts, int idx1, int idx2, const double l[6]);


#endif
