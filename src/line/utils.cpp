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

#include <Eigen/Core>
#define ARMA_DONT_USE_CXX11
#include <armadillo>
#include "utils.h"
//#include "adept.h"

#define USE_VEC4MAT
#define SAVE_MEMORY
extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

void getConfigration (string* datapath, cv::Mat& K, cv::Mat& distCoeffs)  
{		
	string fname = "./settings.ini";
	ifstream config(fname.c_str());

	string cammode;
	if (config.is_open())  {
		getline(config, cammode);
		getline(config, *datapath);
	}
	config.close();
	int cam_model = atoi(cammode.c_str());
	switch (cam_model) {
	case 0: // default
		K = (cv::Mat_<double>(3,3)<<525,  0,  319.5,  
			0,  525,  239.5,
			0,  0,    1.0000);
		distCoeffs = (cv::Mat_<double>(4,1)<<0,0,0,0);
		break;
	case 1: // fr1
		K = (cv::Mat_<double>(3,3)<< 517.3,  0,  318.6,  
			0,  516.5,  255.3,
			0,      0,    1.0000);
		distCoeffs = (cv::Mat_<double>(5,1)<<0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
		break;
	case 2: // fr2
		K = (cv::Mat_<double>(3,3)<< 520.9,  0,  325.1,  
			0,  521.0,  249.7,
			0,      0,    1.0000);
		distCoeffs = (cv::Mat_<double>(5,1)<<0.2312, -0.7849, -0.0033, -0.0001, 0.9172);
		break;
	default:;
	}
}

void getConfigration (vector<string>& datapaths, cv::Mat& K, cv::Mat& distCoeffs, int n_data) 
{		
	string fname = "./settings.ini";
	ifstream config(fname.c_str());

	string cammode;
	if (config.is_open())  {
		getline(config, cammode);
		for(int i=0; i<n_data; ++i) {
			string datapath;
			getline(config, datapath);
			if(datapath.size()<1)
				break; 
			else
				datapaths.push_back(datapath);
		}
	}
	config.close();
	int cam_model = atoi(cammode.c_str());
	switch (cam_model) {
	case 0: // default
		K = (cv::Mat_<double>(3,3)<<525,  0,  319.5,  
			0,  525,  239.5,
			0,  0,    1.0000);
		distCoeffs = (cv::Mat_<double>(4,1)<<0,0,0,0);
		break;
	case 1: // fr1
		K = (cv::Mat_<double>(3,3)<< 517.3,  0,  318.6,  
			0,  516.5,  255.3,
			0,      0,    1.0000);
		distCoeffs = (cv::Mat_<double>(5,1)<<0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
		break;
	case 2: // fr2
		K = (cv::Mat_<double>(3,3)<< 520.9,  0,  325.1,  
			0,  521.0,  249.7,
			0,      0,    1.0000);
		distCoeffs = (cv::Mat_<double>(5,1)<<0.2312, -0.7849, -0.0033, -0.0001, 0.9172);
		break;
	default:;
	}
}


ntuple_list callLsd (IplImage* src) 
	// use LSD to extract line segments from an image
	// input : an image grayscale, grayscale level 0 - 255
	// output: a list of line segments - endpoint x-y coords + ...  
{
	IplImage* src_gray = src;
	image_double image; //image_double is a struct defined in 'lsd.h'
	ntuple_list lsd_out;
	unsigned int w = src->width;
	unsigned int h = src->height;
	image = new_image_double(w,h);
	CvScalar s;// to get image values
	for (int x=0;x<w;++x){
		for(int y=0;y<h;++y){			
			s=cvGet2D(src_gray,y,x); 
			image->data[x+y*image->xsize] = s.val[0];			
		}
	}
	//////////// lsd assumes input image with grayscale level [0, 255], no matter float or char
	lsd_out = lsd(image);	
	
	free_image_double(image);
	return lsd_out;
}

ntuple_list callLsd (IplImage* src, bool bShow) 
	// use LSD to extract line segments from an image
	// input : an image color/grayscale
	// output: a list of line segments - endpoint x-y coords + ...  
{
	IplImage* src_gray = src;
	image_double image; //image_double is a struct defined in 'lsd.h'
	ntuple_list lsd_out;
	unsigned int w = src->width;
	unsigned int h = src->height;
	image = new_image_double(w,h);
	CvScalar s;// to get image values
	for (int x=0;x<w;++x){
		for(int y=0;y<h;++y){			
			s=cvGet2D(src_gray,y,x); 
			image->data[x+y*image->xsize] = s.val[0];
		}
	}
//	MyTimer tm; tm.start();
	lsd_out = lsd(image);
//	tm.end(); cout<<tm.time_ms<<'\t';
	if(bShow){ 		// print output and plot in image 
		IplImage *canvas = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
		printf("%u Line segments found by LSD.\n",lsd_out->size);
		int dim = lsd_out->dim;
		for(int i=0;i<lsd_out->size;i++){
			cvLine(src,
				cvPoint(lsd_out->values[i*dim],    lsd_out->values[i*dim+1]),
				cvPoint(lsd_out->values[i*dim+2],  lsd_out->values[i*dim+3]),
				cvScalar(255, 255, 255, 0),      
				1.5, 8, 0);		
		}
		cvNamedWindow("LSD Result",1);
		cvShowImage("LSD Result",src);
		cvWaitKey();
		cvDestroyWindow("LSD Result");
		cvReleaseImage(&canvas);			
	}
	free_image_double(image);
	//	cvReleaseImage(&src_gray);
	return lsd_out;
}

cv::Point2d mat2cvpt (const cv::Mat& m)
	// 3x1 mat => point
{
	if (m.cols * m.rows == 2)
		return cv::Point2d(m.at<double>(0), m.at<double>(1));
	if (m.cols * m.rows ==3)
		return cv::Point2d(m.at<double>(0)/m.at<double>(2),
		m.at<double>(1)/m.at<double>(2));
	else
		cerr<<"input matrix dimmension wrong!";	
}

cv::Point3d mat2cvpt3d (cv::Mat m)
	// 3x1 mat => point
{
	if (m.cols * m.rows ==3)
		return cv::Point3d(m.at<double>(0),
		m.at<double>(1),
		m.at<double>(2));
	else
		cerr<<"input matrix dimmension wrong!";	
}

cv::Mat cvpt2mat(cv::Point2d p, bool homo)
{
	if (homo)
		return (cv::Mat_<double>(3,1)<<p.x, p.y, 1);
	else
		return (cv::Mat_<double>(2,1)<<p.x, p.y);
}

cv::Mat cvpt2mat(const cv::Point3d& p, bool homo)
	// this function is slow!
	// return cv::Mat(3,1,CV_64,arrary) does not work!
{
	if (homo)
		return (cv::Mat_<double>(4,1)<<p.x, p.y, p.z, 1);
	else {
		return (cv::Mat_<double>(3,1)<<p.x, p.y, p.z);
		
	}
}
cv::Mat array2mat(double a[], int n) // inhomo mat
	// n is the size of a[]
{
	return cv::Mat(n,1,CV_64F,a);
}

cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2d pt)
{
	cv::Mat patch;
	cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
	return patch.at<cv::Vec3b>(0,0);
}

double getMonoSubpix(const cv::Mat& img, cv::Point2d pt)
	// get pixel value of non-integer position
	// input: img must be CV_32F type
{
	cv::Mat patch;
	cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
	return patch.at<float>(0,0);
}

void showImage(string name, cv::Mat *img, int width) 
{
	double ratio = double(width)/img->cols;
	cv::namedWindow(name,0);
	cv::imshow(name,*img);
	cvResizeWindow(name.c_str(),width,int(img->rows*ratio));
}

string num2str(double i)
{
	stringstream ss;
	ss<<i;
	return ss.str();
}

RandomLine3d extract3dline(const vector<cv::Point3d>& pts)
	// extract a single 3d line from point clouds using ransac
	// input: 3d points
	// output: inlier points, line parameters: midpt and direction
{
	int maxIterNo = sysPara.ransac_iters_extract_line;	
	double distThresh = sysPara.pt2line_dist_extractline; // meter
	// distance threshold should be adapted to line length and depth
	int minSolSetSize = 2;

	vector<int> indexes(pts.size());
	for (int i=0; i<indexes.size(); ++i) indexes[i]=i;
	vector<cv::Point3d> maxInlierSet;
	cv::Point3d bestA, bestB;
	for(int iter=0; iter<maxIterNo;iter++) {
		vector<cv::Point3d> inlierSet;
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
		cv::Point3d A = pts[indexes[0]], B = pts[indexes[1]];
		// compute a line from A and B
		if (cv::norm(B-A) < EPS ) continue; 
		for (int i=0; i<pts.size(); ++i) {
			// compute distance to AB
			double dist = dist3d_pt_line(pts[i],A,B);
			if (dist<distThresh) {
				inlierSet.push_back(pts[i]);
			}
		}		
		if(inlierSet.size() > maxInlierSet.size())	{
			if (verify3dLine(inlierSet, A, B)) {
				maxInlierSet = inlierSet;	
				bestA = A; bestB = B;
			}
		}
	}
	RandomLine3d rl;
	for(int i=0; i<maxInlierSet.size(); ++i)
		rl.pts.push_back(RandomPoint3d(maxInlierSet[i]));
	
	if (maxInlierSet.size() >= 2) {
		cv::Point3d m = (bestA+bestB)*0.5, d = bestB-bestA;
		// optimize and reselect inliers
		// compute a 3d line using algebraic method	
		while(true) {
			vector<cv::Point3d> tmpInlierSet;
			cv::Point3d tmp_m, tmp_d;
			computeLine3d_svd(maxInlierSet, tmp_m, tmp_d);
			for(int i=0; i<pts.size(); ++i) {				
				if(dist3d_pt_line(pts[i],tmp_m, tmp_m+tmp_d) < distThresh) {
					tmpInlierSet.push_back(pts[i]);			
				}
			}
			if(tmpInlierSet.size()>maxInlierSet.size()) {
				maxInlierSet = tmpInlierSet;
				m = tmp_m;
				d = tmp_d;
			} else 
				break;
		}
		// find out two endpoints
		double minv=100, maxv=-100;
		int	   idx_end1 = 0, idx_end2 = 0;
		for(int i=0; i<maxInlierSet.size(); ++i) {
			double dproduct = (maxInlierSet[i]-m).dot(d);
			if ( dproduct < minv) {
				minv = dproduct;
				idx_end1 = i;
			}
			if (dproduct > maxv) {
				maxv = dproduct;
				idx_end2 = i;
			}
		}		
		rl.A = maxInlierSet[idx_end1];
		rl.B = maxInlierSet[idx_end2];
	
		rl.pts.clear();
		for(int i=0; i<maxInlierSet.size(); ++i)
			rl.pts.push_back(RandomPoint3d(maxInlierSet[i]));
	}

	return rl;	

}

RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d>& pts)
	// extract a single 3d line from point clouds using ransac and mahalanobis distance
	// input: 3d points and covariances
	// output: inlier points, line parameters: midpt and direction
{
	int maxIterNo = min(sysPara.ransac_iters_extract_line,
					int(pts.size()*(pts.size()-1)*0.5));	
	double distThresh = sysPara.pt2line_mahdist_extractline; // meter
	// distance threshold should be adapted to line length and depth
	int minSolSetSize = 2;

	vector<int> indexes(pts.size());
	for (int i=0; i<indexes.size(); ++i) indexes[i]=i;
	vector<int> maxInlierSet;
	RandomPoint3d bestA, bestB;
	for(int iter=0; iter<maxIterNo;iter++) {
		vector<int> inlierSet;
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
		const RandomPoint3d& A = pts[indexes[0]];
		const RandomPoint3d& B = pts[indexes[1]];
		// compute a line from A and B
		if (cv::norm(B.pos-A.pos) < EPS ) continue; 
		for (int i=0; i<pts.size(); ++i) {
			// compute distance to AB
			double dist = mah_dist3d_pt_line(pts[i], A.pos, B.pos);
			if (dist<distThresh) {
				inlierSet.push_back(i);
			}
		}		
		if(inlierSet.size() > maxInlierSet.size())	{
			vector<RandomPoint3d> inlierPts(inlierSet.size());
			for(int ii=0; ii<inlierSet.size(); ++ii)
				inlierPts[ii]=pts[inlierSet[ii]];
			if (verify3dLine(inlierPts, A.pos, B.pos)) {
				maxInlierSet = inlierSet;	
				bestA = pts[indexes[0]]; bestB = pts[indexes[1]];
			}
		}
	//	cout<<iter<<'\t'<<maxInlierSet.size()<<endl;
		if( maxInlierSet.size() > pts.size()*0.9)
			break;
	}
		
	RandomLine3d rl;
	if (maxInlierSet.size() >= 2) {
		cv::Point3d m = (bestA.pos+bestB.pos)*0.5, d = bestB.pos-bestA.pos;		
		// optimize and reselect inliers
		// compute a 3d line using algebraic method	
		while(true) {
			vector<int> tmpInlierSet;
			cv::Point3d tmp_m, tmp_d;
			computeLine3d_svd(pts,maxInlierSet, tmp_m, tmp_d);
			for(int i=0; i<pts.size(); ++i) {
				if(mah_dist3d_pt_line(pts[i], tmp_m, tmp_m+tmp_d) < distThresh) {
					tmpInlierSet.push_back(i);					
				}
			}
			if(tmpInlierSet.size() > maxInlierSet.size()) {
				maxInlierSet = tmpInlierSet;
				m = tmp_m;
				d = tmp_d;
			} else 
				break;
		}
	  // find out two endpoints
		double minv=100, maxv=-100;
		int	   idx_end1 = 0, idx_end2 = 0;
		for(int i=0; i<maxInlierSet.size(); ++i) {
			double dproduct = (pts[maxInlierSet[i]].pos-m).dot(d);
			if ( dproduct < minv) {
				minv = dproduct;
				idx_end1 = i;
			}
			if (dproduct > maxv) {
				maxv = dproduct;
				idx_end2 = i;
			}
		}	
		rl.A = pts[maxInlierSet[idx_end1]].pos;
		rl.B = pts[maxInlierSet[idx_end2]].pos;	
	}	
	rl.pts.resize(maxInlierSet.size());
	for(int i=0; i< maxInlierSet.size();++i) rl.pts[i]= pts[maxInlierSet[i]] ;
	return rl;		 
}

void computeLine3d_svd (vector<cv::Point3d> pts, cv::Point3d& mean, cv::Point3d& drct)
	// input: collinear 3d points with noise
	// output: line direction vector and point
	// method: linear equation, PCA
{
	int n = pts.size();
	mean = cv::Point3d(0,0,0);
	for(int i=0; i<n; ++i) {
		mean =  mean + pts[i];
	}
	mean = mean * (1.0/n);
	cv::Mat P(3,n,CV_64F);
	for(int i=0; i<n; ++i) {
		pts[i] =  pts[i] - mean;
		cvpt2mat(pts[i],0).copyTo(P.col(i));
	}
	cv::SVD svd(P.t());
	drct = mat2cvpt3d(svd.vt.row(0));
}

void computeLine3d_svd (vector<RandomPoint3d> pts, cv::Point3d& mean, cv::Point3d& drct)
	// input: collinear 3d points with noise
	// output: line direction vector and point
	// method: linear equation, PCA
{
	int n = pts.size();
	mean = cv::Point3d(0,0,0);
	for(int i=0; i<n; ++i) {
		mean =  mean + pts[i].pos;
	}
	mean = mean * (1.0/n);
	cv::Mat P(3,n,CV_64F);
	for(int i=0; i<n; ++i) {
	//	pts[i].pos =  pts[i].pos - mean;
	//	cvpt2mat(pts[i].pos,0).copyTo(P.col(i));
		double pos[3] = {pts[i].pos.x - mean.x,pts[i].pos.y - mean.y,pts[i].pos.z - mean.z};
		array2mat(pos,3).copyTo(P.col(i));
	}
	cv::SVD svd(P.t());
	drct = mat2cvpt3d(svd.vt.row(0));
}

void computeLine3d_svd (const vector<RandomPoint3d>& pts, const vector<int>& idx, cv::Point3d& mean, cv::Point3d& drct)
	// input: collinear 3d points with noise
	// output: line direction vector and point
	// method: linear equation, PCA
{
	int n = idx.size();
	mean = cv::Point3d(0,0,0);
	for(int i=0; i<n; ++i) {
		mean =  mean + pts[idx[i]].pos;
	}
	mean = mean * (1.0/n);
	cv::Mat P(3,n,CV_64F);
	for(int i=0; i<n; ++i) {
	//	pts[i].pos =  pts[i].pos - mean;
	//	cvpt2mat(pts[i].pos,0).copyTo(P.col(i));
		double pos[3] = {pts[idx[i]].pos.x-mean.x, pts[idx[i]].pos.y-mean.y, pts[idx[i]].pos.z-mean.z};
		array2mat(pos,3).copyTo(P.col(i));
	}
	
	cv::SVD svd(P.t(), cv::SVD::MODIFY_A);  // FULL_UV is 60 times slower
	
	drct = mat2cvpt3d(svd.vt.row(0));
}


cv::Point3d projectPt3d2Ln3d (const cv::Point3d& P, const cv::Point3d& mid, const cv::Point3d& drct)
	// project a 3d point P to a 3d line (represented with midpt and direction)
{
	cv::Point3d A = mid;
	cv::Point3d B = mid + drct;
	cv::Point3d AB = B-A;
	cv::Point3d AP = P-A;
	return A + (AB.dot(AP)/(AB.dot(AB)))*AB;
}

cv::Point3d projectPt3d2Ln3d_2 (const cv::Point3d& P, const cv::Point3d& A, const cv::Point3d& B)
	// project a 3d point P to a 3d line (represented by 2 pts A B)
{
	cv::Point3d AB = B-A;
	cv::Point3d AP = P-A;
	return A + (AB.dot(AP)/(AB.dot(AB)))*AB;
}

bool verify3dLine(vector<cv::Point3d> pts, cv::Point3d A, cv::Point3d B)
	// input: line AB, collinear points
	// output: whether AB is a good representation for points
	// method: divide AB (or CD, which is endpoints of the projected points on AB) 
	// into n sub-segments, detect how many sub-segments containing
	// at least one point(projected onto AB), if too few, then it implies invalid line
{
	int nCells = sysPara.num_cells_lineseg_range; // number of cells
	int* cells = new int[nCells];
	double ratio = sysPara.ratio_support_pts_on_line;
	for(int i=0; i<nCells; ++i) cells[i] = 0;
	int nPts = pts.size();
	// find 2 extremities of points along the line direction
	double minv=100, maxv=-100;
	int	   idx1 = 0, idx2 = 0;
	for(int i=0; i<nPts; ++i) {
		if ((pts[i]-A).dot(B-A) < minv) {
			minv = (pts[i]-A).dot(B-A);
			idx1 = i;
		}
		if ((pts[i]-A).dot(B-A) > maxv) {
			maxv = (pts[i]-A).dot(B-A);
			idx2 = i;
		}
	}	
	cv::Point3d C = projectPt3d2Ln3d (pts[idx1], (A+B)*0.5, B-A);
	cv::Point3d D = projectPt3d2Ln3d (pts[idx2], (A+B)*0.5, B-A);
	double cd = cv::norm(D-C);
	if(cd < EPS) {
		delete[] cells;
		return false;
	}
	for(int i=0; i<nPts; ++i) {
		cv::Point3d X = pts[i];
		double lambda = abs((X-C).dot(D-C)/cd/cd); // 0 <= lambd <=1
		if (lambda>=1) {
			cells[nCells-1] += 1;
		} else {			
			cells[(unsigned int)floor(lambda*10)] += 1;
		}		
	}
	double sum = 0;
	for (int i=0; i<nCells; ++i) {
		//		cout<<cells[i]<<"\t";
		if (cells[i] > 0 )
			sum ++;
	}
	//	cout<<'\t'<<sum<<endl;
	delete[] cells;
	if(sum/nCells > ratio) {
		return true;
	} else {
		return false;
	}
}

bool verify3dLine(const vector<RandomPoint3d>& pts, const cv::Point3d& A,  const cv::Point3d& B)
	// input: line AB, collinear points
	// output: whether AB is a good representation for points
	// method: divide AB (or CD, which is endpoints of the projected points on AB) 
	// into n sub-segments, detect how many sub-segments containing
	// at least one point(projected onto AB), if too few, then it implies invalid line
{
	int nCells = sysPara.num_cells_lineseg_range; // number of cells
	int* cells = new int[nCells];
	double ratio = sysPara.ratio_support_pts_on_line;
	for(int i=0; i<nCells; ++i) cells[i] = 0;
	int nPts = pts.size();
	// find 2 extremities of points along the line direction
	double minv=100, maxv=-100;
	int	   idx1 = 0, idx2 = 0;
	for(int i=0; i<nPts; ++i) {
		if ((pts[i].pos-A).dot(B-A) < minv) {
			minv = (pts[i].pos-A).dot(B-A);
			idx1 = i;
		}
		if ((pts[i].pos-A).dot(B-A) > maxv) {
			maxv = (pts[i].pos-A).dot(B-A);
			idx2 = i;
		}
	}	
	cv::Point3d C = projectPt3d2Ln3d (pts[idx1].pos, (A+B)*0.5, B-A);
	cv::Point3d D = projectPt3d2Ln3d (pts[idx2].pos, (A+B)*0.5, B-A);
	double cd = cv::norm(D-C);
	if(cd < EPS) {
		delete[] cells;
		return false;
	}
	for(int i=0; i<nPts; ++i) {
		cv::Point3d X = pts[i].pos;
		double lambda = abs((X-C).dot(D-C)/cd/cd); // 0 <= lambd <=1
		if (lambda>=1) {
			cells[nCells-1] += 1;
		} else {			
			cells[(unsigned int)floor(lambda*10)] += 1;
		}		
	}
	double sum = 0;
	for (int i=0; i<nCells; ++i) {
		//		cout<<cells[i]<<"\t";
		if (cells[i] > 0 )
			sum=sum+1;
	}

	delete[] cells;
	if(sum/nCells > ratio) {
		return true;
	} else {
		return false;
	}
}

double dist3d_pt_line (cv::Point3d X, cv::Point3d A, cv::Point3d B)
	// input: point X, line (A,B)
{
	if(cv::norm(A-B)<EPS) {
		cerr<<"error in function dist3d_pt_line: line length can not be 0!"<<endl;
		return -1;
	}
	double ax = cv::norm(X-A);
	cv::Point3d nvAB = (B-A) * (1/cv::norm(A-B));
	return sqrt(abs( ax*ax - ((X-A).dot(nvAB))*((X-A).dot(nvAB))));
}

double dist3d_pt_line (Eigen::Vector4f X, Eigen::Vector4f A, Eigen::Vector4f B)
	// input: point X, line (A,B)
// the last element of vector4f is 1.0
{
	if((A-B).norm()<EPS) {
		cerr<<"error in function dist3d_pt_line: line length can not be 0!"<<endl;
		return -1;
	}
	double ax = (X-A).norm();
	Eigen::Vector3f nvAB = (B.head(3)-A.head(3))/(A-B).norm();
	return sqrt(abs( ax*ax - (X.head(3)-A.head(3)).dot(nvAB) * (X.head(3)-A.head(3)).dot(nvAB) ));
}

double dist3d_pt_line (cv::Mat X, cv::Point3d A, cv::Point3d B)
// input: point X, line (A,B)
{
	return dist3d_pt_line(mat2cvpt3d(X),A,B);
}

double depthStdDev (double d) 
	// standard deviation of depth d 
	// in meter
{	
	double c1, c2, c3;
	c1 = sysPara.depth_stdev_coeff_c1;
	c2 = sysPara.depth_stdev_coeff_c2;
	c3 = sysPara.depth_stdev_coeff_c3; 

	return c1*d*d + c2*d + c3;

//	return sysPara.depth_stdev_coeff_c1*d*d + sysPara.depth_stdev_coeff_c2*d + sysPara.depth_stdev_coeff_c3;
}

double depthStdDev (double d, double time_diff_sec) 
	// standard deviation of depth d in meter
{	
	if (time_diff_sec > 1000)
{
	cout<<"depthStdDev: "<<time_diff_sec<<endl;
	int tmp; cin>>tmp;
}
	double c1, c2, c3;
	c1 = sysPara.depth_stdev_coeff_c1;
	c2 = sysPara.depth_stdev_coeff_c2 + max(time_diff_sec - 0.005, 0.0) * 0.5;
	c3 = sysPara.depth_stdev_coeff_c3; 
	
	return c1*d*d + c2*d + c3;

//	return sysPara.depth_stdev_coeff_c1*d*d + sysPara.depth_stdev_coeff_c2*d + sysPara.depth_stdev_coeff_c3;
}


RandomPoint3d compPt3dCov (cv::Point3d pt, cv::Mat K, double time_diff_sec)
{

	double f = K.at<double>(0,0), // focal length
		cu = K.at<double>(0,2),
		cv = K.at<double>(1,2);
	double sigma_impt = sysPara.stdev_sample_pt_imgline; // std dev of image sample point
	
/*	//// opencv mat operation is slower than armadillo
	cv::Mat J0 = (cv::Mat_<double>(3,3)<< pt.z/f, 0, pt.x/pt.z,
		0, pt.z/f, pt.y/pt.z,
		0,0,1);
	cv::Mat cov_g_d0 = (cv::Mat_<double>(3,3)<<sigma_impt*sigma_impt, 0, 0,
		0, sigma_impt*sigma_impt, 0,
		0, 0, depthStdDev(pt.z)*depthStdDev(pt.z));
	cv::Mat cov0 = J0 * cov_g_d0 * J0.t();
	return RandomPoint3d(pt, cov0);
*/

	arma::mat J(3,3,arma::fill::zeros), cov_g_d(3,3,arma::fill::zeros);
	J(0,0) = pt.z/f; J(0,2) = pt.x/pt.z;
	J(1,1) = pt.z/f; J(1,2) = pt.y/pt.z; J(2,2) = 1;
	cov_g_d(0,0) = sigma_impt*sigma_impt;
	cov_g_d(1,1) = sigma_impt*sigma_impt;
	cov_g_d(2,2) = depthStdDev(pt.z, time_diff_sec)*depthStdDev(pt.z, time_diff_sec);
	arma::mat cov = J*cov_g_d*J.t();

	cv::Mat c = (cv::Mat_<double>(3,3) << cov(0,0), cov(0,1), cov(0,2),
							cov(1,0), cov(1,1), cov(1,2),
							cov(2,0), cov(2,1), cov(2,2));	
	return RandomPoint3d(pt,c);

}

Eigen::Matrix3f compPt3dCov (Eigen::Vector3f pt, double f, double cu, double cv, double time_diff_sec)
{

	double sigma_impt = sysPara.stdev_sample_pt_imgline; // std dev of image sample point
	
	arma::mat J(3,3,arma::fill::zeros), cov_g_d(3,3,arma::fill::zeros);
	J(0,0) = pt(2)/f; J(0,2) = pt(0)/pt(2);
	J(1,1) = pt(2)/f; J(1,2) = pt(1)/pt(2); J(2,2) = 1;
	cov_g_d(0,0) = sigma_impt*sigma_impt;
	cov_g_d(1,1) = sigma_impt*sigma_impt;
	cov_g_d(2,2) = depthStdDev(pt(2), time_diff_sec)*depthStdDev(pt(2), time_diff_sec);
	arma::mat cov = J*cov_g_d*J.t();

	Eigen::Matrix3f c; c << cov(0,0), cov(0,1), cov(0,2),
							cov(1,0), cov(1,1), cov(1,2),
							cov(2,0), cov(2,1), cov(2,2);	
	return c;

}


double mah_dist3d_pt_line (cv::Point3d p, cv::Mat C, cv::Point3d q1, cv::Point3d q2)
	// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
{
	if (C.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: cov matrix must be 3x3"<<endl;
		return -1;
	}
	cv::SVD svd(C);
	cv::Mat invRootD = (cv::Mat_<double>(3,3)<< 1/sqrt(svd.w.at<double>(0)), 0, 0,
		0, 1/sqrt(svd.w.at<double>(1)), 0,
		0, 0, 1/sqrt(svd.w.at<double>(2)));
	cv::Mat q1_ = invRootD * svd.u.t() * (cvpt2mat(q1-p,0)),
		q2_ = invRootD * svd.u.t() * (cvpt2mat(q2-p,0));
	return cv::norm(q1_.cross(q2_))/cv::norm(q1_-q2_);
}

double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Point3d& q1, const cv::Point3d& q2)
// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{	
	if (pt.U.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: R matrix must be 3x3"<<endl;
		return -1;
	}
	double out;
/*	double s0 = pt.W_sqrt[0], s1 = pt.W_sqrt[1], s2 = pt.W_sqrt[2]; 
	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);

	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;
	
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);	 
	 out = cv::norm(q1n.cross(q2n))/cv::norm(q1n-q2n);	
*/

	double xa = q1.x, ya = q1.y, za = q1.z;
	double xb = q2.x, yb = q2.y, zb = q2.z;
	double c1 = pt.DU[0], c2 = pt.DU[1], c3 = pt.DU[2],
	c4 = pt.DU[3], c5 = pt.DU[4], c6 = pt.DU[5],
	c7 = pt.DU[6], c8 = pt.DU[7], c9 = pt.DU[8]; 

	double x1 = pt.pos.x, x2 = pt.pos.y, x3 = pt.pos.z;
	double term1 = ((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c4*(x1-xb)+c5*(x2-yb)+c6*(x3-zb))
    	       -(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c1*(x1-xb)+c2*(x2-yb)+c3*(x3-zb))),
	term2 = ((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c7*(x1-xb)+c8*(x2-yb)+c9*(x3-zb))
    		-(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c1*(x1-xb)+c2*(x2-yb)+c3*(x3-zb))),
	term3 = ((c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c7*(x1-xb)+c8*(x2-yb)+c9*(x3-zb))
    		-(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c4*(x1-xb)+c5*(x2-yb)+c6*(x3-zb))),
	term4 = (c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb)),
	term5 = (c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb)) ,
	term6 = (c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));
    out = sqrt((term1 * term1 + term2* term2 + term3 * term3)/( term4*term4 + term5*term5 + term6*term6));
/*
    double x1 = pt.dux[0], x2 = pt.dux[1], x3 = pt.dux[2];
    double term1 = (-x1+c1*xa+c2*ya+c3*za)*(-x2+c4*xb+c5*yb+c6*zb)-(-x2+c4*xa+c5*ya+c6*za)*(-x1+c1*xb+c2*yb+c3*zb),
     term2 =(-x1+c1*xa+c2*ya+c3*za)*(-x3+c7*xb+c8*yb+c9*zb)-(-x3+c7*xa+c8*ya+c9*za)*(-x1+c1*xb+c2*yb+c3*zb),
     term3 = (-x2+c4*xa+c5*ya+c6*za)*(-x3+c7*xb+c8*yb+c9*zb)-(-x3+c7*xa+c8*ya+c9*za)*(-x2+c4*xb+c5*yb+c6*zb),
     term4 = c1*xa-c1*xb+c2*ya-c2*yb+c3*za-c3*zb,
     term5 = c4*xa-c4*xb+c5*ya-c5*yb+c6*za-c6*zb,
     term6 = c7*xa-c7*xb+c8*ya-c8*yb+c9*za-c9*zb;
    out = sqrt(((term1*term1)+(term2*term2)+(term3*term3))/((term4*term4)+(term5*term5)+(term6*term6)));
*/
    return out;
	
}

cv::Point3d mahvec_3d_pt_line (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the Mahalanobis distance vector between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{	
	if (pt.U.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: R matrix must be 3x3"<<endl;
		exit(0);
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;

//	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	double s0 = pt.W_sqrt[0], s1 = pt.W_sqrt[1], s2 = pt.W_sqrt[2];
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);

	double t = - q1n.dot(q2n-q1n)/((q2n-q1n).dot(q2n-q1n));

	cv::Point3d out = q1n + t * (q2n - q1n);
	
	return out;
}

cv::Point3d closest_3dpt_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the closest point using the Mahalanobis distance from a random 3d point p to line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
	if (pt.U.cols != 3)	{
		cerr<<"Error in closest_3dpt_online_mah: R matrix must be 3x3"<<endl;
		exit(0);
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;

	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);
//	double out = cv::norm(q1n.cross(q2n))/cv::norm(q1n-q2n);
	double t = - q1n.dot(q2n-q1n)/((q2n-q1n).dot(q2n-q1n));
	cv::Point3d cpt =  q1n + t * (q2n - q1n);
	cv::Point3d cpt_w(cpt.x*s0*r11+cpt.y*s1*r12+cpt.z*s2*r13+pt.pos.x,
				cpt.x*s0*r21+cpt.y*s1*r22+cpt.z*s2*r23+pt.pos.y,
				cpt.x*s0*r31+cpt.y*s1*r32+cpt.z*s2*r33+pt.pos.z);
	return cpt_w;
}

double closest_3dpt_ratio_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the closest point using the Mahalanobis distance from a random 3d point p to line (q1,q2)
// return the ratio t, such that q1 + t * (q2 - q1) is the closest point 
{
	if (pt.U.cols != 3)	{
		cerr<<"Error in closest_3dpt_ratio_online_mah: R matrix must be 3x3"<<endl;
		exit(0);
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;

	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);
	double t = - q1n.dot(q2n-q1n)/((q2n-q1n).dot(q2n-q1n));
	return t;
}

double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Mat& q1, const cv::Mat& q2)
// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
//	cout<<pt.cov<<endl;
	return mah_dist3d_pt_line(pt, mat2cvpt3d(q1), mat2cvpt3d(q2));
}

cv::Point3d mahvec_3d_pt_line(const RandomPoint3d& pt, cv::Mat q1, cv::Mat q2)
{
	return mahvec_3d_pt_line(pt, mat2cvpt3d(q1), mat2cvpt3d(q2));
}



struct Data_MLEstimateLine3d
{
	int idx1, idx2;
	vector<RandomPoint3d>& pts;
	Eigen::Matrix3d cov_inv_idx1, cov_inv_idx2;
	Data_MLEstimateLine3d(vector<RandomPoint3d>& _pts) :pts(_pts) {}

};

void costFun_MLEstimateLine3d(double *p, double *error, int m, int n, void *adata)
{
	struct Data_MLEstimateLine3d* dptr;
	dptr = (struct Data_MLEstimateLine3d *) adata;	
	int curParaIdx = 0, curErrIdx = 0;
	int idx1 = dptr->idx1, idx2 = dptr->idx2;
	cv::Point3d ap(p[0], p[1], p[2]);
	cv::Point3d bp(p[3], p[4], p[5]);
	Eigen::Vector3d ea(p[0], p[1], p[2]);
	Eigen::Vector3d eb(p[3], p[4], p[5]);
	double cost = 0;
	for(int i=0; i<dptr->pts.size(); ++i) {
		if(i==idx1) {
		Eigen::Vector3d pti(dptr->pts[i].pos.x,dptr->pts[i].pos.y,dptr->pts[i].pos.z);
		error[i] = (ea - pti).transpose() * dptr->cov_inv_idx1 * (ea - pti);		
		} else if(i==idx2) {
			Eigen::Vector3d pti(dptr->pts[i].pos.x,dptr->pts[i].pos.y,dptr->pts[i].pos.z);
			error[i] = (eb - pti).transpose() * dptr->cov_inv_idx2 * (eb - pti);
		} else {			
			error[i] = mah_dist3d_pt_line (dptr->pts[i], ap, bp); // 2 times slower than plain dist mahvec_3d_pt_line
		}
	//	cost += error[i]*error[i];
	}
//	cout<<cost<<", ";
}

void MLEstimateLine3d (RandomLine3d& line,	int maxIter)
	// optimally estimate a 3d line from a set of collinear random 3d points
	// 3d line is represented by two points
{
	
	// ----- preprocessing: find 2 extremities of points along the line direction -----
	double minv=100, maxv=-100;
	int	   idx_end1 = 0, idx_end2 = 0;
	for(int i=0; i<line.pts.size(); ++i) {
		double dproduct = (line.pts[i].pos-line.A).dot(line.A-line.B);
		if ( dproduct < minv) {
			minv = dproduct;
			idx_end1 = i;
		}
		if (dproduct > maxv) {
			maxv = dproduct;
			idx_end2 = i;
		}
	}	
	if(idx_end1 > idx_end2) swap(idx_end1, idx_end2); // ensure idx_end1 < idx_end2
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;

	// ----- optimization parameters -----
	Data_MLEstimateLine3d data(line.pts);
	data.idx1 = idx_end1;
	data.idx2 = idx_end2;
    cv::Mat inf1 = line.pts[idx_end1].cov.inv();
    cv::Mat inf2 = line.pts[idx_end2].cov.inv();
    for(int i=0; i<3; ++i) {
    	for(int j=0; j<3; ++j) {
    		data.cov_inv_idx1(i,j) = inf1.at<double>(i,j);
    		data.cov_inv_idx2(i,j) = inf2.at<double>(i,j);
    	}
    }

	vector<double> paraVec, measVec;
	paraVec.reserve(line.pts.size()+4); measVec.reserve(line.pts.size());
	for(int i = 0; i < line.pts.size(); ++i) {	
			measVec.push_back(0);
		if (i == idx_end1 || i == idx_end2) {
			paraVec.push_back(line.pts[i].pos.x);
			paraVec.push_back(line.pts[i].pos.y);
			paraVec.push_back(line.pts[i].pos.z);			
		} 
	}
	int numPara = paraVec.size(); int numMeas = measVec.size();
	double* para = paraVec.data(); double* meas = measVec.data();

	// ----- start LM solver -----
	double* lm_cov = new double[numPara*numPara];
	int nit = dlevmar_dif(costFun_MLEstimateLine3d, para, meas, numPara, numMeas, maxIter, opts, info, NULL, NULL, (void*)&data);
	
	// ------ update line endpoints ------
	line.A = cv::Point3d (para[0],para[1],para[2]);
	line.B = cv::Point3d (para[3],para[4],para[5]);
	
	// ------ compute line uncertainty [fast version] ------
	cv::Mat H = MleLine3dCov(line.pts, idx_end1, idx_end2, para);
	cv::Mat cov = H.inv();
	line.covA = cov.rowRange(0,3).colRange(0,3);
	line.covB = cov.rowRange(3,6).colRange(3,6);
	line.rndA = RandomPoint3d(line.A, line.covA);
	line.rndB = RandomPoint3d(line.B, line.covB);
	line.pts.clear();
	return;

	// xxxxx compute cov of MLE result [slow, out-dated] xxxxx
/*	cv::Mat cov_meas_inv = cv::Mat::zeros(3*line.pts.size(), 3*line.pts.size(), CV_64F);
	cv::Mat tmp;
	double* p = new double[line.pts.size()+4]; // mimic the full/long parameter vector
	int idx = 0;
	for(int i=0; i<line.pts.size();++i) {
		tmp = line.pts[i].cov.inv();
		tmp.copyTo(cov_meas_inv.rowRange(i*3,i*3+3).colRange(i*3,i*3+3));
		if(i==idx_end1) {
			p[idx] = para[0];
			p[idx+1] = para[1];
			p[idx+2] = para[2];
			idx = idx + 3;
		} else if(i==idx_end2){
			p[idx] = para[3];
			p[idx+1] = para[4];
			p[idx+2] = para[5];
			idx = idx + 3;
		}else {
			// project pt to line to get ratio
			p[idx] = 1- closest_3dpt_ratio_online_mah (line.pts[i], cv::Point3d(para[0],para[1],para[2]), cv::Point3d(para[3],para[4],para[5]));			
			++idx;
		}		
	}
	// ---- compute line endpoint uncerainty ----	
	MlestimateLine3dCov (p, line.pts.size()+4, idx_end1, idx_end2, cov_meas_inv, line.covA, line.covB); // most expensive part
	// refine line endpoint positions
	line.A = cv::Point3d (p[idx_end1],p[idx_end1+1],p[idx_end1+2]);
	line.B = cv::Point3d (p[idx_end2+2],p[idx_end2+3],p[idx_end2+4]);
	line.rndA = RandomPoint3d(line.A, line.covA);
	line.rndB = RandomPoint3d(line.B, line.covB);	
*/	
}

cv::Mat jac_rpt2ln_mahvec_wrt_ln(const RandomPoint3d& pt, const double l[6]) 
{
	double xa = l[0], ya = l[1], za = l[2],
		   xb = l[3], yb = l[4], zb = l[5],
		   x1 = pt.pos.x, x2 = pt.pos.y, x3 = pt.pos.z,
		   c1 = pt.DU[0], c2 = pt.DU[1], c3 = pt.DU[2],
		   c4 = pt.DU[3], c5 = pt.DU[4], c6 = pt.DU[5],
		   c7 = pt.DU[6], c8 = pt.DU[7], c9 = pt.DU[8];
	cv::Mat jac(3,6,CV_64F);
	jac.at<double>(0,0) = c1-(c1*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*(c1*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c4*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c7*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb));
  	jac.at<double>(0,1) = c2-(c2*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*(c2*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c5*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c8*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb));
  	jac.at<double>(0,2) = c3-(c3*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*(c3*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c6*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c9*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb));
  	jac.at<double>(0,3) = ((c1*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c4*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c7*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c1*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb));
  	jac.at<double>(0,4) = ((c2*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c5*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c8*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c2*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb));
  	jac.at<double>(0,5) = ((c3*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c6*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c9*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c3*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb));
  	jac.at<double>(1,0) = c4-(c4*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*(c1*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c4*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c7*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb));
  	jac.at<double>(1,1) = c5-(c5*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*(c2*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c5*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c8*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb));
  	jac.at<double>(1,2) = c6-(c6*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*(c3*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c6*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c9*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb));
  	jac.at<double>(1,3) = ((c1*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c4*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c7*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c4*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb));
  	jac.at<double>(1,4) = ((c2*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c5*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c8*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c5*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb));
  	jac.at<double>(1,5) = ((c3*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c6*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c9*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c6*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb));
  	jac.at<double>(2,0) = c7-(c7*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*(c1*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c4*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c7*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));
  	jac.at<double>(2,1) = c8-(c8*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*(c2*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c5*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c8*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));
  	jac.at<double>(2,2) = c9-(c9*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-((c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*(c3*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c6*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c9*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))+c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));
  	jac.at<double>(2,3) = ((c1*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c4*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c7*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c7*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c1*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c4*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c7*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));
  	jac.at<double>(2,4) = ((c2*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c5*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c8*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c8*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c2*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c5*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c8*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));
  	jac.at<double>(2,5) = ((c3*(c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))+c6*(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))+c9*(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za)))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))+(c9*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))))/(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0))-1.0/pow(pow(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb),2.0)+pow(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb),2.0)+pow(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb),2.0),2.0)*((c1*(x1-xa)+c2*(x2-ya)+c3*(x3-za))*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))+(c4*(x1-xa)+c5*(x2-ya)+c6*(x3-za))*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))+(c7*(x1-xa)+c8*(x2-ya)+c9*(x3-za))*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb)))*(c3*(c1*(x1-xa)-c1*(x1-xb)+c2*(x2-ya)-c2*(x2-yb)+c3*(x3-za)-c3*(x3-zb))*2.0+c6*(c4*(x1-xa)-c4*(x1-xb)+c5*(x2-ya)-c5*(x2-yb)+c6*(x3-za)-c6*(x3-zb))*2.0+c9*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb))*2.0)*(c7*(x1-xa)-c7*(x1-xb)+c8*(x2-ya)-c8*(x2-yb)+c9*(x3-za)-c9*(x3-zb));

	return jac;
}


cv::Mat jac_pt2pt_mahvec_wrt_pt (const RandomPoint3d& pt, const double A[3]) 
{
	double xa = A[0], ya = A[1], za = A[2],
		   x1 = pt.pos.x, x2 = pt.pos.y, x3 = pt.pos.z,
		   c1 = pt.DU[0], c2 = pt.DU[1], c3 = pt.DU[2],
		   c4 = pt.DU[3], c5 = pt.DU[4], c6 = pt.DU[5],
		   c7 = pt.DU[6], c8 = pt.DU[7], c9 = pt.DU[8];
	cv::Mat jac(3,3,CV_64F);
	jac.at<double>(0,0) = -c1;
	jac.at<double>(0,1) = -c2;
	jac.at<double>(0,2) = -c3;
	jac.at<double>(1,0) = -c4;
	jac.at<double>(1,1) = -c5;
	jac.at<double>(1,2) = -c6;
	jac.at<double>(2,0) = -c7;
	jac.at<double>(2,1) = -c8;
	jac.at<double>(2,2) = -c9;
	return jac;
}

cv::Mat MleLine3dCov(const vector<RandomPoint3d>& pts, int idx1, int idx2, const double l[6])
{
	cv::Mat J = cv::Mat::zeros(3*pts.size(), 6, CV_64F);
	for(int i=0; i<pts.size(); ++i) {
		if(i == idx1) {
			cv::Mat jac = jac_pt2pt_mahvec_wrt_pt(pts[i], &l[0]);
			cv::Mat tmp = J(cv::Rect(0, i*3,3,3));
			jac.copyTo(tmp);
		} else if(i==idx2) {
			cv::Mat jac = jac_pt2pt_mahvec_wrt_pt(pts[i], &l[3]);
			cv::Mat tmp = J(cv::Rect(3, i*3,3,3));
			jac.copyTo(tmp);
		} else {
			cv::Mat tmp = J(cv::Rect(0, i*3,6,3));
			cv::Mat jac = jac_rpt2ln_mahvec_wrt_ln(pts[i],l);
			jac.copyTo(tmp);
		}
	}
	cv::Mat H = J.t()*J;		

	return H;
}

void MlestimateLine3dCov (double* p, const int n, int i1, int i2, const cv::Mat& cov_meas_inv,
	cv::Mat& cov_i1, cv::Mat& cov_i2)
	// compute the covariance of the MLE of 3d line
	// input: p - MLE parameter vector 
	//		  n - dimension of p
	//		  i1,i2 - index of points used to represent 3d line (the i1-th and i2-th point of all points)
	//		  cov_meas - covariance of measurement, i.e. all collinear points 
	// output: cov_i1 - covariane of point i1
{	
	// compute jacobian
	cv::Mat J = cv::Mat::zeros(3*(n-4), n, CV_64F);	
	J.at<double>(3*i1,i1) = 1;
	J.at<double>(3*i1+1,i1+1) = 1;
	J.at<double>(3*i1+2,i1+2) = 1;
	J.at<double>(3*i2,i2+2) = 1;
	J.at<double>(3*i2+1,i2+3) = 1;
	J.at<double>(3*i2+2,i2+4) = 1;
	for(int i=0; i<n-4; ++i) {
		if(i==i1 || i==i2) continue;
		int col =-1;
		if (i < i1) {
			col = i;
		}
		if (i > i1 && i < i2) {
			col = i + 2;
		}
		if (i > i2) {
			col = i + 4;
		}
		J.at<double>(3*i, i1) = p[col];
		J.at<double>(3*i+1, i1+1) = p[col];
		J.at<double>(3*i+2, i1+2) = p[col];
		J.at<double>(3*i, col) = p[i1] - p[i2+2];
		J.at<double>(3*i+1,col) = p[i1+1] - p[i2+3];
		J.at<double>(3*i+2,col) = p[i1+2] - p[i2+4];
		J.at<double>(3*i,i2+2) = 1-p[col];
		J.at<double>(3*i+1,i2+3) = 1-p[col];
		J.at<double>(3*i+2,i2+4) = 1-p[col];
	}

	arma::mat Jt(reinterpret_cast<double*>(J.data), J.cols, J.rows);
	arma::mat cmit(reinterpret_cast<double*>(cov_meas_inv.data), cov_meas_inv.rows, cov_meas_inv.cols); 
	arma::mat cp;

	if(n > 25) { // magic number...
    	arma::mat Jtc = arma::zeros(n,3*(n-4));// =Jt*cmit
    	for(int i=0; i<(n-4); ++i) {
    		Jtc(arma::span::all,arma::span(i*3,i*3+2)) 
    		= Jt(arma::span::all,arma::span(i*3,i*3+2)) * cmit(arma::span(i*3,i*3+2),arma::span(i*3,i*3+2));    
    	}
    	cp = (Jtc * Jt.t()).i();
	} else {
		cp = (Jt*cmit*Jt.t()).i();  // this is slower due to large matrix
	}

	cov_i1.create(3,3,CV_64F); cov_i2.create(3,3,CV_64F);
	for(int i=0; i<3; ++i) {
		for(int j=0; j<3; ++j) {
			cov_i1.at<double>(i,j) = cp(i1+i,i1+j);
			cov_i2.at<double>(i,j) = cp(i2+2+i,i2+2+j);
		}
	}

}



void termReason(int info)
{
	switch(info) {
	case 1:
		{cout<<"Termination reason 1: stopped by small gradient J^T e."<<endl;break;}
	case 2:
		{cout<<"Termination reason 2: stopped by small Dp."<<endl;break;}
	case 3:
		{cout<<"Termination reason 3: stopped by itmax."<<endl;break;}
	case 4:
		{cout<<"Termination reason 4: singular matrix. Restart from current p with increased mu."<<endl;break;}
	case 5:
		{cout<<"Termination reason 5: no further error reduction is possible. Restart with increased mu."<<endl;break;}
	case 6:
		{cout<<"Termination reason 6: stopped by small ||e||_2."<<endl;break;}
	case 7:
		{cout<<"Termination reason 7: stopped by invalid (i.e. NaN or Inf) 'func' values; a user error."<<endl;break;}
	default:
		{cout<<"Termination reason: Unknown..."<<endl;}
	}
}

double pt_to_line_dist2d(const cv::Point2d& p, const double l[3])
	// distance from point(x,y) to line ax+by+c=0;
	// l=(a, b, c), p = (x,y)
{
	if(abs(l[0]*l[0]+l[1]*l[1]-1)>EPS) {
		cout<<"pt_to_line_dist2d: error,l should be normalized/initialized\n";
		exit(0);
	}
	double a = l[0],
		b = l[1],
		c = l[2],
		x = p.x,
		y = p.y;
	return abs((a*x+b*y+c))/sqrt(a*a+b*b);
}
double line_to_line_dist2d(const FrameLine& a, const FrameLine& b)
	// compute line to line distance by averaging 4 endpoint to line distances
	// a and b must be almost parallel to make any sense
{
	return 0.25*pt_to_line_dist2d(a.p,b.lineEq2d)
		+ 0.25*pt_to_line_dist2d(a.q, b.lineEq2d)
		+ 0.25*pt_to_line_dist2d(b.p, a.lineEq2d)
		+ 0.25*pt_to_line_dist2d(b.q, a.lineEq2d);
}

void trackLine (const vector<FrameLine>& f1, const vector<FrameLine>& f2, vector<vector<int> >& matches)
	// line segment tracking
	// input: 
	// finishing in <10 ms
{

	double lineDistThresh  = 25; // pixel
	double lineAngleThresh = 25 * PI/180; // 30 degree
	double desDiffThresh   = 0.85;
	double lineOverlapThresh = 3; // pixels
	double ratio_dist_1st2nd = 0.7;


	if(sysPara.fast_motion) {
		lineDistThresh = 45;
		lineAngleThresh = 30 * PI/180;
		desDiffThresh   = 0.85;
		lineOverlapThresh = -1;
	}

	if(sysPara.dark_ligthing) {
		ratio_dist_1st2nd = 0.85;
		desDiffThresh = 1.5;
		lineDistThresh = 20;
		lineAngleThresh = 10 * PI/180;

	}

	cv::Mat desDiff = cv::Mat::zeros(f1.size(), f2.size(), CV_64F)+100;
	for(int i=0; i<f1.size(); ++i) {		
		for(int j=0; j<f2.size(); ++j) {
			if((f1[i].r.dot(f2[j].r) > cos(lineAngleThresh)) && // angle between gradients
				(line_to_line_dist2d(f1[i],f2[j]) < lineDistThresh) &&
				(lineSegmentOverlap(f1[i],f2[j]) > lineOverlapThresh )) // line (parallel) distance
			{
				desDiff.at<double>(i,j) = cv::norm(f1[i].des - f2[j].des);
			}
		}
	}
	
	for(int i=0; i<desDiff.rows; ++i) {
		vector<int> onePairIdx;
		double minVal;
		cv::Point minPos;
		cv::minMaxLoc(desDiff.row(i),&minVal,NULL,&minPos,NULL);
		if (minVal < desDiffThresh) {
			double minV;
			cv::Point minP;
			cv::minMaxLoc(desDiff.col(minPos.x),&minV,NULL,&minP,NULL);			
			if (i==minP.y) {    // commnent this for more potential matches 
				//further check distance ratio
				double rowmin2 = 100, colmin2 = 100;
				for(int j=0; j<desDiff.cols; ++j) {
					if (j == minPos.x) continue;
					if (rowmin2 > desDiff.at<double>(i,j)) {
						rowmin2 = desDiff.at<double>(i,j);
					}
				}
				for(int j=0; j<desDiff.rows; ++j) {
					if (j == minP.y) continue;
					if (colmin2 > desDiff.at<double>(j,minPos.x)) {
						colmin2 = desDiff.at<double>(j,minPos.x);
					}
				}
				if(rowmin2*ratio_dist_1st2nd > minVal && colmin2*ratio_dist_1st2nd > minVal) {
					onePairIdx.push_back(i);
					onePairIdx.push_back(minPos.x);
					matches.push_back(onePairIdx);
				}
			}
		}
	}	
}

void matchLine (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches)
	// line segment matching (for loop closure)
	// input: 
	// finishing in <10 ms
{

	double lineDistThresh  = 80; // pixel
	double lineAngleThresh = 25 * PI/180; // 30 degree
	double desDiffThresh   = 0.7;
	double lineOverlapThresh = -1; // pixels
	double ratio_dist_1st2nd = 0.7;

	cv::Mat desDiff = cv::Mat::zeros(f1.size(), f2.size(), CV_64F)+100;
	for(int i=0; i<f1.size(); ++i) {		
		for(int j=0; j<f2.size(); ++j) {
			if((f1[i].r.dot(f2[j].r) > cos(lineAngleThresh)) && // angle between gradients
				(line_to_line_dist2d(f1[i],f2[j]) < lineDistThresh) &&
				(lineSegmentOverlap(f1[i],f2[j]) > lineOverlapThresh )) // line (parallel) distance
			{
				desDiff.at<double>(i,j) = cv::norm(f1[i].des - f2[j].des);
			}
		}
	}
	
	for(int i=0; i<desDiff.rows; ++i) {
		vector<int> onePairIdx;
		double minVal;
		cv::Point minPos;
		cv::minMaxLoc(desDiff.row(i),&minVal,NULL,&minPos,NULL);
		if (minVal < desDiffThresh) {
			double minV;
			cv::Point minP;
			cv::minMaxLoc(desDiff.col(minPos.x),&minV,NULL,&minP,NULL);			
			if (i==minP.y) {    // commnent this for more potential matches 
				//further check distance ratio
				double rowmin2 = 100, colmin2 = 100;
				for(int j=0; j<desDiff.cols; ++j) {
					if (j == minPos.x) continue;
					if (rowmin2 > desDiff.at<double>(i,j)) {
						rowmin2 = desDiff.at<double>(i,j);
					}
				}
				for(int j=0; j<desDiff.rows; ++j) {
					if (j == minP.y) continue;
					if (colmin2 > desDiff.at<double>(j,minPos.x)) {
						colmin2 = desDiff.at<double>(j,minPos.x);
					}
				}
				if(rowmin2*ratio_dist_1st2nd > minVal && colmin2*ratio_dist_1st2nd > minVal) {
					onePairIdx.push_back(i);
					onePairIdx.push_back(minPos.x);
					matches.push_back(onePairIdx);
				}
			}
		}
	}	
}


int computeSubPSR (cv::Mat* xGradient, cv::Mat* yGradient,
	cv::Point2d p, double s, cv::Point2d g, vector<double>& vs) {
		/* input: p - 2D point position
		s - side length of square region
		g - unit vector of gradient of line 
		output: vs = (v1, v2, v3, v4)
		*/ 
		double tl_x = floor(p.x - s/2), tl_y = floor(p.y - s/2);
		if (tl_x < 0 || tl_y < 0 || 
			tl_x+s+1 > xGradient->cols || tl_y+s+1 > xGradient->rows)
			return 0; // out of image
		double v1=0, v2=0, v3=0, v4=0; 
		for (int y = tl_y; y < tl_y+s; ++y) {			
			for (int x  = tl_x; x < tl_x+s; ++x) {
				double tmp1 = 
					xGradient->at<double>(y,x)*g.x + yGradient->at<double>(y,x)*g.y;
				double tmp2 = 
					xGradient->at<double>(y,x)*(-g.y) + yGradient->at<double>(y,x)*g.x;
				if ( tmp1 >= 0 )
					v1 = v1 + tmp1;
				else
					v2 = v2 - tmp1;
				if (tmp2 >= 0)
					v3 = v3 + tmp2;
				else
					v4 = v4 - tmp2;
			}
		}
		vs.resize(4);
		vs[0] = v1; vs[1] = v2;
		vs[2] = v3; vs[3] = v4;
		return 1;
}

int computeMSLD (FrameLine& l, cv::Mat* xGradient, cv::Mat* yGradient) 
	// compute msld and gradient
{	 
	cv::Point2d gradient = l.getGradient(xGradient, yGradient);	
	l.r = gradient;
	int s = 5 * xGradient->cols/800.0;
	double len = cv::norm(l.p-l.q);

	vector<vector<double> > GDM; //GDM.reserve(2*(int)len);
	double step = sysPara.msld_sample_interval; // the step length between sample points on line segment
	for (int i=0; i*step < len; ++i) {
		vector<double> col; col.reserve(36);
		cv::Point2d pt =    // compute point position on the line
			l.p + (l.q - l.p) * (i*step/len);		
		bool fail = false;
		for (int j=-4; j <= 4; ++j ) { // 9 PSR for each point on line
			vector<double> psr(4);
			if (computeSubPSR (xGradient, yGradient, pt+j*s*gradient, s, gradient, psr)) {
				col.push_back(psr[0]);
				col.push_back(psr[1]);
				col.push_back(psr[2]);
				col.push_back(psr[3]);
			} else
				fail = true;
		}
		if (fail)
			continue;
		GDM.push_back(col);
	}
//	tm.end(); cout<<"psr "<<tm.time_ms<<'\t'; tm.start();
	cv::Mat MS(72, 1, CV_64F);
	if (GDM.size() ==0 ) {
		for (int i=0; i<MS.rows; ++i)
			MS.at<double>(i,0) = rand(); // if not computable, assign random num
		l.des = MS;
		return 0;
	}

	double gauss[9] = { 0.24142,0.30046,0.35127,0.38579,0.39804,
		0.38579,0.35127,0.30046,0.24142};
	for (int i=0; i < 36; ++i) {
		double sum=0, sum2=0, mean, std;
		for (int j=0; j < GDM.size(); ++j) {
			GDM[j][i] = GDM[j][i] * gauss[i/4];
			sum += GDM[j][i];
			sum2 += GDM[j][i]*GDM[j][i];
		}
		mean = sum/GDM.size();
		std = sqrt(sum2/GDM.size() - mean*mean);
		MS.at<double>(i,0)		= mean;
		MS.at<double>(i+36, 0)	= std;
	}
	// normalize mean and std vector, respcectively
	MS.rowRange(0,36) = MS.rowRange(0,36) / cv::norm(MS.rowRange(0,36));
	MS.rowRange(36,72) = MS.rowRange(36,72) / cv::norm(MS.rowRange(36,72));
	for (int i=0; i < MS.rows; ++i) {
		if (MS.at<double>(i,0) > 0.4)
			MS.at<double>(i,0) = 0.4;
	}
	MS = MS/cv::norm(MS);
	l.des.create(72, 1, CV_64F);
	l.des = MS;

//	tm.end(); cout<<"rest "<<tm.time_ms<<'\n';
	return 1;
}


int computeSubPSR (double* xGradient, double* yGradient,
	cv::Point2d p, double s, cv::Point2d g, int width, int height, vector<double>& vs) {
		/* input: p - 2D point position
		s - side length of square region
		g - unit vector of gradient of line 
		output: vs = (v1, v2, v3, v4)
		*/ 
		double tl_x = floor(p.x - s/2), tl_y = floor(p.y - s/2);
		if (tl_x < 0 || tl_y < 0 || 
			tl_x+s+1 > width || tl_y+s+1 > height)
			return 0; // out of image
		double v1=0, v2=0, v3=0, v4=0; 
		for (int y = tl_y; y < tl_y+s; ++y) {			
			for (int x  = tl_x; x < tl_x+s; ++x) {
				double tmp1 = 
					xGradient[y*width+x]*g.x + yGradient[y*width+x]*g.y;
				double tmp2 = 
					xGradient[y*width+x]*(-g.y) + yGradient[y*width+x]*g.x;
				if ( tmp1 >= 0 )
					v1 = v1 + tmp1;
				else
					v2 = v2 - tmp1;
				if (tmp2 >= 0)
					v3 = v3 + tmp2;
				else
					v4 = v4 - tmp2;
			}
		}
		vs.resize(4);
		vs[0] = v1; vs[1] = v2;
		vs[2] = v3; vs[3] = v4;
		return 1;
}

int computeMSLD (FrameLine& l, double* xGradient, double* yGradient, int width, int height) 
	// compute msld and gradient
{	 
	int s = 5 * width/800.0;
	double len = cv::norm(l.p-l.q);

	vector<vector<double> > GDM; GDM.reserve(2*(int)len);
	double step = sysPara.msld_sample_interval; // the step length between sample points on line segment
//	MyTimer tm; tm.start();
	for (int i=0; i*step < len; ++i) {
		vector<double> col; col.reserve(36);
		cv::Point2d pt =    // compute point position on the line
			l.p + (l.q - l.p) * (i*step/len);		
		bool fail = false;
		for (int j=-4; j <= 4; ++j ) { // 9 PSR for each point on line
			vector<double> psr(4);
			if (computeSubPSR (xGradient, yGradient, pt+j*s*l.r, s, l.r, width, height, psr)) {		
				col.push_back(psr[0]);
				col.push_back(psr[1]);
				col.push_back(psr[2]);
				col.push_back(psr[3]);
			} else {
				fail = true;
				break;
			}
		}
		if (fail)
			continue;
		GDM.push_back(col);
	}
//	tm.end(); cout<<"psr "<<tm.time_ms<<'\t'; tm.start();
	cv::Mat MS(72, 1, CV_64F);
	if (GDM.size() ==0 ) {
		for (int i=0; i<MS.rows; ++i)
			MS.at<double>(i,0) = rand(); // if not computable, assign random num
		l.des = MS;
		return 0;
	}

	double gauss[9] = { 0.24142,0.30046,0.35127,0.38579,0.39804,
		0.38579,0.35127,0.30046,0.24142};
	for (int i=0; i < 36; ++i) {
		double sum=0, sum2=0, mean, std;
		for (int j=0; j < GDM.size(); ++j) {
			GDM[j][i] = GDM[j][i] * gauss[i/4];
			sum += GDM[j][i];
			sum2 += GDM[j][i]*GDM[j][i];
		}
		mean = sum/GDM.size();
		std = sqrt(sum2/GDM.size() - mean*mean);
		MS.at<double>(i,0)		= mean;
		MS.at<double>(i+36, 0)	= std;
	}
	// normalize mean and std vector, respcectively
	MS.rowRange(0,36) = MS.rowRange(0,36) / cv::norm(MS.rowRange(0,36));
	MS.rowRange(36,72) = MS.rowRange(36,72) / cv::norm(MS.rowRange(36,72));
	for (int i=0; i < MS.rows; ++i) {
		if (MS.at<double>(i,0) > 0.4)
			MS.at<double>(i,0) = 0.4;
	}
	MS = MS/cv::norm(MS);
	l.des.create(72, 1, CV_64F);
	l.des = MS;

//	tm.end(); cout<<"rest "<<tm.time_ms<<'\n';
	return 1;
}

double projectPt2d_to_line2d(const cv::Point2d& X, const cv::Point2d& A, const cv::Point2d& B) 
{
	// X' = lambda*A + (1-lambda)*B; X' is the projection of X on AB
	cv::Point2d BX = X-B, BA = A-B;
	double lambda = BX.dot(BA)/cv::norm(BA)/cv::norm(BA);
	return lambda;
}

double lineSegmentOverlap(const FrameLine& a, const FrameLine& b)
	// compute the overlap length of two line segments in their parallel direction
{
	if(cv::norm(a.p-a.q) < cv::norm(b.p-b.q)) {// a is shorter than b
		double lambda_p = projectPt2d_to_line2d(a.p, b.p, b.q);
		double lambda_q = projectPt2d_to_line2d(a.q, b.p, b.q);
		if( (lambda_p < 0 && lambda_q < 0) || (lambda_p > 1 && lambda_q > 1) )
			return -1;
		else
			return abs(lambda_p - lambda_q) * cv::norm(b.p-b.q);
	} else {
		double lambda_p = projectPt2d_to_line2d(b.p, a.p, a.q);
		double lambda_q = projectPt2d_to_line2d(b.q, a.p, a.q);
		if( (lambda_p < 0 && lambda_q < 0) || (lambda_p > 1 && lambda_q > 1) )
			return -1;
		else
			return abs(lambda_p - lambda_q) * cv::norm(a.p-a.q);
	}
}


cv::Mat vec2SkewMat (cv::Mat vec)
{
	cv::Mat m = (cv::Mat_<double>(3,3) <<
		0, -vec.at<double>(2), vec.at<double>(1),
		vec.at<double>(2), 0, -vec.at<double>(0),
		-vec.at<double>(1), vec.at<double>(0), 0);
	return m;
}
cv::Mat vec2SkewMat (cv::Point3d vec)
{
	cv::Mat m = (cv::Mat_<double>(3,3) <<
		0, -vec.z, vec.y,
		vec.z, 0, -vec.x,
		-vec.y, vec.x, 0);
	return m;
}


cv::Mat q2r (cv::Mat q)
// input: unit quaternion representing rotation
// output: 3x3 rotation matrix
// note: q=(a,b,c,d)=a + b i + c j + d k, where (b,c,d) is the rotation axis
{
	double a = q.at<double>(0),	b = q.at<double>(1),
		c = q.at<double>(2), d = q.at<double>(3);
	double nm = sqrt(a*a+b*b+c*c+d*d);
		a = a/nm;
		b = b/nm;
		c = c/nm;
		d = d/nm;
	cv::Mat R = (cv::Mat_<double>(3,3)<< 
		a*a+b*b-c*c-d*d,	2*b*c-2*a*d,		2*b*d+2*a*c,
		2*b*c+2*a*d,		a*a-b*b+c*c-d*d,	2*c*d-2*a*b,
		2*b*d-2*a*c,		2*c*d+2*a*b,		a*a-b*b-c*c+d*d);
	return R.clone();
}
cv::Mat q2r (double* q)
	// input: unit quaternion representing rotation
	// output: 3x3 rotation matrix
	// note: q=(a,b,c,d)=a + b i + c j + d k, where (b,c,d) is the rotation axis
{
	double  a = q[0],	b = q[1],
			c = q[2],	d = q[3];
	double nm = sqrt(a*a+b*b+c*c+d*d);	
		a = a/nm;
		b = b/nm;
		c = c/nm;
		d = d/nm;
	cv::Mat R = (cv::Mat_<double>(3,3)<< 
		a*a+b*b-c*c-d*d,	2*b*c-2*a*d,		2*b*d+2*a*c,
		2*b*c+2*a*d,		a*a-b*b+c*c-d*d,	2*c*d-2*a*b,
		2*b*d-2*a*c,		2*c*d+2*a*b,		a*a-b*b-c*c+d*d);
	return R.clone();
}

cv::Mat r2q(cv::Mat R)
{	
	double t = R.at<double>(0,0)+R.at<double>(1,1)+R.at<double>(2,2);
	double r = sqrt(1+t);
	double s = 0.5/r;
	double w = 0.5*r; 
	double x = (R.at<double>(2,1)-R.at<double>(1,2))*s;
	double y = (R.at<double>(0,2)-R.at<double>(2,0))*s;
	double z = (R.at<double>(1,0)-R.at<double>(0,1))*s;
	cv::Mat q = (cv::Mat_<double>(4,1)<<w,x,y,z);
	return q;
}

Eigen::Vector4d r2q(Eigen::Matrix3d R)
{	
	double t = R(0,0)+R(1,1)+R(2,2);
	double r = sqrt(1+t);
	double s = 0.5/r;
	double w = 0.5*r; 
	double x = (R(2,1)-R(1,2))*s;
	double y = (R(0,2)-R(2,0))*s;
	double z = (R(1,0)-R(0,1))*s;
	Eigen::Vector4d q (w,x,y,z);
	return q;
}

void write2file (Map3d& m, string suffix)
{
	
#ifdef _WIN32	
	string prefix = m.datapath.substr(51, m.datapath.size()-51-1);
	string fname = "./src/"+prefix+"_line"+suffix+".txt", 
		fname2 = "./src/"+prefix+"_lmk"+suffix+".txt";
#else
	string prefix = m.datapath.substr(59, m.datapath.size()-59-1);
	string fname = prefix+"_line"+suffix+".txt", 
		fname2 = prefix+"_lmk"+suffix+".txt";	
#endif		
	ofstream f(fname.c_str()), f2(fname2.c_str());
	f.precision(16);
	f.precision(16);
	// === Note 1 ===
	// The output format is [tx ty tz qx qy qz qw], following that of the RGBD-SLAM dataset groundtruth, as explained below.
	// Letting t = [tx ty tz]', q = [qw qx qy qz], and R = q2r(q),
	// then (R,t) transform a point from current-camera coordsys to world coordsys, i.e. X_w = R*X_c + t;
	// So, the (R,t) pair here is DIFFERENT (actually, inverse) to conventional camera projection matrix
	// === Note 2 ===
	// The output quaternion format is different than what used in this c++ program. 
	for(int i=0; i<m.keyframeIdx.size();++i){
		cv::Mat t = -m.frames[m.keyframeIdx[i]].R.t()*m.frames[m.keyframeIdx[i]].t;
		cv::Mat q = r2q(m.frames[m.keyframeIdx[i]].R.t());
		f<<m.frames[m.keyframeIdx[i]].timestamp<<'\t'
		 <<t.at<double>(0)<<'\t'<<t.at<double>(1)<<'\t'<<t.at<double>(2)<<'\t'
		 <<q.at<double>(1)<<'\t'<<q.at<double>(2)<<'\t'<<q.at<double>(3)<<'\t'<<q.at<double>(0)<<endl;
	}
	f.close();
	// 
	for(int i=0; i<m.lmklines.size(); ++i) {
		f2<<m.lmklines[i].A.x<<'\t'<<m.lmklines[i].A.y<<'\t'<<m.lmklines[i].A.z<<'\t'
		  <<m.lmklines[i].B.x<<'\t'<<m.lmklines[i].B.y<<'\t'<<m.lmklines[i].B.z<<'\n';
	}
	f2.close();

}

void write_linepairs_tofile(vector<RandomLine3d> a, vector<RandomLine3d> b, string fname, double timestamp)
{
	ofstream file(fname.c_str());
	file.precision(16);
	for(int i=0; i<a.size(); ++i) {		
		file<<a[i].A.x<<'\t'<<a[i].A.y<<'\t'<<a[i].A.z<<'\t'
			<<a[i].B.x<<'\t'<<a[i].B.y<<'\t'<<a[i].B.z<<'\t'
			<<b[i].A.x<<'\t'<<b[i].A.y<<'\t'<<b[i].A.z<<'\t'
			<<b[i].B.x<<'\t'<<b[i].B.y<<'\t'<<b[i].B.z<<'\t'<<timestamp<<endl;
	}
	file.close();
}

double pesudoHuber(double e, double band)
// pesudo Huber cost function
// e : error (distance)
// ouput: bounded e*e
{
	return 2*band*band*(sqrt(1+(e/band)*(e/band))-1);
}

double rotAngle (cv::Mat R)
{
	return acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2));
}

double ave_img_bright(cv::Mat img)
// (grayscale) image brightness average value
{
	cv::Mat gray;
	if(img.channels()>1)
	{
		cv::cvtColor(img, gray, CV_BGR2GRAY);
	} else {
		gray = img.clone();
	}
	cv::Scalar m = cv::mean(gray);
	return m.val[0];
}

bool get_pt_3d (cv::Point2d p2, cv::Point3d& p3, const cv::Mat& depth)
{
	if(p2.x<0 || p2.y<0 || p2.x >= depth.cols || p2.y >= depth.rows ) 
		return false;
	int row, col; // nearest pixel for pt
	if((floor(p2.x) == p2.x) && (floor(p2.y) == p2.y)) {// boundary issue
		col = max(int(p2.x-1),0);
		row = max(int(p2.y-1),0);
	} else {
		col = int(p2.x);
		row = int(p2.y);
	}

	if(depth.at<double>(row,col) < EPS) { // no depth info
		return false;
	} else {
		double zval = depth.at<double>(row,col)/5000; // in meter, z-value
		cv::Point2d tmp = mat2cvpt(K.inv()*cvpt2mat(p2))*zval;	
		p3.x = tmp.x;
		p3.y = tmp.y;
		p3.z = zval;
		return true;
	}
}

/// Function prototype for DetectEdgesByED exported by EDLinesLib.a
LS *DetectLinesByED(unsigned char *srcImg, int width, int height, int *pNoLines);

LS* callEDLines (const cv::Mat& im_uchar, int* numLines)
{
	
	if (im_uchar.type() != CV_8UC1) 
	{
		cerr<<"error: callEDLines requires uchar image type\n";
		exit(0);
	}

	//// image data copy 
	uchar* p = new uchar[im_uchar.rows*im_uchar.cols];	
    uchar* srcImg = p;	
    for(int i=0; i<im_uchar.rows; ++i) {
    	const uchar* pr = im_uchar.ptr<uchar>(i);
      	for(int j=0; j<im_uchar.cols; ++j) {
	   		*(p++) =  pr[j];
	   	}
	}

	LS* ls = DetectLinesByED(srcImg, im_uchar.cols, im_uchar.rows, numLines);

	delete[] srcImg;
	
	return ls;
}


double rotateAngleDeg(cv::Mat R) // return degree
{
	return acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2))
			* 180/PI;
}

cv::Mat triangulatePoint (const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& K,
	cv::Point2d pt1, cv::Point2d pt2)
{
	CvMat cvP1 = P1, cvP2 = P2; // projection matrix
	cv::Mat X(4,1,CV_64F);
	cv::Mat x1 = cvpt2mat(mat2cvpt(K.inv()*cvpt2mat(pt1)),0),
		x2 = cvpt2mat(mat2cvpt(K.inv()*cvpt2mat(pt2)),0);
	CvMat cx1 = x1, cx2 = x2, cX = X; 
	cvTriangulatePoints(&cvP1, &cvP2, &cx1, &cx2, &cX);
	return (cv::Mat_<double>(3,1)<<cX.data.db[0]/cX.data.db[3], 
		cX.data.db[1]/cX.data.db[3], cX.data.db[2]/cX.data.db[3]);
}
