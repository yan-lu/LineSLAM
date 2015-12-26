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

#include "utils.h"
using namespace Eigen;


int essn_ransac (cv::Mat* pts1, cv::Mat* pts2, cv::Mat* E, cv::Mat K, 
	cv::Mat* inlierMask, int imsize)
	// running 5-point algorithm 
	// re-estimation at the end
	// Input: points pairs normalized by camera matrix K
	// output: 
{
	cv::Mat bestMss1, bestMss2;
	int maxIterNum = 500, N = pts1->cols;
	double confidence = 0.995;
	double threshDist = 1;// (0.5*imsize/640.0);		//  in image units 0.64
	int iter = 0;
	vector<int> rnd;
	for (int i=0; i<N; ++i)
		rnd.push_back(i);
	cv::Mat mss1(3,5,CV_64F), mss2(3,5,CV_64F);
	vector<int> maxInlierSet;
	cv::Mat bestE;
	while (iter < maxIterNum) {
		++iter;
		// --- choose minimal solution set: mss1<->mss2
		random_shuffle(rnd.begin(),rnd.end());
		for (int i=0; i < mss1.cols; ++i) {
			pts1->col(rnd[i]).copyTo(mss1.col(i));
			pts2->col(rnd[i]).copyTo(mss2.col(i));
			// DEGENERATE COFIGURATION ????????????????????s
		}		
		// compute minimal solution by 5-point
		vector<cv::Mat> Es;   // multiple results from 5point alg.
		fivepoint_stewnister(mss1, mss2, Es);
		// find concensus set
		vector<int> curInlierSet;

		for (int k=0; k < Es.size(); ++k) {
			curInlierSet.clear();
			cv::Mat F = K.t().inv() * Es[k] * K.inv();
			
			for (int i=0; i<N; ++i) {
				double dist = sqrt(fund_samperr(K*pts1->col(i), K*pts2->col(i),F));
				if (dist < threshDist)
					curInlierSet.push_back(i);
			}
			if(curInlierSet.size() > maxInlierSet.size()) {
				maxInlierSet = curInlierSet;
				bestE = Es[k];
			}
		}
		//re-compute maximum iteration number:maxIterNum
//		maxIterNum = abs(log(1-confidence)
//			/log(1-pow(double(maxInlierSet.size())/N,5.0)));
	}
	if (maxInlierSet.size() < mss1.cols) {
		//cout<< <<endl;
		cout<<"essn_ransac: Largest concensus set is too small for minimal estimation."
			<<endl;
		return 0;
	}

	cv::Mat tmpE = bestE.clone();
	while (true) {		
		vector<cv::Mat> Es;
		vector<int> maxInlierSet_k;
		cv::Mat bestE_k;		
		// re-estimate
		cv::Mat inliers1(3,maxInlierSet.size(),CV_64F),
			inliers2(3,maxInlierSet.size(),CV_64F);
		for (int i=0; i<maxInlierSet.size(); ++i) {
			pts1->col(maxInlierSet[i]).copyTo(inliers1.col(i));
			pts2->col(maxInlierSet[i]).copyTo(inliers2.col(i));
		}				
		//opt_essn_pts (inliers1, inliers2, &tmpE);
		optimizeEmat (inliers1, inliers2, K, &tmpE);
		vector<int> curInlierSet;
		for (int i=0; i<N; ++i) {
			double dist = sqrt(fund_samperr(K*pts1->col(i), K*pts2->col(i),
				K.t().inv() * tmpE * K.inv()));
			if (dist < threshDist)
				curInlierSet.push_back(i);
		}
		if(curInlierSet.size() > maxInlierSet.size())  {
			maxInlierSet = curInlierSet;
			bestE = tmpE;
		} else
			break;
	}

	cv::Mat inliers1(3,maxInlierSet.size(),CV_64F),
		inliers2(3,maxInlierSet.size(),CV_64F);
	for (int i=0; i<maxInlierSet.size(); ++i) {
		pts1->col(maxInlierSet[i]).copyTo(inliers1.col(i));
		pts2->col(maxInlierSet[i]).copyTo(inliers2.col(i));
	}	
	//	opt_essn_pts (inliers1, inliers2, &bestE);
	inlierMask->create(N,1,CV_8U);
	*inlierMask = *inlierMask*0;
	for (int i=0; i<maxInlierSet.size(); ++i)
		inlierMask->at<uchar>(maxInlierSet[i]) = 1; 
	*E = bestE;

	return 1;
}

void essn_ransac (cv::Mat* pts1, cv::Mat* pts2, vector<cv::Mat>& bestEs, cv::Mat K, 
					vector<cv::Mat>& inlierMasks, int imsize, bool usePrior, cv::Mat t_prior)
	// running 5-point algorithm 
	// re-estimation at the end
	// Input: points pairs normalized by camera matrix K
	// output: 

{
	bestEs.clear();
	inlierMasks.clear();
	int maxSize;
	int trialNum = 5;
	int maxIterNum = 500, maxIterNum0 = 500;
	double confidence = 0.999;
	double threshDist = 1;		//  in image units 

	vector<int> numInlier_E; // num of inliers for each E 
	vector<double> resds;
	for (int trial = 0; trial < trialNum;  ++trial) {		
		cv::Mat bestMss1, bestMss2;
		int N = pts1->cols;		
		cv::Mat Kpts1 = K*(*pts1);
		cv::Mat Kpts2 = K*(*pts2);
		vector<int> rnd;
		for (int i=0; i<N; ++i)	rnd.push_back(i);
		cv::Mat mss1(3,5,CV_64F), mss2(3,5,CV_64F);	
		vector<int> maxInlierSet;
		cv::Mat bestE;
		double best_resd;
		int iter = 0;

		while (iter < maxIterNum) {
			++iter;
			// --- choose minimal solution set: mss1<->mss2
			random_shuffle(rnd.begin(),rnd.end());
			for (int i=0; i<mss1.cols; ++i) {
				pts1->col(rnd[i]).copyTo(mss1.col(i));
				pts2->col(rnd[i]).copyTo(mss2.col(i));
				// DEGENERATE COFIGURATION ????????????????????s
			}		
			// compute minimal solution by 5-point
			vector<cv::Mat> Es;   // multiple results from 5point alg.		
			fivepoint_stewnister(mss1, mss2, Es);

			// find concensus set
			vector<int> curInlierSet;
			for (int k=0; k < Es.size(); ++k) {
				curInlierSet.clear();
				bool consitent_with_prior = true;
				if(usePrior) { //use prior translation vector to filter 
					cv::Mat Ra, Rb, t1;
					decEssential (&Es[k], &Ra, &Rb, &t1);
					double angleThresh = 40; // degree
					if (abs(t1.dot(t_prior)/cv::norm(t1)/cv::norm(t_prior)) < cos(angleThresh*PI/180))
						consitent_with_prior = false;				
				}
				if(!consitent_with_prior)	continue;

				// check if already found
				bool existE = false;
				for(int kk=0; kk < bestEs.size(); ++ kk) {
					if(isEssnMatSimilar(bestEs[kk], Es[k])) {
						existE = true;
						break;
					}
				}
				if(existE) continue;
				double resd = 0;
				cv::Mat F = K.t().inv() * Es[k] * K.inv();
				for (int i=0; i<N; ++i) {				
					double dist_sq = fund_samperr(Kpts1.col(i), Kpts2.col(i),F);		
					if (dist_sq < threshDist * threshDist) {
						curInlierSet.push_back(i);
						resd += dist_sq;
					}
				}
				resd = resd/curInlierSet.size();
				if(curInlierSet.size() > maxInlierSet.size()) {
					maxInlierSet = curInlierSet;
					bestE = Es[k];
					best_resd = resd;
				}
			}
			//re-compute maximum iteration number:maxIterNum
			if(maxInlierSet.size()>0) {
			maxIterNum = min(maxIterNum0, 
				(int)abs(log(1-confidence)/log(1-pow(double(maxInlierSet.size())/N,5.0))));
			}
		}
		if (maxInlierSet.size()<mss1.cols) {
			continue;
		}
		
		cv::Mat tmpE = bestE.clone();	
		while (true) {		
			vector<cv::Mat> Es;
			vector<int> maxInlierSet_k;
			cv::Mat bestE_k;				
			// re-estimate by optimization
			cv::Mat inliers1(3,maxInlierSet.size(),CV_64F),
				inliers2(3,maxInlierSet.size(),CV_64F);
			for (int i=0; i<maxInlierSet.size(); ++i) {
				pts1->col(maxInlierSet[i]).copyTo(inliers1.col(i));
				pts2->col(maxInlierSet[i]).copyTo(inliers2.col(i));
			}				
		//	optimize_E_g2o(inliers1, inliers2, K, &tmpE);
			optimizeEmat (inliers1, inliers2, K, &tmpE);			
			vector<int> curInlierSet;
			double resd = 0;
			cv::Mat F = K.t().inv() * tmpE * K.inv();
			for (int i=0; i<N; ++i) {
				double dist_sq = fund_samperr(Kpts1.col(i), Kpts2.col(i), F);
				if (dist_sq < threshDist * threshDist) {
					curInlierSet.push_back(i);
					resd += dist_sq;
				}
			}
			resd = resd/curInlierSet.size();
			if(curInlierSet.size() > maxInlierSet.size())  {
				maxInlierSet = curInlierSet;
				bestE = tmpE;
				best_resd = resd;				
			} else
				break;
		}
		cv::Mat inliers1(3,maxInlierSet.size(),CV_64F),
			inliers2(3,maxInlierSet.size(),CV_64F);
		for (int i=0; i<maxInlierSet.size(); ++i) {
			pts1->col(maxInlierSet[i]).copyTo(inliers1.col(i));
			pts2->col(maxInlierSet[i]).copyTo(inliers2.col(i));
		}	
	
		cv::Mat inlierMask = cv::Mat::zeros(N,1,CV_8U);
		for (int i=0; i<maxInlierSet.size(); ++i)
			inlierMask.at<uchar>(maxInlierSet[i]) = 1; 

		if(bestEs.empty()) {
			bestEs.push_back(bestE);
			maxSize = maxInlierSet.size();
			inlierMasks.push_back(inlierMask);
			numInlier_E.push_back(maxInlierSet.size());
			resds.push_back(best_resd);
		} else {
			bool existE = false;
				for(int k=0; k < bestEs.size(); ++ k) {
					if(isEssnMatSimilar(bestEs[k], bestE)) {
						existE = true;
						break;
					}
				}
			if (!existE && maxInlierSet.size() > 0.9 * maxSize 
				&& abs((int)maxInlierSet.size()-maxSize) <= 15) {
				bestEs.push_back(bestE);
				maxSize = max((int)maxInlierSet.size(), maxSize);
				inlierMasks.push_back(inlierMask);
				numInlier_E.push_back(maxInlierSet.size());
				resds.push_back(best_resd);
			}
		}
		if(bestEs.size() >= 4) break;

		cv::Mat Ra, Rb, t;
		decEssential (&bestEs.back(), &Ra, &Rb, &t);
		if(rotateAngleDeg(Ra) < 3 && rotateAngleDeg(Rb) < 3) {
			trialNum = 2; 
		//	cout<<rotateAngleDeg(Ra)<<'\t'<<rotateAngleDeg(Rb)<<endl;
		}
}

		for(int i=0; i<bestEs.size(); ++i) {
		cv::Mat Ra, Rb, t;
		decEssential (&bestEs[i], &Ra, &Rb, &t);
//		cout<<t;
//		cout<<numInlier_E[i]<<", res="<<resds[i]<<endl;
		}
	}
	
	
bool isEssnMatSimilar(cv::Mat E1, cv::Mat E2)
{
	cv::Mat Ra, Rb, t1, t2;
	decEssential (&E1, &Ra, &Rb, &t1);
	decEssential (&E2, &Ra, &Rb, &t2);
	//compare Es by compare ts
	double angleThresh = 20; // degree
	if (abs(t1.dot(t2)) < cos(angleThresh*PI/180))
		return false;
	else // better check Rs also
	{
		return true;
	}
}

double fund_samperr (cv::Mat x1, cv::Mat x2, cv::Mat F) 
	// sampson error for fundmental matrix F between two image points x1, x2 from
	// I1 and I2, respectively
	// Result unit is sum of squred distance
{/*
	cv::Mat fx1 = F*x1, fx2 = F.t()*x2;	
	double d = (x2.t()*F*x1).dot(x2.t()*F*x1)
		/(fx1.at<double>(0)*fx1.at<double>(0)+
		  fx1.at<double>(1)*fx1.at<double>(1)+
		  fx2.at<double>(0)*fx2.at<double>(0)+
		  fx2.at<double>(1)*fx2.at<double>(1));
*/
	// faster version, same value as above
	double x10 = x1.at<double>(0), x11 = x1.at<double>(1), x12, 
		   x20 = x2.at<double>(0), x21 = x2.at<double>(1), x22,
		   f00 = F.at<double>(0,0),f01 = F.at<double>(0,1),f02 = F.at<double>(0,2),
		   f10 = F.at<double>(1,0),f11 = F.at<double>(1,1),f12 = F.at<double>(1,2),
		   f20 = F.at<double>(2,0),f21 = F.at<double>(2,1),f22 = F.at<double>(2,2);
	if(x1.cols*x1.rows == 3) {
		x12 = x1.at<double>(2);
		x22 = x2.at<double>(2);
	} else {
		x12 = 1;
		x22 = 1;
	}
	double	d = (x10*(f00*x20+f10*x21+f20*x22)+x11*(f01*x20+f11*x21+f21*x22)+x12*(f02*x20+f12*x21+f22*x22))*(x10*(f00*x20+f10*x21+f20*x22)+x11*(f01*x20+f11*x21+f21*x22)+x12*(f02*x20+f12*x21+f22*x22))/
		((f00*x10+f01*x11+f02*x12)*(f00*x10+f01*x11+f02*x12) + (f10*x10+f11*x11+f12*x12)*(f10*x10+f11*x11+f12*x12)+
		(f00*x20+f10*x21+f20*x22)*(f00*x20+f10*x21+f20*x22) + (f01*x20+f11*x21+f21*x22)*(f01*x20+f11*x21+f21*x22));
	return d;
}

float fund_samperr_float (cv::Mat x1, cv::Mat x2, cv::Mat F) 
	// sampson error for fundmental matrix F between two image points x1, x2 from
	// I1 and I2, respectively
	// Result unit is sum of squred distance
{	
	float x10 = x1.at<float>(0), x11 = x1.at<float>(1), x12, 
		   x20 = x2.at<float>(0), x21 = x2.at<float>(1), x22;
	float  f00 = F.at<double>(0,0),f01 = F.at<double>(0,1),f02 = F.at<double>(0,2),
		   f10 = F.at<double>(1,0),f11 = F.at<double>(1,1),f12 = F.at<double>(1,2),
		   f20 = F.at<double>(2,0),f21 = F.at<double>(2,1),f22 = F.at<double>(2,2);
	if(x1.cols*x1.rows == 3) {
		x12 = x1.at<float>(2);
		x22 = x2.at<float>(2);
	} else {
		x12 = 1;
		x22 = 1;
	}
	float	d = (x10*(f00*x20+f10*x21+f20*x22)+x11*(f01*x20+f11*x21+f21*x22)+x12*(f02*x20+f12*x21+f22*x22))*(x10*(f00*x20+f10*x21+f20*x22)+x11*(f01*x20+f11*x21+f21*x22)+x12*(f02*x20+f12*x21+f22*x22))/
		((f00*x10+f01*x11+f02*x12)*(f00*x10+f01*x11+f02*x12) + (f10*x10+f11*x11+f12*x12)*(f10*x10+f11*x11+f12*x12)+
		(f00*x20+f10*x21+f20*x22)*(f00*x20+f10*x21+f20*x22) + (f01*x20+f11*x21+f21*x22)*(f01*x20+f11*x21+f21*x22));
	return d;
}

struct data_essn_pts 
{
	cv::Mat p1;
	cv::Mat p2;
};
void costfun_essn_pts (double *p, double *error, int m, int n, void *adata)
{
	struct data_essn_pts* dptr = (struct data_essn_pts *) adata;

	Quaterniond q( p[0], p[1], p[2], p[3]);
	q.normalize();
	cv::Mat R = (cv::Mat_<double>(3,3) 
		<< q.matrix()(0,0), q.matrix()(0,1), q.matrix()(0,2),
		q.matrix()(1,0), q.matrix()(1,1), q.matrix()(1,2),
		q.matrix()(2,0), q.matrix()(2,1), q.matrix()(2,2));

	double t_norm = sqrt(p[4]*p[4] + p[5]*p[5] + p[6]*p[6]); 
	p[4] = p[4]/t_norm;
	p[5] = p[5]/t_norm;
	p[6] = p[6]/t_norm;
	cv::Mat tx = (cv::Mat_<double>(3,3)<< 0, -p[6], p[5],
		p[6], 0 ,  -p[4],
		-p[5], p[4], 0 );
	cv::Mat E = tx * R;
	double cost=0;	
	for (int i=0; i < n; ++i) {		
		error[i] = sqrt(fund_samperr (dptr->p1.col(i), dptr->p2.col(i), E));
		cost = cost+error[i]*error[i];
	}
	//	cout<<cost<<"\t";
	//		<<"t=["<<-tx.at<double>(1,2)<<","<<tx.at<double>(0,2)<<","<< 
	//		-tx.at<double>(0,1)<<"]"<<endl;
}
void opt_essn_pts (cv::Mat p1, cv::Mat p2, cv::Mat *E)
	// input: p1, p2, normalized image points correspondences
{
	int n = p1.cols;
	double* measurement = new double[n];	
	for (int i=0; i<n; ++i) 
		measurement[i] = 0;	
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU*0.5; //
	opts[1]=1E-15;
	opts[2]=1E-50; // original 1e-50
	opts[3]=1E-20;
	opts[4]= -LM_DIFF_DELTA;
	int matIter = 100;

	cv::Mat R1, R2, t;
	decEssential (E, &R1, &R2, &t);
	cv::Mat Rt = // find true R and t
		findTrueRt(R1,R2,t,mat2cvpt(p1.col(0)),mat2cvpt(p2.col(0)));
	cv::Mat R =  Rt.colRange(0,3);
	t = Rt.col(3);

	Matrix3d Rx;
	Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
		R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
		R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	Quaterniond q(Rx);

	double para[7] = {q.w(), q.x(), q.y(), q.z(), t.at<double>(0),
		t.at<double>(1), t.at<double>(2)};

	data_essn_pts data;
	data.p1 = p1;
	data.p2 = p2;

	int ret = dlevmar_dif(costfun_essn_pts, para, measurement, 7, n,
		matIter, opts, info, NULL, NULL, (void*)&data);
	delete[] measurement;
	q.w() = para[0];
	q.x() = para[1];
	q.y() = para[2];
	q.z() = para[3];
	q.normalize();
	for (int i=0; i<3; ++i)
		for (int j=0; j<3; ++j)
			R.at<double>(i,j) = q.matrix()(i,j);
	cv::Mat tx = (cv::Mat_<double>(3,3)<< 0, -para[6], para[5],
		para[6], 0 ,  -para[4],
		-para[5], para[4], 0 );
	tx = tx/sqrt(para[4]*para[4]+para[5]*para[5]+para[6]*para[6]);
	*E = tx * R;
	/*	cout<<"optimal t=["<<-tx.at<double>(1,2)<<","<<tx.at<double>(0,2)<<","<< 
	-tx.at<double>(0,1)<<"]"<<endl;

	switch(int(info[6])) {
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
	*/
}

struct data_optimizeEmat
{
	cv::Mat p1;  // normalized points (by K)
	cv::Mat p2;
	cv::Mat K;
};
void costfun_optimizeEmat (double *p, double *error, int m, int n, void *adata)
{
	struct data_optimizeEmat* dptr = (struct data_optimizeEmat *) adata;
	cv::Mat K = dptr->K;

	Quaterniond q( p[0], p[1], p[2], p[3]);
	q.normalize();
	cv::Mat R = (cv::Mat_<double>(3,3) 
		<< q.matrix()(0,0), q.matrix()(0,1), q.matrix()(0,2),
		q.matrix()(1,0), q.matrix()(1,1), q.matrix()(1,2),
		q.matrix()(2,0), q.matrix()(2,1), q.matrix()(2,2));

	double t_norm = sqrt(p[4]*p[4] + p[5]*p[5] + p[6]*p[6]); 
	p[4] = p[4]/t_norm;
	p[5] = p[5]/t_norm;
	p[6] = p[6]/t_norm;
	cv::Mat tx = (cv::Mat_<double>(3,3)<< 0, -p[6], p[5],
										p[6],  0 , -p[4],
									   -p[5], p[4], 0 );	
	cv::Mat E = tx * R;
	cv::Mat F = K.t().inv() * E * K.inv();
	double cost=0;	
	cv::Mat Kp1 = K*dptr->p1, Kp2 = K * dptr->p2;
	for (int i=0; i < n; ++i) {		
		error[i] = sqrt(fund_samperr (Kp1.col(i), Kp2.col(i), F));
//		cost = cost+error[i]*error[i];
	}
//	cout<<cost<<"\t";
}
void optimizeEmat (cv::Mat p1, cv::Mat p2, cv::Mat K, cv::Mat *E)
	// input: p1, p2, normalized image points correspondences
{
	int n = p1.cols;
	double* measurement = new double[n];	
	for (int i=0; i<n; ++i) 
		measurement[i] = 0;	
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU*0.5; //
	opts[1]=1E-50;
	opts[2]=1E-100; // original 1e-50
	opts[3]=1E-20;
	opts[4]= -LM_DIFF_DELTA;
	int matIter = 100;
	
	cv::Mat R1, R2, t;
	
	decEssential (E, &R1, &R2, &t);
	
	cv::Mat Rt = // find true R and t
		findTrueRt(R1,R2,t,mat2cvpt(p1.col(0)),mat2cvpt(p2.col(0)));
	
	if(Rt.cols < 3) return;

	cv::Mat R =  Rt.colRange(0,3);
	t = Rt.col(3);
	Matrix3d Rx;
	Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
		R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
		R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	Quaterniond q(Rx);

	double para[7] = {q.w(), q.x(), q.y(), q.z(), t.at<double>(0),
		t.at<double>(1), t.at<double>(2)};

	data_optimizeEmat data;
	data.p1 = p1;
	data.p2 = p2;
	data.K  = K;
	
	int ret = dlevmar_dif(costfun_optimizeEmat, para, measurement, 7, n,
		matIter, opts, info, NULL, NULL, (void*)&data);
	delete[] measurement;
	q.w() = para[0];
	q.x() = para[1];
	q.y() = para[2];
	q.z() = para[3];
	q.normalize();
	
	for (int i=0; i<3; ++i)
		for (int j=0; j<3; ++j)
			R.at<double>(i,j) = q.matrix()(i,j);

	cv::Mat tx = (cv::Mat_<double>(3,3)<< 0, -para[6], para[5],
		para[6], 0 ,  -para[4],
		-para[5], para[4], 0 );
	tx = tx/sqrt(para[4]*para[4]+para[5]*para[5]+para[6]*para[6]);
	*E = tx * R;
//	cout<<" "<<R<<endl<<para[4]<<"\t"<<para[5]<<"\t"<<para[6]<<endl;
}

void decEssential (cv::Mat *E, cv::Mat *R1, cv::Mat *R2, cv::Mat *t) 
	// decompose essential matrix E into R1, R2 and t.
	// following the method in HZ's book.
{
	if (cv::determinant(*E)<0)	{
		//	cout << cv::determinant(*E)<<endl;
		*E = -*E;
	}
	
	for (int i=0; i<2; ++i) {
		
		cv::SVD svd(*E);
		
		cv::Mat W = (cv::Mat_<double>(3,3)<< 0,-1,0,1,0,0,0,0,1),
			Z = (cv::Mat_<double>(3,3)<< 0,1,0,-1,0,0,0,0,0);
		R1->create(3,3,CV_64F);
		R2->create(3,3,CV_64F);
		t->create(3,1,CV_64F);

		*R1 = svd.u*W*svd.vt;

		*R2 = svd.u*W.t()*svd.vt;
		
		*t = svd.u.col(2);
		// To ensure a proper rotation, check determinant >0
		if (cv::determinant(*R1)<0 || cv::determinant(*R2)<0) {
			*E = -*E;
			//			cout<<"...oh, det(R)<0..."<<endl;
		}
		else
			break;
	}
	if (cv::determinant(*R1)<0 && cv::determinant(*R2)<0) {
		*R1 = -*R1;
		*R2 = -*R2;
	}
	return;
}

cv::Mat linearTriangulate (cv::Mat P1,cv::Mat P2,cv::Point2d q1,cv::Point2d q2)
{
	cv::Mat L(4,4,CV_64F);
	cv::Mat vec;
	vec = q1.x*P1.row(2)-P1.row(0);
	vec.copyTo(L.row(0));
	vec = q1.y*P1.row(2)-P1.row(1);
	vec.copyTo(L.row(1));
	vec = q2.x*P2.row(2)-P2.row(0);
	vec.copyTo(L.row(2));
	vec = q2.y*P2.row(2)-P2.row(1);
	vec.copyTo(L.row(3));
	cv::SVD svd(L);
	return svd.vt.t().col(3);
}

bool checkCheirality (cv::Mat P, cv::Mat X) // cheirality constraint
	//	input: Project Matrix P 4x3, P = [R | t], Homogeneous 3D point X 4x1 
	//  output: true - if X is in front of camera
{
	if(X.cols * X.rows < 4) {
		X = (cv::Mat_<double>(4,1)<<X.at<double>(0),X.at<double>(1),X.at<double>(2),1);
	}
	cv::Mat x = P*X;
	return cv::determinant(P.colRange(0,3))*x.at<double>(2)*X.at<double>(3)>0;
}

bool checkCheirality (cv::Mat R, cv::Mat t, cv::Mat X) // cheirality point constraint
	//	input: Project Matrix P 4x3, P = [R | t], Homogeneous 3D point X 4x1 
	//  output: true - if X is in front of camera
{
	if(X.cols * X.rows < 4) {
		X = (cv::Mat_<double>(4,1)<<X.at<double>(0),X.at<double>(1),X.at<double>(2),1);
	}
	cv::Mat P(3,4,CV_64F);
	R.copyTo(P.colRange(0,3)); 	 t.copyTo(P.col(3));
	cv::Mat x = P*X;
	return cv::determinant(P.colRange(0,3))*x.at<double>(2)*X.at<double>(3) > 0;
}


cv::Mat 
	findTrueRt(cv::Mat R1,cv::Mat R2, cv::Mat t,cv::Point2d q1,cv::Point2d q2) 
	// find the true R and t from the decomposition result of Essential matrix
	// q1 and q2 must be normalized points in image coordinates
{
	cv::Mat t2 = -t, P2;
	cv::Mat P1 = (cv::Mat_<double>(3,4)<<1,0,0,0,
		0,1,0,0,
		0,0,1,0);
	cv::Mat Pa(3,4,CV_64F),Pb(3,4,CV_64F),Pc(3,4,CV_64F),Pd(3,4,CV_64F);
	R1.copyTo(Pa.colRange(0,3)); 	 t.copyTo(Pa.col(3));
	R1.copyTo(Pb.colRange(0,3));	 t2.copyTo(Pb.col(3));
	R2.copyTo(Pc.colRange(0,3));	 t.copyTo(Pc.col(3));
	R2.copyTo(Pd.colRange(0,3));	 t2.copyTo(Pd.col(3));

	cv::Mat tp;
//	tp = linearTriangulate (P1,Pa,q1,q2);
	tp = triangulatePoint (P1, Pa, cv::Mat::eye(3,3,CV_64F), q1, q2);
	if (checkCheirality(P1,tp) && checkCheirality(Pa,tp))
		P2 = Pa;

//	tp = linearTriangulate (P1,Pb,q1,q2);
	tp = triangulatePoint (P1, Pb, cv::Mat::eye(3,3,CV_64F), q1, q2);
	if (checkCheirality(P1,tp) && checkCheirality(Pb,tp))
		P2 = Pb;

//	tp = linearTriangulate (P1,Pc,q1,q2);
	tp = triangulatePoint (P1, Pc, cv::Mat::eye(3,3,CV_64F), q1, q2);
	if (checkCheirality(P1,tp) && checkCheirality(Pc,tp))
		P2 = Pc;

//	tp = linearTriangulate (P1,Pd,q1,q2);
	tp = triangulatePoint (P1, Pd, cv::Mat::eye(3,3,CV_64F), q1, q2);
	if (checkCheirality(P1,tp) && checkCheirality(Pd,tp))
		P2 = Pd;

	return P2;
}


bool fund_ransac (cv::Mat pts1, cv::Mat pts2, cv::Mat F, vector<uchar>& mask, double distThresh, double confidence)
	// running 8-point algorithm 
	// re-estimation at the end
	// Input: points pairs normalized by camera matrix K
	// output: 
{
	int maxIterNum = 500, N = pts1.cols;
	int iter = 0;
	vector<int> rnd;
	for (int i=0; i<N; ++i)		rnd.push_back(i);
	cv::Mat mss1(2,8,CV_32F), mss2(2,8,CV_32F);
	vector<int> maxInlierSet;
	cv::Mat bestF;
	vector<uchar> maxInlierMask(N);

	while (iter < maxIterNum) {
		++iter;
		// --- choose minimal solution set: mss1<->mss2
//		random_shuffle(rnd.begin(),rnd.end());
//		random_unique(rnd.begin(), rnd.end(), mss1.cols);
		for (int i=0; i < mss1.cols; ++i) {
			pts1.col((i*iter)%N).copyTo(mss1.col(i));
			pts2.col((i*iter)%N).copyTo(mss2.col(i));
		}		
		// compute minimal solution by 8-point
		cv::Mat minF = cv::findFundamentalMat(mss1.t(),mss2.t(), cv::FM_8POINT);
		// find concensus set
		vector<int> curInlierSet;
		vector<uchar> inlierMask(N);
			for (int i=0; i<N; ++i) {
				float dist_sq = fund_samperr_float(pts1.col(i), pts2.col(i), minF);
				if (dist_sq < distThresh*distThresh) {
					curInlierSet.push_back(i);
					inlierMask[i] = 1; 
				} else {
					inlierMask[i] = 0;
				}
			}
			if(curInlierSet.size() > maxInlierSet.size()) {
				maxInlierSet = curInlierSet;
				bestF = minF;
				maxInlierMask = inlierMask;
			}

		//re-compute maximum iteration number:maxIterNum
		maxIterNum = abs(log(1-confidence)/log(1-pow(double(maxInlierSet.size())/N,8.0)));
	}
	if (maxInlierSet.size() < mss1.cols) {
		return false;
	}
	mask = maxInlierMask;
	F = bestF;
	return true;
}
