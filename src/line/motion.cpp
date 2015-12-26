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
#include <Eigen/Core>
#include <Eigen/Geometry>
 #include <Eigen/StdVector> // for vector of Eigen::Vector4f
#include <pcl/common/transformation_from_correspondences.h>
#include "../misc.h"
#include "../transformation_estimation.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/types/slam3d/se3quat.h"
#include "g2o/types/slam3d/edge_se3_pointxyz_depth.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif


extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;
#define OPT_USE_MAHDIST
#define MOTION_USE_MAHDIST

struct Data_optimizeRelmotion
{
	vector<RandomLine3d>& a;
	vector<RandomLine3d>& b;
	Data_optimizeRelmotion(vector<RandomLine3d>&ina, vector<RandomLine3d>& inb):a(ina),b(inb){}
};
void costFun_optimizeRelmotion(double *p, double *error, int m, int n, void *adata)
{
//	MyTimer t; t.start();
	struct Data_optimizeRelmotion* dptr;
	dptr = (struct Data_optimizeRelmotion *) adata;
	cv::Mat R = q2r(p);
	cv::Mat t = cv::Mat(3,1,CV_64F, &p[4]);// (cv::Mat_<double>(3,1)<<p[4],p[5],p[6]);
	double cost = 0;
	
	for(int i=0; i<dptr->a.size(); ++i)	{
#ifdef OPT_USE_MAHDIST
	/*	error[i] = 0.25*(mah_dist3d_pt_line(dptr->b[i].rndA, R*cvpt2mat(dptr->a[i].A,0)+t, R*cvpt2mat(dptr->a[i].B,0)+t)+
				   mah_dist3d_pt_line(dptr->b[i].rndB, R*cvpt2mat(dptr->a[i].A,0)+t, R*cvpt2mat(dptr->a[i].B,0)+t)+
				   mah_dist3d_pt_line(dptr->a[i].rndA, R.t()*(cvpt2mat(dptr->b[i].A,0)-t), R.t()*(cvpt2mat(dptr->b[i].B,0)-t))+
				   mah_dist3d_pt_line(dptr->a[i].rndB, R.t()*(cvpt2mat(dptr->b[i].A,0)-t), R.t()*(cvpt2mat(dptr->b[i].B,0)-t)));
	*/
		// faster computing than above
		double aiA[3] = {dptr->a[i].A.x,dptr->a[i].A.y,dptr->a[i].A.z},
			aiB[3] = {dptr->a[i].B.x,dptr->a[i].B.y,dptr->a[i].B.z},
			biA[3] = {dptr->b[i].A.x,dptr->b[i].A.y,dptr->b[i].A.z},
			biB[3] = {dptr->b[i].B.x,dptr->b[i].B.y,dptr->b[i].B.z};
		error[i] = 0.25*(mah_dist3d_pt_line(dptr->b[i].rndA, R*array2mat(aiA,3)+t, R*array2mat(aiB,3)+t)+
				   mah_dist3d_pt_line(dptr->b[i].rndB, R*array2mat(aiA,3)+t, R*array2mat(aiB,3)+t)+
				   mah_dist3d_pt_line(dptr->a[i].rndA, R.t()*(array2mat(biA,3)-t), R.t()*(array2mat(biB,3)-t))+
				   mah_dist3d_pt_line(dptr->a[i].rndB, R.t()*(array2mat(biA,3)-t), R.t()*(array2mat(biB,3)-t)));
	
#else
		error[i]= 0.25*(dist3d_pt_line(dptr->b[i].A, mat2cvpt3d(R*cvpt2mat(dptr->a[i].A,0)+t), mat2cvpt3d(R*cvpt2mat(dptr->a[i].B,0)+t))
				+ dist3d_pt_line(dptr->b[i].B, mat2cvpt3d(R*cvpt2mat(dptr->a[i].A,0)+t), mat2cvpt3d(R*cvpt2mat(dptr->a[i].B,0)+t))
				+ dist3d_pt_line(dptr->a[i].A, mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].A,0)-t)), mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].B,0)-t)))
				+ dist3d_pt_line(dptr->a[i].B, mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].A,0)-t)), mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].B,0)-t))));
#endif
		cost += error[i]*error[i];
	}
//	cout<<cost<<'\t';

}

void optimizeRelmotion(vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
{	
	cv::Mat q = r2q(R);
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;
	int maxIter = 50;

	// ----- optimization parameters -----
	int numPara = 7;
	double* para = new double[numPara];
	para[0] = q.at<double>(0);
	para[1] = q.at<double>(1);
	para[2] = q.at<double>(2);
	para[3] = q.at<double>(3);
	para[4] = t.at<double>(0);
	para[5] = t.at<double>(1);
	para[6] = t.at<double>(2);
	
	// ----- measurements -----
	int numMeas = a.size();
	double* meas = new double[numMeas];
	for(int i=0; i<numMeas; ++i) meas[i] = 0;

	Data_optimizeRelmotion data(a,b);
	// ----- start LM solver -----
	
//	MyTimer timer; 	timer.start();
	int ret = dlevmar_dif(costFun_optimizeRelmotion, para, meas, numPara, numMeas,
							maxIter, opts, info, NULL, NULL, (void*)&data);
//	timer.end();	cout<<"optimizeRelmotion Time used: "<<timer.time_ms<<" ms. "<<endl;
	
	R = q2r(para);
	t = (cv::Mat_<double>(3,1)<<para[4],para[5],para[6]);
	delete[] meas;
	delete[] para;

}


void optimize_motion_g2o(vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
{
	g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();	  
	optimizer->setVerbose(0);

 	g2o::BlockSolverX::LinearSolverType * linearSolver = new g2o::LinearSolverCholmod<g2o ::BlockSolverX::PoseMatrixType>();  
 	g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
	
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);   
	optimizer->setAlgorithm(solver);
 // -- add the parameter representing the sensor offset  !!!!
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(0);
	optimizer->addParameter(sensorOffset);

  	g2o::VertexSE3 *vc2 = new g2o::VertexSE3();
    {// set up rotation and translation for the a's node as identity
      Eigen::Quaterniond q(1,0,0,0);
      Eigen::Vector3d t(0,0,0);
      q.setIdentity();
      g2o::SE3Quat cam2(q,t);

      vc2->setEstimate(cam2);
      vc2->setId(1); 
      vc2->setFixed(true);
      optimizer->addVertex(vc2);
    }

    Eigen::Matrix4f tf;
    tf << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
          R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
          R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2),
          0, 0, 0, 1;

    g2o::VertexSE3 *vc1 = new g2o::VertexSE3();
    {
      Eigen::Quaterniond q(tf.topLeftCorner<3,3>().cast<double>());//initialize rotation from estimate 
      Eigen::Vector3d t(tf.topRightCorner<3,1>().cast<double>());  //initialize translation from estimate
      g2o::SE3Quat cam1(q,t);
      vc1->setEstimate(cam1);
      vc1->setId(0);  
      vc1->setFixed(false);
      optimizer->addVertex(vc1);
    }

    int v_id = optimizer->vertices().size();
    for (int i=0; i<a.size(); ++i) {    
    	g2o::VertexLineEndpts* v = new g2o::VertexLineEndpts();
   		Eigen::Vector6d line_a;
    	line_a << a[i].A.x, a[i].A.y, a[i].A.z, a[i].B.x, a[i].B.y, a[i].B.z;
    	v->setEstimate(line_a); // line represented wrt the a's node (fixed one)
    	v->setId(v_id++);
    	v->setFixed(false);
    	optimizer->addVertex(v);
    //////// create edges between line and cams    
    //// to a's node
    
    g2o::EdgeSE3LineEndpts * e_a = new g2o::EdgeSE3LineEndpts();
    e_a->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(vc2); 
    e_a->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
    e_a->setMeasurement(line_a);
    e_a->information() = Eigen::Matrix6d::Identity();  // must be identity!
    cv::Mat covA = a[i].rndA.cov,
            covB = a[i].rndB.cov;            
    e_a->endptCov = Eigen::Matrix6d::Identity();
    for(int ii=0; ii<3; ++ii) {
      for(int jj=0; jj<3; ++jj) {
        e_a->endptCov(ii,jj) = covA.at<double>(ii,jj);
      }
    }
    for(int ii=0; ii<3; ++ii) {
      for(int jj=0; jj<3; ++jj) {
        e_a->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
      }
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_a->endptCov.block<3,3>(0,0),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_a->endpt_AffnMat.block<3,3>(0,0) = am;
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_a->endptCov.block<3,3>(3,3),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_a->endpt_AffnMat.block<3,3>(3,3) = am;
    }


    if(sysPara.g2o_BA_use_kernel) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sysPara.g2o_BA_kernel_delta);
      e_a->setRobustKernel(rk);
    }
    e_a->setParameterId(0,0);// param id 0 of the edge corresponds to param id 0 of the optimizer     
    optimizer->addEdge(e_a);
    
    
    //// edge to b's node    
    g2o::EdgeSE3LineEndpts * e_b = new g2o::EdgeSE3LineEndpts();
    e_b->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(vc1); 
    e_b->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
    Eigen::Vector6d line_b;
    line_b << b[i].A.x, b[i].A.y, b[i].A.z, b[i].B.x, b[i].B.y, b[i].B.z;
    e_b->setMeasurement(line_b);
    e_b->information() = Eigen::Matrix6d::Identity();  // must be identity!
    covA = b[i].rndA.cov;
    covB = b[i].rndB.cov;
    e_b->endptCov = Eigen::Matrix6d::Identity();
    for(int ii=0; ii<3; ++ii) {
      for(int jj=0; jj<3; ++jj) {
        e_b->endptCov(ii,jj) = covA.at<double>(ii,jj);
      }
    }
    for(int ii=0; ii<3; ++ii) {
      for(int jj=0; jj<3; ++jj) {
        e_b->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
      }
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_b->endptCov.block<3,3>(0,0),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_b->endpt_AffnMat.block<3,3>(0,0) = am;
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_b->endptCov.block<3,3>(3,3),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_b->endpt_AffnMat.block<3,3>(3,3) = am;
    }


    if(sysPara.g2o_BA_use_kernel) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sysPara.g2o_BA_kernel_delta);
      e_b->setRobustKernel(rk);
    }
    e_b->setParameterId(0,0);
    optimizer->addEdge(e_b);  
  }

  optimizer->initializeOptimization(); 
  optimizer->optimize(15);

  Eigen::Matrix4f rst = vc1->estimate().cast<float>().inverse().matrix();
  for(int i=0; i<3; ++i) {
  	t.at<double>(i) = rst(i,3);
  	for(int j=0; j<3; ++j) {
  		R.at<double>(i,j) = rst(i,j);
  	}
  }

  delete optimizer;
}

bool computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
	// input needs at least 2 correspondences of non-parallel lines
	// the resulting R and t works as below: x'=Rx+t for point pair(x,x');
{
	if(a.size()<2)	{
		cerr<<"Error in computeRelativeMotion_svd: input needs at least 2 pairs!\n";
		return false;
	}
	// convert to the representation of Zhang's paper
	for(int i=0; i<a.size(); ++i) {
		cv::Point3d l, m;
		if(cv::norm(a[i].u)<0.9) {
			l = a[i].B - a[i].A;
			m = (a[i].A + a[i].B) * 0.5;
			a[i].u = l * (1/cv::norm(l));
			a[i].d = a[i].u.cross(m);
		//	cout<<"in computeRelativeMotion_svd compute \n";
		}
		if(cv::norm(b[i].u)<0.9){		
			l = b[i].B - b[i].A;
			m = (b[i].A + b[i].B) * 0.5;
			b[i].u = l * (1/cv::norm(l));
			b[i].d = b[i].u.cross(m);
		}
	}

	cv::Mat A = cv::Mat::zeros(4,4,CV_64F);
	for(int i=0; i<a.size(); ++i) {
		cv::Mat Ai = cv::Mat::zeros(4,4,CV_64F);
		Ai.at<double>(0,1) = (a[i].u-b[i].u).x;
		Ai.at<double>(0,2) = (a[i].u-b[i].u).y;
		Ai.at<double>(0,3) = (a[i].u-b[i].u).z;
		Ai.at<double>(1,0) = (b[i].u-a[i].u).x;
		Ai.at<double>(2,0) = (b[i].u-a[i].u).y;
		Ai.at<double>(3,0) = (b[i].u-a[i].u).z;
		vec2SkewMat(a[i].u+b[i].u).copyTo(Ai.rowRange(1,4).colRange(1,4));
		A = A + Ai.t()*Ai;
	}
	cv::SVD svd(A);
	cv::Mat q = svd.u.col(3);
	//	cout<<"q="<<q<<endl;
	R = q2r(q);
	cv::Mat uu = cv::Mat::zeros(3,3,CV_64F),
		udr= cv::Mat::zeros(3,1,CV_64F);
	for(int i=0; i<a.size(); ++i) {
		uu = uu + vec2SkewMat(b[i].u)*vec2SkewMat(b[i].u).t();
		udr = udr + vec2SkewMat(b[i].u).t()* (cvpt2mat(b[i].d,0)-R*cvpt2mat(a[i].d,0));
	}
	t = uu.inv()*udr;	
	return true;
}

vector<int> computeRelativeMotion_Ransac (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& Ro, cv::Mat& to)
	// compute relative pose between two cameras using 3d line correspondences
	// ransac
{
	if(a.size()<3 || b.size()<3) {
	//	std::cout<<"WARNing: computeRelativeMotion_Ransac input size < 3 \n";
		std::vector<int> tmp;
		return tmp;
	}

	// convert to the representation of Zhang's paper
	vector<Eigen::Vector3d> eaA(a.size()), eaB(a.size()), eaAB(a.size()), ebAB(a.size());
	for(int i=0; i<a.size(); ++i) {
		cv::Point3d l = a[i].B - a[i].A;
		cv::Point3d m = (a[i].A + a[i].B) * 0.5;
		a[i].u = l * (1/cv::norm(l));
		a[i].d = a[i].u.cross(m);
		l = b[i].B - b[i].A;
		m = (b[i].A + b[i].B) * 0.5;
		b[i].u = l * (1/cv::norm(l));
		b[i].d = b[i].u.cross(m);
		eaA[i][0] = a[i].A.x;
		eaA[i][1] = a[i].A.y;
		eaA[i][2] = a[i].A.z;		
		eaB[i][0] = a[i].B.x;
		eaB[i][1] = a[i].B.y;
		eaB[i][2] = a[i].B.z;
		eaAB[i][0] = a[i].A.x-a[i].B.x;
		eaAB[i][1] = a[i].A.y-a[i].B.y;
		eaAB[i][2] = a[i].A.z-a[i].B.z;
		ebAB[i][0] = b[i].A.x-b[i].B.x;
		ebAB[i][1] = b[i].A.y-b[i].B.y;
		ebAB[i][2] = b[i].A.z-b[i].B.z;

	}	

	// ----- start ransac -----
	int minSolSetSize = 3, maxIters = sysPara.ransac_iters_line_motion;
	double distThresh = sysPara.pt2line3d_dist_relmotion; // in meter
	double angThresh  = sysPara.line3d_angle_relmotion; // deg
	double lineAngleThresh_degeneracy = 5*PI/180; //5 degree
	double inlierRatio = 1;

	vector<int> indexes;
	for(int i=0; i<a.size(); ++i)	indexes.push_back(i);
	int iter = 0;
	vector<int> maxConSet;
	cv::Mat bR, bt;
	while(iter < maxIters) {
		vector<int> inlier;
		iter++;
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
		
		vector<RandomLine3d> suba, subb;
		for(int i=0; i<minSolSetSize; ++i) {
			suba.push_back(a[indexes[i]]);
			subb.push_back(b[indexes[i]]);
		}
		// ---- check degeneracy ---- 	
		bool degenerate = true;
		// if at least one pair is not parallel, then it's non-degenerate
		for(int i=0; i<minSolSetSize; ++i){
			for(int j=i+1; j<minSolSetSize; ++j) {
				if(abs(suba[i].u.dot(suba[j].u)) < cos(lineAngleThresh_degeneracy)) {
					degenerate = false;
					break;
				}
			}
			if(!degenerate)
				break;
		}
		if(degenerate) continue; // degenerate set is not usable

		cv::Mat R, t;
		computeRelativeMotion_svd(suba, subb, R, t);
		// find consensus
		Eigen::Matrix3d eR;
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j)
				eR(i,j) = R.at<double>(i,j);
		Eigen::Vector3d et(t.at<double>(0),t.at<double>(1),t.at<double>(2));	
		for(int i=0; i<a.size(); ++i) {
			Eigen::Vector3d aA_ = eR*eaA[i]+et, aB_ = eR*eaB[i]+et;
		
			double dist = 0.5*dist3d_pt_line (cv::Point3d(aA_(0),aA_(1),aA_(2)), b[i].A, b[i].B)
						+ 0.5*dist3d_pt_line (cv::Point3d(aB_(0),aB_(1),aB_(2)), b[i].A, b[i].B);
			double angle = 180*acos(abs((eR * eaAB[i]).dot(ebAB[i])/
							cv::norm(a[i].A - a[i].B)/cv::norm(b[i].A - b[i].B)))/PI; 

			if(dist < distThresh && angle < angThresh) {		
				inlier.push_back(i);
			}			
		}
		
		if(inlier.size() > maxConSet.size()) {
			maxConSet = inlier;
			bR = R;
			bt = t;
		}
		if(maxConSet.size() >= a.size()*inlierRatio)
			break;

	}
/*	//// donot try finding more inliers, save time, not affect performance much
	Ro = bR; to = bt;
	return maxConSet; 
*/
	if(maxConSet.size()<1)
		return maxConSet;
	
	Ro = bR; to = bt;
	if(maxConSet.size()<4)
		return maxConSet;
	// ---- apply svd to all inliers ----
	vector<RandomLine3d> ina, inb;
	for(int i=0; i<maxConSet.size();++i) {
		ina.push_back(a[maxConSet[i]]);
		inb.push_back(b[maxConSet[i]]);
	}
//	optimize_motion_g2o(ina, inb, Ro, to);
	optimizeRelmotion(ina, inb, Ro, to); // this optimization slow, non-thread-safe and rarely-helpful.
	
	cv::Mat R = Ro, t = to;
	vector<int> prevConSet ;
	while(1) {		
		vector<int> conset;
		Eigen::Matrix3d eR;
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j)
				eR(i,j) = R.at<double>(i,j);
		Eigen::Vector3d et(t.at<double>(0),t.at<double>(1),t.at<double>(2));	
		for(int i=0; i<a.size(); ++i) {
			Eigen::Vector3d aA_ = eR*eaA[i]+et, aB_ = eR*eaB[i]+et;
		
			double dist = 0.5*dist3d_pt_line (cv::Point3d(aA_(0),aA_(1),aA_(2)), b[i].A, b[i].B)
						+ 0.5*dist3d_pt_line (cv::Point3d(aB_(0),aB_(1),aB_(2)), b[i].A, b[i].B);
			double angle = 180*acos(abs((eR * eaAB[i]).dot(ebAB[i])/
							cv::norm(a[i].A - a[i].B)/cv::norm(b[i].A - b[i].B)))/PI; 
			if(dist < distThresh && angle < angThresh) {
				conset.push_back(i);
			}
		}
		if(conset.size() <= prevConSet.size()) 
			break;
		else {
			prevConSet = conset;
			Ro = R;
			to = t;
			ina.clear(); inb.clear();
			for(int i=0; i<prevConSet.size();++i) {
				ina.push_back(a[prevConSet[i]]);
				inb.push_back(b[prevConSet[i]]);
			}
			optimizeRelmotion(ina, inb, R, t); // this optimization slow, non-thread-safe and rarely-helpful.
		//	optimize_motion_g2o(ina, inb, R, t);
		}	
	}
	
	return prevConSet;
}



Eigen::Matrix4f getTransform_Lns_Pts_pcl(const Node* trainNode, const Node* queryNode, 
						  			 	const std::vector<cv::DMatch>& point_matches,
						  			 	const std::vector<cv::DMatch>& line_matches,
						  			 	bool& valid)
// Note: the result transforms a point from the queryNode's CS to the trainNode's CS
{
	if(point_matches.size()<1 || point_matches.size()+line_matches.size()<3) { 
		valid = false;
		return Eigen::Matrix4f();
	}
	pcl::TransformationFromCorrespondences tfc;

	//// project the point to all lines
	for(int i=0; i < line_matches.size(); ++i) {
		int ptidx = rand()%point_matches.size();
		cv::Point3d train_pt(trainNode->feature_locations_3d_[point_matches[ptidx].trainIdx](0),
  						 trainNode->feature_locations_3d_[point_matches[ptidx].trainIdx](1),
  						 trainNode->feature_locations_3d_[point_matches[ptidx].trainIdx](2));
		cv::Point3d query_pt(queryNode->feature_locations_3d_[point_matches[ptidx].queryIdx](0),
  						 queryNode->feature_locations_3d_[point_matches[ptidx].queryIdx](1),
  						 queryNode->feature_locations_3d_[point_matches[ptidx].queryIdx](2));
		const cv::DMatch& m = line_matches[i];
		cv::Point3d train_prj = projectPt3d2Ln3d_2 (train_pt, trainNode->lines[m.trainIdx].line3d.A, trainNode->lines[m.trainIdx].line3d.B);
  		cv::Point3d query_prj = projectPt3d2Ln3d_2 (query_pt, queryNode->lines[m.queryIdx].line3d.A, queryNode->lines[m.queryIdx].line3d.B);
  		Eigen::Vector3f from(query_prj.x, query_prj.y, query_prj.z),
  						to(train_prj.x, train_prj.y, train_prj.z);
  		if(isnan(from(2)) || isnan(to(2)))
  		  	continue;
  		float weight =1/(abs(to[2]) + abs(from[2]));
  		tfc.add(from, to, weight);
	}
	//// add points to the tfc
	for(int i=0; i<point_matches.size();++i) {
		const cv::DMatch& m = point_matches[i];
		Eigen::Vector3f from = queryNode->feature_locations_3d_[m.queryIdx].head<3>();
  		Eigen::Vector3f to = trainNode->feature_locations_3d_[m.trainIdx].head<3>();
  		if(isnan(from(2)) || isnan(to(2)))
  			continue;
  		float weight = 1.0;
  		weight =1/(abs(to[2]) + abs(from[2]));
  		tfc.add(from, to, weight);
  	}

	if(tfc.getNoOfSamples()<3)
		valid = false;
  	else
  		valid = true;
  // get relative movement from samples
  	return tfc.getTransformation().matrix();
}

Eigen::Matrix4f getTransform_Line_svd(const Node* trainNode, const Node* queryNode, 
						  			 	 const std::vector<cv::DMatch>& matches,
						  			 	 bool& valid)
// Note: the result transforms a point from the queryNode's CS to the trainNode's CS
{
	if(matches.size()<2) {
		valid = false;
		return Eigen::Matrix4f();
	}
	vector<RandomLine3d> train(matches.size()), query(matches.size());
	for(int i=0; i<matches.size(); ++i) {
		train[i] = trainNode->lines[matches[i].trainIdx].line3d;
		query[i] = queryNode->lines[matches[i].queryIdx].line3d;
	}
	cv::Mat R, t;
	valid = computeRelativeMotion_svd (query, train, R, t);
	Eigen::Matrix4f tf;
	tf << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
	      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
	      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2),
	      0, 0, 0, 1;
	return tf;
}

bool getTransform_PtsLines_ransac (const Node* trainNode, const Node* queryNode, 
									const std::vector<cv::DMatch> all_point_matches,
									const std::vector<cv::DMatch> all_line_matches,
									std::vector<cv::DMatch>& output_point_inlier_matches,
									std::vector<cv::DMatch>& output_line_inlier_matches,
									Eigen::Matrix4f& ransac_tf, 
									float& inlier_rmse)
// input: 3d point matches + 3d line matches
// output: rigid transform between two frames
{
	int nPt = all_point_matches.size();
	int nLn = all_line_matches.size();

	int min_inlier_nmb = sysPara.min_feature_matches;
	int line_weight = sysPara.line_match_number_weight;

	if(nPt + nLn * line_weight < min_inlier_nmb) {
		inlier_rmse  = 1e9;
		return false;
	}

  	if(min_inlier_nmb > 0.7 * (nPt + nLn * line_weight)){
    	min_inlier_nmb = 0.7 * (nPt + nLn * line_weight); // adjust threshold
  	}

  	// ==== for loop closing pairs ====
  	if(abs(trainNode->id_-queryNode->id_) > 50) {
  		min_inlier_nmb = sysPara.min_matches_loopclose;
  	}

	vector<int> indexes(nPt+nLn);
	for(int i=0; i<indexes.size(); ++i) indexes[i] = i;
	int maxIter = sysPara.ransac_iters_line_motion;
	int minSolSetSize = 3;
	double 	ln_angle_thres_deg = sysPara.line3d_angle_relmotion, 
			mahdist4inlier = sysPara.max_mah_dist_for_inliers; //

    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > train_lines_A(nLn), train_lines_B(nLn), query_lines_A(nLn), query_lines_B(nLn);
    for(int i=0; i<nLn; ++i) {
    	const cv::DMatch& m = all_line_matches[i];
    	train_lines_A[i] = (Eigen::Vector4f(trainNode->lines[m.trainIdx].line3d.A.x, trainNode->lines[m.trainIdx].line3d.A.y, trainNode->lines[m.trainIdx].line3d.A.z, 1));
    	train_lines_B[i] = (Eigen::Vector4f(trainNode->lines[m.trainIdx].line3d.B.x, trainNode->lines[m.trainIdx].line3d.B.y, trainNode->lines[m.trainIdx].line3d.B.z, 1));
    	query_lines_A[i] = (Eigen::Vector4f(queryNode->lines[m.queryIdx].line3d.A.x, queryNode->lines[m.queryIdx].line3d.A.y, queryNode->lines[m.queryIdx].line3d.A.z, 1));
    	query_lines_B[i] = (Eigen::Vector4f(queryNode->lines[m.queryIdx].line3d.B.x, queryNode->lines[m.queryIdx].line3d.B.y, queryNode->lines[m.queryIdx].line3d.B.z, 1));
    } 

	int iter = 0; 	
	float sum_squared_error = 1e9;
	vector<cv::DMatch> max_point_inlier_set, max_line_inlier_set;
	Eigen::Matrix4f tf_best;
	while (iter < maxIter) {
		++iter;
		///// get a minimal solution /////
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle indexes
		std::vector<cv::DMatch> ptMch, lnMch;
        for(int i=0; i<minSolSetSize; ++i) {
        	if(indexes[i]<nPt) 
        		ptMch.push_back(all_point_matches[indexes[i]]);
        	else
        		lnMch.push_back(all_line_matches[indexes[i]-nPt]);
        }
        Eigen::Matrix4f tf;
        bool validTf;
        float sse = 0;
        if(lnMch.size() == minSolSetSize) { // all lines
        	tf = getTransform_Line_svd(trainNode, queryNode, lnMch, validTf); // 0.03 ms, 10 times of others
        } else { // mixed pt + line or pure pts
        	tf = getTransform_Lns_Pts_pcl(trainNode, queryNode, ptMch, lnMch, validTf);
        }
        if(!validTf) continue;
        
        ///// evaluate minimal solution /////
        vector<cv::DMatch> inlier_point_matches, inlier_line_matches;
        inlier_point_matches.reserve(nPt); inlier_line_matches.reserve(nLn);
        Eigen::Matrix4d tf_d = tf.cast<double>();
        for(int i=0; i<all_point_matches.size(); ++i)  {
        	double mah_dist_sq = errorFunction2(queryNode->feature_locations_3d_[all_point_matches[i].queryIdx], 
        		                             trainNode->feature_locations_3d_[all_point_matches[i].trainIdx], tf_d);
        	if (mah_dist_sq < mahdist4inlier * mahdist4inlier) {
        		inlier_point_matches.push_back(all_point_matches[i]);
        		sse += mah_dist_sq; 
        	}        	
        }
        for(int i=0; i<all_line_matches.size(); ++i) {
        	Eigen::Vector4f qA_tf = tf * query_lines_A[i];
        	Eigen::Vector4f qB_tf = tf * query_lines_B[i];		
			cv::Point3d qA_in_train(qA_tf(0),qA_tf(1),qA_tf(2)), qB_in_train(qB_tf(0),qB_tf(1),qB_tf(2));
			double mah_dist_a = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndA, qA_in_train, qB_in_train) ;
			double mah_dist_b = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndB, qA_in_train, qB_in_train);
#ifdef MOTION_USE_MAHDIST			
			if(mah_dist_a < mahdist4inlier && mah_dist_b < mahdist4inlier) {
				inlier_line_matches.push_back(all_line_matches[i]);
				sse += mah_dist_a * mah_dist_a+ mah_dist_b * mah_dist_b;
			}		
		
#endif
			double dist = 0.5*dist3d_pt_line (qA_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B)
						+ 0.5*dist3d_pt_line (qB_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B);
			cv::Point3d l1 = qA_in_train - qB_in_train, 
				l2 = trainNode->lines[all_line_matches[i].trainIdx].line3d.A - trainNode->lines[all_line_matches[i].trainIdx].line3d.B;			
			double angle = 180*acos(abs(l1.dot(l2)/cv::norm(l1)/cv::norm(l2)))/PI; 
#ifndef MOTION_USE_MAHDIST
			if(dist < sysPara.pt2line3d_dist_relmotion && angle < sysPara.line3d_angle_relmotion) {
				inlier_line_matches.push_back(all_line_matches[i]);
				sse += mah_dist_a * mah_dist_a+ mah_dist_b * mah_dist_b;
			}
#endif
	
        }
        if(inlier_point_matches.size() + line_weight * inlier_line_matches.size() 
        	> max_point_inlier_set.size() + line_weight * max_line_inlier_set.size()) {
        	max_point_inlier_set = inlier_point_matches;
        	max_line_inlier_set  = inlier_line_matches;
        	tf_best = tf;
        	sum_squared_error = sse;
        }
	}
	if(max_point_inlier_set.size() + max_line_inlier_set.size() < 3) 
	{
		return false;
	}
		///////// refine solution /////////
	vector<cv::DMatch> refined_point_inliers, refined_line_inliers ;
	Eigen::Matrix4f refined_tf = tf_best;
	int tmp_best_line_mah;
	getTransformFromHybridMatchesG2O (trainNode, queryNode, max_point_inlier_set, max_line_inlier_set, refined_tf, 25);
	double refined_rmse = sqrt(sum_squared_error/(max_point_inlier_set.size() + max_line_inlier_set.size()));

/*
///// write matches to file //////
	ofstream ofs_l; ofs_l.open("/home/lu/data_line.txt");
	ofstream ofs_p; ofs_p.open("/home/lu/data_pt.txt");
	cout<<"max_point_inlier_set="<<max_point_inlier_set.size()<<endl;
	cout<<"max_line_inlier_set="<<max_line_inlier_set.size()<<endl;
	for(int i=0; i<all_point_matches.size();++i) {
		ofs_p << trainNode->feature_locations_3d_[all_point_matches[i].trainIdx][0] << '\t'
			  << trainNode->feature_locations_3d_[all_point_matches[i].trainIdx][1] << '\t'
			  << trainNode->feature_locations_3d_[all_point_matches[i].trainIdx][2] << '\n';
	}
	ofs_p.close();
	for(int i=0; i<max_line_inlier_set.size();++i) {
		ofs_l << trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.pos.x <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.pos.y <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.pos.z <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.pos.x <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.pos.y <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.pos.z <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(0,0) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(0,1) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(0,2) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(1,0) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(1,1) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(1,2) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(2,0) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(2,1) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndA.cov.at<double>(2,2) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(0,0) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(0,1) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(0,2) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(1,0) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(1,1) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(1,2) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(2,0) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(2,1) <<'\t'
			<< trainNode->lines[max_line_inlier_set[i].trainIdx].line3d.rndB.cov.at<double>(2,2) <<'\n';
	}
	ofs_l.close();
//	int tmp; cin>>tmp;
*/

	for(int iter = 0; iter <20; ++iter) {
		vector<cv::DMatch> tmp_pt_inliers, tmp_ln_inliers;
		tmp_pt_inliers.reserve(all_point_matches.size()); tmp_ln_inliers.reserve(all_line_matches.size());
		double tmp_sse = 0;
		
		Eigen::Matrix4d refined_tf_d = refined_tf.cast<double>();
        for(int i=0; i<all_point_matches.size(); ++i)  {
        	double mah_dist_sq = errorFunction2(queryNode->feature_locations_3d_[all_point_matches[i].queryIdx], 
        		                             trainNode->feature_locations_3d_[all_point_matches[i].trainIdx], refined_tf_d);
        	
        /*	Eigen::Vector4f q_tf = refined_tf * queryNode->feature_locations_3d_[all_point_matches[i].queryIdx];
        	Eigen::Vector4f train = trainNode->feature_locations_3d_[all_point_matches[i].trainIdx];
        	RandomPoint3d train_rnd = compPt3dCov(cv::Point3d(train(0),train(1),train(2)), K);
        	cv::Mat dif = (cv::Mat_<double>(3,1)<< q_tf(0) - train(0), q_tf(1) - train(1), q_tf(2) - train(2));
        	double my_mah_dist_sq = cv::norm(dif.t() * train_rnd.cov.inv() * dif);
        	cout<<"mah_dist = "<< sqrt(mah_dist_sq) <<"\t" << sqrt(my_mah_dist_sq) << "\t" << sqrt(my_mah_dist_sq)/sqrt(mah_dist_sq)<<endl;
		*/
        	if (mah_dist_sq < mahdist4inlier * mahdist4inlier) {
        		tmp_pt_inliers.push_back(all_point_matches[i]);
        		tmp_sse += mah_dist_sq; 
        	}        	
        }

        int tmp_line_mah_inlier = 0;
        for(int i=0; i<all_line_matches.size(); ++i) {
        	Eigen::Vector4f qA_tf = refined_tf * query_lines_A[i];
        	Eigen::Vector4f qB_tf = refined_tf * query_lines_B[i];		
			cv::Point3d qA_in_train(qA_tf(0),qA_tf(1),qA_tf(2)), qB_in_train(qB_tf(0),qB_tf(1),qB_tf(2));
			double mah_dist_a = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndA, qA_in_train, qB_in_train); 
			double mah_dist_b = mah_dist3d_pt_line(trainNode->lines[all_line_matches[i].trainIdx].line3d.rndB, qA_in_train, qB_in_train);
			if(mah_dist_a < mahdist4inlier && mah_dist_b < mahdist4inlier) {
				tmp_line_mah_inlier++;
#ifdef MOTION_USE_MAHDIST
				tmp_ln_inliers.push_back(all_line_matches[i]);
				tmp_sse += mah_dist_a * mah_dist_a + mah_dist_b * mah_dist_b;
#endif
			}					
			
			double dist = 0.5*dist3d_pt_line (qA_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B)
						+ 0.5*dist3d_pt_line (qB_in_train, trainNode->lines[all_line_matches[i].trainIdx].line3d.A, trainNode->lines[all_line_matches[i].trainIdx].line3d.B);
			cv::Point3d l1 = qA_in_train - qB_in_train, 
				l2 = trainNode->lines[all_line_matches[i].trainIdx].line3d.A - trainNode->lines[all_line_matches[i].trainIdx].line3d.B;			
			double angle = 180*acos(abs(l1.dot(l2)/cv::norm(l1)/cv::norm(l2)))/PI; 
#ifndef MOTION_USE_MAHDIST
			if(dist < sysPara.pt2line3d_dist_relmotion && angle < sysPara.line3d_angle_relmotion) {
				tmp_ln_inliers.push_back(all_line_matches[i]);
				tmp_sse += mah_dist_a * mah_dist_a+ mah_dist_b * mah_dist_b;
			//	cout<<mah_dist_a<<"\t"<<mah_dist_b<<endl;
			}
#endif		
        }
        if( tmp_pt_inliers.size() + tmp_ln_inliers.size() * sysPara.line_match_number_weight 
              > refined_point_inliers.size() + refined_line_inliers.size() * sysPara.line_match_number_weight
         //   && sqrt(tmp_sse/(tmp_pt_inliers.size() + tmp_ln_inliers.size())) <= refined_rmse 
           ) {
           	tmp_best_line_mah = tmp_line_mah_inlier;
        	refined_point_inliers = tmp_pt_inliers;
        	refined_line_inliers = tmp_ln_inliers;
        	refined_rmse = sqrt(tmp_sse/(tmp_pt_inliers.size() + tmp_ln_inliers.size()));       
        	getTransformFromHybridMatchesG2O (trainNode, queryNode, refined_point_inliers, refined_line_inliers, refined_tf, 20);

        } else
        	break;

   	}

	output_point_inlier_matches = refined_point_inliers;
	output_line_inlier_matches = refined_line_inliers;
	inlier_rmse = refined_rmse;
	ransac_tf = refined_tf;

//	cout<<"line inliers "<<refined_line_inliers.size() <<"\t"<<tmp_best_line_mah<<"\t"
//		<<(double)refined_line_inliers.size()/all_line_matches.size()<<"\t"<<(double)tmp_best_line_mah/all_line_matches.size()<<endl;
	return (refined_point_inliers.size() + line_weight * refined_line_inliers.size()) >= min_inlier_nmb;
} 


cv::Mat getRotationFromPoints( const Node* trainNode, const Node* queryNode, 
							   const std::vector<cv::DMatch> all_matches,
								std::vector<cv::DMatch>& output_inlier_matches,
								bool& valid)
{
	if(trainNode->feature_locations_2d_.size() != trainNode->feature_locations_3d_.size())
		cout<<trainNode->feature_locations_2d_.size()<<" =x= "<<trainNode->feature_locations_3d_.size()<<endl;

	int n = all_matches.size();
	cv::Mat pts_train(3, n, CV_64FC1), pts_query(3, n, CV_64FC1);
	for(int i=0; i<n; ++i) {
		cv::DMatch m = all_matches[i];
		pts_train.at<double>(0,i) = trainNode->feature_locations_2d_[m.trainIdx].pt.x;
		pts_train.at<double>(1,i) = trainNode->feature_locations_2d_[m.trainIdx].pt.y;
		pts_train.at<double>(2,i) = 1;
		pts_query.at<double>(0,i) = queryNode->feature_locations_2d_[m.queryIdx].pt.x;
		pts_query.at<double>(1,i) = queryNode->feature_locations_2d_[m.queryIdx].pt.y;
		pts_query.at<double>(2,i) = 1;
	}
	cv::Mat normalized_pts_train = K.inv() * pts_train;
	cv::Mat normalized_pts_query = K.inv() * pts_query;
	cv::Mat E, inlierMask;
	while (!essn_ransac (&normalized_pts_train, &normalized_pts_query, &E, K, &inlierMask, 640));

	cv::Mat R, t, R1, R2, trans;
	decEssential (&E, &R1, &R2, &trans); 
	//--- check Cheirality ---
	// use all point matches instead of just one for robustness (very necessary!)
	int posTrans = 0, negTrans = 0; 
	cv::Mat Rt, pRt, nRt;
	for (int i=0; i < n; ++i) {
		if (!inlierMask.at<uchar>(i)) continue; //outlier
		Rt = findTrueRt(R1, R2, trans, mat2cvpt(normalized_pts_train.col(i)), mat2cvpt(normalized_pts_query.col(i)));
		if (Rt.cols != 4) continue;
		if(trans.dot(Rt.col(3)) > 0) {
			++posTrans;
			pRt = Rt;
		} else	{
			++negTrans;
			nRt = Rt;		
		}
	}	
	if (posTrans > negTrans) {
		t = trans;
		R = pRt.colRange(0,3);
	} else {
		t = -trans;
		R = nRt.colRange(0,3);
	}
	return R;
}