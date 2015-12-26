#include "node.h"
#include "scoped_timer.h"
#include "transformation_estimation.h"
//#include "g2o/core/graph_optimizer_sparse.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/types/icp/types_icp.h"

#include "g2o/types/slam3d/se3quat.h"
#include "g2o/types/slam3d/edge_se3_pointxyz_depth.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "misc2.h" //Only for point_information_matrix. TODO: Move to misc2.h
#include <Eigen/SVD>
//TODO: Move these definitions and includes into a common header, g2o.h
#include "g2o/core/estimate_propagator.h"
//#include "g2o/core/factory.h"
//#include "g2o/core/solver_factory.h"
#include "g2o/core/hyper_dijkstra.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"

using namespace Eigen;
using namespace std;
using namespace g2o;

#ifdef USE_LINES
#include "Eigen/src/SVD/JacobiSVD.h"
#include "line/utils.h"
extern SystemParameters sysPara;
extern cv::Mat K;
#endif

typedef g2o::VertexPointXYZ  feature_vertex_type;
typedef g2o::EdgeSE3PointXYZDepth feature_edge_type;
//TODO: Make a class of this, with optimizerSetup being the constructor.
//      getTransformFromMatchesG2O into a method for adding a node that 
//      can be called as often as desired and one evaluation method.
//      Needs to keep track of mapping (node id, feature id) -> vertex id

//!Set parameters for icp optimizer
void optimizerSetup(g2o::SparseOptimizer& optimizer){
  //optimizer.setMethod(g2o::SparseOptimizer::LevenbergMarquardt);
  //g2o::Optimizato
  optimizer.setVerbose(0);

  // variable-size block solver
  g2o::BlockSolverX::LinearSolverType * linearSolver
      = new g2o::LinearSolverCholmod<g2o ::BlockSolverX::PoseMatrixType>();


  g2o::BlockSolverX * solver_ptr
      = new g2o::BlockSolverX(linearSolver);

 // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr); // commented out by ylu
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // added by ylu, GaussNewton produces poor/problematic results
  
  optimizer.setAlgorithm(solver);
  //optimizer.setSolver(solver_ptr);

  g2o::ParameterCamera* cameraParams = new g2o::ParameterCamera();
  //FIXME From Parameter server or cam calibration
//  cameraParams->setKcam(521,521,319.5,239.5);
  cameraParams->setKcam(525,525,319.5,239.5); // added by ylu, from cam_info
  g2o::SE3Quat offset; // identity
  cameraParams->setOffset(offset);
  cameraParams->setId(0);
  optimizer.addParameter(cameraParams);
}

void optimizerSetup2(g2o::SparseOptimizer& optimizer) {
  // some handy typedefs
  typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
  typedef g2o::LinearSolverCSparse<MyBlockSolver::PoseMatrixType> MyLinearSolver;
  
  optimizer.setVerbose(false);
  MyLinearSolver* linearSolver = new MyLinearSolver();
  
  MyBlockSolver* solver_ptr = new MyBlockSolver(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(solver);

  g2o::ParameterCamera* cameraParams = new g2o::ParameterCamera();
  //FIXME From Parameter server or cam calibration
//  cameraParams->setKcam(521,521,319.5,239.5);
  cameraParams->setKcam(525,525,319.5,239.5); // added by ylu, from cam_info
  g2o::SE3Quat offset; // identity
  cameraParams->setOffset(offset);
  cameraParams->setId(0);
  optimizer.addParameter(cameraParams);
}

//Second cam is fixed, therefore the transformation from the second to the first cam will be computed
std::pair<g2o::VertexSE3*, g2o::VertexSE3*>  sensorVerticesSetup(g2o::SparseOptimizer& optimizer, Eigen::Matrix4f& tf_estimate)
{
    g2o::VertexSE3 *vc2 = new VertexSE3();
    {// set up rotation and translation for the newer node as identity
      Eigen::Quaterniond q(1,0,0,0);
      Eigen::Vector3d t(0,0,0);
      q.setIdentity();
      g2o::SE3Quat cam2(q,t);

      vc2->setEstimate(cam2);
      vc2->setId(1); 
      vc2->setFixed(true);
      optimizer.addVertex(vc2);
    }

    g2o::VertexSE3 *vc1 = new VertexSE3();
    {
      Eigen::Quaterniond q(tf_estimate.topLeftCorner<3,3>().cast<double>());//initialize rotation from estimate 
      Eigen::Vector3d t(tf_estimate.topRightCorner<3,1>().cast<double>());  //initialize translation from estimate
      g2o::SE3Quat cam1(q,t);
      vc1->setEstimate(cam1);
      vc1->setId(0);  
      vc1->setFixed(false);
      optimizer.addVertex(vc1);
    }

    // add to optimizer
    return std::make_pair(vc1, vc2);
}
feature_edge_type* edgeToFeature(const Node* node, 
                              unsigned int feature_id,
                              g2o::VertexSE3* camera_vertex,
                              g2o::VertexPointXYZ* feature_vertex)
{
   feature_edge_type* edge = new feature_edge_type();
   cv::KeyPoint kp = node->feature_locations_2d_[feature_id];
   Vector4f position = node->feature_locations_3d_[feature_id];
   float depth = position(2);
   if(!isnan(depth))
   {
     Eigen::Vector3d pix_d(kp.pt.x,kp.pt.y,depth);
     //ROS_INFO_STREAM("Edge from camera to position "<< pix_d.transpose());
     edge->setMeasurement(pix_d);
     Eigen::Matrix3d info_mat = point_information_matrix(depth);
     edge->setInformation(info_mat);
     feature_vertex->setEstimate(position.cast<double>().head<3>());

   } 
   else {//FIXME instead of using an arbitrary depth value and high uncertainty, use the correct vertex type (on the other hand Rainer suggested this proceeding too)
     Eigen::Vector3d pix_d(kp.pt.x,kp.pt.y,10.0);
     //ROS_INFO_STREAM("Edge from camera to position "<< pix_d.transpose());
     edge->setMeasurement(pix_d);
     Eigen::Matrix3d info_mat = point_information_matrix(10);//as uncertain as if it was ...meter away
     edge->setInformation(info_mat);
     feature_vertex->setEstimate(Eigen::Vector3d(position(0)*10, position(1)*10, 10.0));//Move from 1m to 10 m distance. Shouldn't matter much
   }
   edge->setParameterId(0,0);
   //edge->setRobustKernel(true);
   edge->vertices()[0] = camera_vertex;
   edge->vertices()[1] = feature_vertex;
   feature_vertex->setFixed(false);
   return edge;
}
//!Compute 
void getTransformFromMatchesG2O(const Node* earlier_node,
                                const Node* newer_node,
                                const std::vector<cv::DMatch> & matches,
                                Eigen::Matrix4f& transformation_estimate, //Input (initial guess) and Output
                                int iterations)
{
  ScopedTimer s(__FUNCTION__);
  //G2O Initialization
  g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();//TODO: Speicherleck
  //Set parameters for icp optimizer
  optimizerSetup(*optimizer);
  //First camera is earlier_node, second camera is newer_node
  //Second cam is set to fixed, therefore the transformation from the second to the first cam will be computed
  Eigen::Matrix4f tfinv = transformation_estimate.inverse(); // added by ylu, bug fixed
  std::pair<g2o::VertexSE3*, g2o::VertexSE3*> cams = sensorVerticesSetup(*optimizer, tfinv);


    //std::vector<feature_vertex_type*> feature_vertices;
  int v_id = optimizer->vertices().size(); //0 and 1 are taken by sensor vertices
  //For each match, create a vertex and connect it to the sensor vertices with the measured position
  BOOST_FOREACH(const cv::DMatch& m, matches)
  {
    feature_vertex_type* v = new feature_vertex_type();//TODO: Speicherleck?
    v->setId(v_id++);
    v->setFixed(false);

    optimizer->addVertex(v);
    //feature_vertices.push_back(v);

    feature_edge_type* e1 = edgeToFeature(earlier_node, m.trainIdx, cams.first, v);
    optimizer->addEdge(e1);
    /* ylu: the estimate of v is finally set in edgeToFeature below, overwriting the one above. 
    / this is correct since second cam is fixed, though it's very confusing at first glance.
    / the order of calling edgeToFeature here matters! */
    feature_edge_type* e2 = edgeToFeature(newer_node, m.queryIdx, cams.second, v);
    optimizer->addEdge(e2);  
  }

  optimizer->initializeOptimization();

  ROS_INFO("Optimizer Size: %zu", optimizer->vertices().size());
//  optimizer->computeActiveErrors(); std::cout<<"g2o error "<< optimizer->activeChi2();

  optimizer->optimize(iterations);

 // optimizer->computeActiveErrors(); std::cout<<" => " <<optimizer->activeChi2()<<'\n';

  transformation_estimate = cams.first->estimate().cast<float>().inverse().matrix();


  delete optimizer;

  
}

#ifdef USE_LINES

void getTransformFromHybridMatchesG2O (const Node* earlier_node,
                                       const Node* newer_node,
                                       const std::vector<cv::DMatch> & pt_matches,
                                       const std::vector<cv::DMatch> & ln_matches,
                                       Eigen::Matrix4f& transformation_estimate, //Input (initial guess) and Output
                                       int iterations)
{
  //G2O Initialization
  g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();
  //Set parameters for optimizer
  optimizerSetup(*optimizer);
  //First camera is earlier_node, second camera is newer_node
  //Second cam is set to fixed, therefore the transformation from the second to the first cam will be computed
  Eigen::Matrix4f tfinv = transformation_estimate.inverse();
  std::pair<g2o::VertexSE3*, g2o::VertexSE3*> cams = sensorVerticesSetup(*optimizer, tfinv);

  int v_id = optimizer->vertices().size(); //0 and 1 are taken by sensor vertices
  vector<g2o::EdgeSE3LineEndpts* > ln_edges;

////// add the parameter representing the sensor offset  !!!!
    g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
    sensorOffset->setOffset(Eigen::Isometry3d::Identity());
    sensorOffset->setId(1);
    optimizer->addParameter(sensorOffset);


// camera intrinsic parameters
  double focal_len = K.at<double>(0,0), // focal length
         cu = K.at<double>(0,2),
         cv = K.at<double>(1,2);
  // ************** point matches *****************
#ifdef USE_MY_SE3_PTXYZ_EDGE

  vector<g2o::EdgeSE3PointXYZ*> pt_edges;
   BOOST_FOREACH(const cv::DMatch& m, pt_matches)
  {
   
    Eigen::Vector3f pt_newer = newer_node->feature_locations_3d_[m.queryIdx].head<3>();

    g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ(); 
    v->setEstimate(pt_newer.cast<double>());
    v->setId(v_id++);
    v->setFixed(false);
    optimizer->addVertex(v);

    g2o::EdgeSE3PointXYZ* e_newer = new g2o::EdgeSE3PointXYZ();
    e_newer->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.second); 
    e_newer->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
    e_newer->setMeasurement(pt_newer.cast<double>());
    e_newer->information() = compPt3dCov(pt_newer, focal_len, cu, cv, newer_node->asynch_time_diff_sec_).cast<double>().inverse();

    if(sysPara.g2o_BA_use_kernel) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sysPara.g2o_BA_kernel_delta);
      e_newer->setRobustKernel(rk);
    }
    e_newer->setParameterId(0,1);
    optimizer->addEdge(e_newer);


    g2o::EdgeSE3PointXYZ* e_older = new g2o::EdgeSE3PointXYZ();
    e_older->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.first); 
    e_older->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
    Eigen::Vector3f pt_older = earlier_node->feature_locations_3d_[m.trainIdx].head<3>();
    e_older->setMeasurement(pt_older.cast<double>());
    e_older->information() = compPt3dCov(pt_older, focal_len, cu, cv, earlier_node->asynch_time_diff_sec_).cast<double>().inverse();
    if(sysPara.g2o_BA_use_kernel) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sysPara.g2o_BA_kernel_delta);
      e_older->setRobustKernel(rk);
    }
    e_older->setParameterId(0,1);
    optimizer->addEdge(e_older);
    pt_edges.push_back(e_newer);
    pt_edges.push_back(e_older);
  }
  
#else  
  vector<feature_edge_type*> pt_edges;
  //For each match, create a vertex and connect it to the sensor vertices with the measured position
  BOOST_FOREACH(const cv::DMatch& m, pt_matches)
  {
    feature_vertex_type* v = new feature_vertex_type();//
    v->setId(v_id++);
    v->setFixed(false);

    optimizer->addVertex(v);
    //feature_vertices.push_back(v);

    feature_edge_type* e1 = edgeToFeature(earlier_node, m.trainIdx, cams.first, v);
    optimizer->addEdge(e1);
    // ylu: the estimate of v is finally set in edgeToFeature below, overwriting the one above. 
    // this is correct since second cam is fixed, though it's very confusing at first glance.
    // the order of calling edgeToFeature here matters! 
    feature_edge_type* e2 = edgeToFeature(newer_node, m.queryIdx, cams.second, v);
    optimizer->addEdge(e2);
    pt_edges.push_back(e1);
    pt_edges.push_back(e2);   

  }
#endif
  // *************** line matches ****************  
  BOOST_FOREACH(const cv::DMatch& m, ln_matches) 
  {
    //// create a line vertex 
    g2o::VertexLineEndpts* v = new g2o::VertexLineEndpts();
    Eigen::Vector6d line_new;
    
    line_new << newer_node->lines[m.queryIdx].line3d.A.x, newer_node->lines[m.queryIdx].line3d.A.y, newer_node->lines[m.queryIdx].line3d.A.z,
                newer_node->lines[m.queryIdx].line3d.B.x, newer_node->lines[m.queryIdx].line3d.B.y, newer_node->lines[m.queryIdx].line3d.B.z;
    v->setEstimate(line_new); // line represented wrt the newer node (second, fixed one)
    v->setId(v_id++);
    v->setFixed(false);
    optimizer->addVertex(v);

    //////// create edges between line and cams    
    //// to newer node
    
    g2o::EdgeSE3LineEndpts * e_newer = new g2o::EdgeSE3LineEndpts();
    e_newer->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.second); 
    e_newer->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
    e_newer->setMeasurement(line_new);
    e_newer->information() = Matrix6d::Identity() * sysPara.g2o_line_error_weight;  // must be identity!
    cv::Mat covA = newer_node->lines[m.queryIdx].line3d.rndA.cov,
            covB = newer_node->lines[m.queryIdx].line3d.rndB.cov;            
    e_newer->endptCov = Matrix6d::Identity();
    for(int ii=0; ii<3; ++ii) {
      for(int jj=0; jj<3; ++jj) {
        e_newer->endptCov(ii,jj) = covA.at<double>(ii,jj);
        e_newer->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
      }
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_newer->endptCov.block<3,3>(0,0),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_newer->endpt_AffnMat.block<3,3>(0,0) = am;
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_newer->endptCov.block<3,3>(3,3),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_newer->endpt_AffnMat.block<3,3>(3,3) = am;
    }

    if(sysPara.g2o_BA_use_kernel) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sysPara.g2o_BA_kernel_delta);
      e_newer->setRobustKernel(rk);
    }
    e_newer->setParameterId(0,1);// param id 0 of the edge corresponds to param id 1 of the optimizer     
    optimizer->addEdge(e_newer);    
    
    //// edge to older node
    
    g2o::EdgeSE3LineEndpts* e_older = new g2o::EdgeSE3LineEndpts();
    e_older->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cams.first); 
    e_older->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v);
    Eigen::Vector6d line_older;
    line_older << earlier_node->lines[m.trainIdx].line3d.A.x, earlier_node->lines[m.trainIdx].line3d.A.y, earlier_node->lines[m.trainIdx].line3d.A.z,
                  earlier_node->lines[m.trainIdx].line3d.B.x, earlier_node->lines[m.trainIdx].line3d.B.y, earlier_node->lines[m.trainIdx].line3d.B.z;
    e_older->setMeasurement(line_older);
    e_older->information() = Eigen::Matrix6d::Identity() * sysPara.g2o_line_error_weight;  // must be identity!
    covA = earlier_node->lines[m.trainIdx].line3d.rndA.cov;            
    covB = earlier_node->lines[m.trainIdx].line3d.rndB.cov;
    e_older->endptCov = Eigen::Matrix6d::Identity() ;
    for(int ii=0; ii<3; ++ii) {
      for(int jj=0; jj<3; ++jj) {
        e_older->endptCov(ii,jj) = covA.at<double>(ii,jj);
        e_older->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
      }
    }

    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_older->endptCov.block<3,3>(0,0),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_older->endpt_AffnMat.block<3,3>(0,0) = am;
    }
    {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(e_older->endptCov.block<3,3>(3,3),Eigen::ComputeFullU);
      Eigen::Matrix3d D_invsqrt;
      D_invsqrt.fill(0);
      D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
      D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
      D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
      Eigen::Matrix3d am = D_invsqrt * svd.matrixU().transpose();
      e_older->endpt_AffnMat.block<3,3>(3,3) = am;
    }


    if(sysPara.g2o_BA_use_kernel) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sysPara.g2o_BA_kernel_delta);
      e_older->setRobustKernel(rk);
    }
    e_older->setParameterId(0,1);
    optimizer->addEdge(e_older); 

    ln_edges.push_back(e_newer);
    ln_edges.push_back(e_older); 
  }

  optimizer->initializeOptimization();

  double pterr = 0, lnerr = 0;
  for(int i=0; i<pt_edges.size();++i) {
    pt_edges[i]->computeError();
    pterr += pt_edges[i]->chi2();
  }
  for(int i=0; i<ln_edges.size();++i) {
    ln_edges[i]->computeError();
    lnerr += ln_edges[i]->chi2();
  }
//  cout<<"initial "<<pterr<<" + "<<lnerr<<" = "<<pterr+lnerr<<endl;

  optimizer->optimize(iterations);

  pterr = 0, lnerr = 0;
  for(int i=0; i<pt_edges.size();++i) {
    pt_edges[i]->computeError();
    pterr += pt_edges[i]->chi2();
  }
  for(int i=0; i<ln_edges.size();++i) {
    ln_edges[i]->computeError();
    lnerr += ln_edges[i]->chi2();
  }
// cout<<pterr<<" \t "<<lnerr<<" \t "<<lnerr/(pterr+lnerr)<<endl;

  transformation_estimate = cams.first->estimate().cast<float>().inverse().matrix();
  delete optimizer;
}  

#endif
