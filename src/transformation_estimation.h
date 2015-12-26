#ifndef RGBD_SLAM_TRAFO_EST_H_
#define RGBD_SLAM_TRAFO_EST_H_

#include <Eigen/Core>
#include <opencv2/features2d/features2d.hpp>

#ifdef USE_LINES
#include "line/edge_se3_lineendpts.h"
#endif

class Node; //Fwd declaration

//!Do sparse bundle adjustment for the node pair
void getTransformFromMatchesG2O(const Node* earlier_node,
                                const Node* newer_node,
                                const std::vector<cv::DMatch> & matches,
                                Eigen::Matrix4f& transformation_estimate, //Input (initial guess) and Output
                                int iterations = 10);

#ifdef USE_LINES
void getTransformFromHybridMatchesG2O (const Node* earlier_node,
                                	   const Node* newer_node,
                                	   const std::vector<cv::DMatch> & pt_matches,
                                	   const std::vector<cv::DMatch> & ln_matches,
                                	   Eigen::Matrix4f& transformation_estimate, //Input (initial guess) and Output
                                	   int iterations = 10);
#endif

#endif
