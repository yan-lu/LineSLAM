cmake_minimum_required(VERSION 2.4.6)
#########################################################
### USER CONFIG #########################################
#########################################################
set(USE_SIFT_GPU        1) 
set(ENV{SIFT_GPU_MODE}	2) #CUDA = 1, GLSL = 2
set(ENV{SIFT_GPU_CUDA_PATH}	/usr/local/cuda)	
set(USE_GICP_BIN		0)
set(USE_GICP_CODE		0)
set(USE_PCL_ICP		  1)
set(USE_GL2PS       0) #For printing the 3D Window to PostScript - requires installation of additional library: libgl2ps-dev
 
set(USE_LINES 		1) ## extract 2d/3d line segments from each rgbd image
set(ALLOW_LOAD_RAWDATA  1)
set(MODEL_ASYNCH	0) ## model depth noise caused by asynchronization
set(USE_MY_SE3_PTXYZ_EDGE 1) ## use our own pt cov model in motion estimation

IF (${USE_LINES})
   add_definitions(-DUSE_LINES) 		
ENDIF(${USE_LINES}) 


IF (${ALLOW_LOAD_RAWDATA})
   add_definitions(-DALLOW_LOAD_RAWDATA)
ENDIF(${ALLOW_LOAD_RAWDATA})

IF (${MODEL_ASYNCH})
   add_definitions(-DMODEL_ASYNCH)
ENDIF (${MODEL_ASYNCH})

IF (${USE_MY_SE3_PTXYZ_EDGE})
   add_definitions(-DUSE_MY_SE3_PTXYZ_EDGE)
ENDIF(${USE_MY_SE3_PTXYZ_EDGE})


IF ("${ROS_PARALLEL_JOBS}" EQUAL "")
  set(ROS_PARALLEL_JOBS 1)
endif ("${ROS_PARALLEL_JOBS}" EQUAL "")
#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#SET(CMAKE_CXX_FLAGS "-O3 -fPIC -std=c++0x")
#For annotated profiling with perf: 
SET(CMAKE_CXX_FLAGS "-ggdb -O3 -fPIC -std=c++0x -march=corei7-avx")

IF (${USE_GL2PS})
  add_definitions(-DGL2PS)
ENDIF (${USE_GL2PS})
#is used for compiling the libraries 
set(USE_GICP 0)
IF (${USE_GICP_BIN} OR ${USE_GICP_CODE})
	set(USE_GICP 1)
ENDIF (${USE_GICP_BIN} OR ${USE_GICP_CODE})
#########################################################
#########################################################


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
####################################################
# ROS
####################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/")
#message(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
find_package(catkin REQUIRED COMPONENTS pcl_conversions pcl_ros roscpp image_transport cv_bridge sensor_msgs 
             geometry_msgs visualization_msgs std_msgs tf message_generation message_filters rosbag rosconsole)

project(lineslam)

#add_definitions(-DROSCONSOLE_SEVERITY_INFO)

add_service_files(DIRECTORY srv FILES lineslam_ros_ui.srv lineslam_ros_ui_b.srv lineslam_ros_ui_f.srv lineslam_ros_ui_s.srv) 
generate_messages()
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_pcl_tutorial
  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs 
  DEPENDS message_runtime std_msgs 
)
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})
link_libraries(lineslam ${catkin_LIBRARIES})

####################################################
# Required 3rd party
####################################################
# PCL #######################
find_package(PCL 1.7 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Octomap ###################
find_package(octomap 1.6 REQUIRED)
include_directories(BEFORE ${OCTOMAP_INCLUDE_DIRS})
link_directories(${OCTOMAP_LIBRARY_DIRS})
link_libraries(${OCTOMAP_LIBRARIES})

# g2o #######################
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})
link_libraries(${G2O_LIBRARIES})
message("G2o-libraries " ${G2O_LIBRARIES})

# Qt ########################
# http://qtnode.net/wiki/Qt4_with_cmake
find_package(Qt4 REQUIRED)
SET(QT_USE_QTXML TRUE)
SET(QT_USE_QTOPENGL TRUE)
ADD_DEFINITIONS(-DQT_NO_KEYWORDS)
include(${QT_USE_FILE})

# OpenMP ########################
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
# This is necessary as all ui files etc will get dumped in the bottom of then binary directory. 
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${QT_QTOPENGL_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/external/gicp/ann_1.1.2/include/ANN)

 	
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})


SET(${CMAKE_LIBRARY_PATH} ${LIBRARY_OUTPUT_PATH})
#SET(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE TRUE)


#Only possible when not subscribing to point cloud topic. Possibly dangerous anyway, but halves storage space per point cloud
#ADD_DEFINITIONS(-DRGB_IS_4TH_DIM) 
#ADD_DEFINITIONS(-DDO_FEATURE_OPTIMIZATION)
#ADD_DEFINITIONS(-DHEMACLOUDS) 
ADD_DEFINITIONS(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO)

#############################
# SiftGPU ###################
#############################
IF (${USE_SIFT_GPU})
  add_subdirectory(external)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/SiftGPU/src)
  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/SiftGPU/bin)	
	
	
	add_definitions(-DUSE_SIFT_GPU)
	add_definitions(-DSIFT_GPU_MODE=$ENV{SIFT_GPU_MODE})
ENDIF (${USE_SIFT_GPU})

############################
# LevMar ###################
############################
IF (${USE_LINES})
add_subdirectory(external)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/levmar-2.6)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/levmar-2.6)

SET(Line_LIBS levmar )
ENDIF (${USE_LINES})

############################
# EDLines ##################
############################
IF (${USE_LINES})
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/EDLines) 
SET(Line_LIBS ${Line_LIBS} EDLines)
ENDIF (${USE_LINES})

############################
# Armadillo#################
############################
IF (${USE_LINES})
SET(Line_LIBS ${Line_LIBS} openblas armadillo) #armadillo needs openblas
ENDIF (${USE_LINES})


#############################
# GICP ######################
#############################
IF (${USE_GICP})
	SET(gicp_build_path ${CMAKE_CURRENT_SOURCE_DIR}/external/gicp/ann_1.1.2)
  FILE(MAKE_DIRECTORY ${gicp_build_path}/lib/)
  MESSAGE("\n------------------------------------------------------------------\n")
	MESSAGE("Compiling GICP...")
	EXECUTE_PROCESS(
	    COMMAND make ${ROS_PARALLEL_JOBS} linux-g++
	    WORKING_DIRECTORY ${gicp_build_path}
	    RESULT_VARIABLE gicp_return
    )
    MESSAGE("\n------------------------------------------------------------------\n")
	
	#on error
	IF (NOT ${gicp_return} EQUAL 0)
		MESSAGE(FATAL_ERROR "GICP cannot be compiled. Returned: ${gicp_return}")
	ENDIF (NOT ${gicp_return} EQUAL 0)
	
	FILE(COPY ${gicp_build_path}/lib/libANN.a DESTINATION ${LIBRARY_OUTPUT_PATH})
ENDIF (${USE_GICP})

IF (${USE_GICP_BIN})
	add_definitions(-DUSE_ICP_BIN)
ENDIF (${USE_GICP_BIN})

IF (${USE_GICP_CODE})
	add_definitions(-DUSE_ICP_CODE) 
ENDIF (${USE_GICP_CODE})
IF (${USE_PCL_ICP})
  add_definitions(-DUSE_PCL_ICP) 
ENDIF (${USE_PCL_ICP})



#############################
# OpenCV ####################
#############################
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

#############################
# Eigen ####################
#############################
find_package(Eigen)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

##############################################################################
# Some QT Magic (MOC Creation)
##############################################################################

file(GLOB QT_FORMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ui/*.ui)
file(GLOB QT_RESOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} resources/*.qrc)
file(GLOB_RECURSE QT_MOC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS include/local/*.hpp src/qtros.h src/openni_listener.h src/qt_gui.h src/graph_manager.h src/glviewer.h src/ros_service_ui.h)

QT4_ADD_RESOURCES(QT_RESOURCES_CPP ${QT_RESOURCES})
QT4_WRAP_UI(QT_FORMS_HPP ${QT_FORMS})
QT4_WRAP_CPP(QT_MOC_HPP ${QT_MOC})  

##############################################################################
# Sources to Compile
##############################################################################
SET(ADDITIONAL_SOURCES src/gicp-fallback.cpp src/main.cpp src/qtros.cpp  src/openni_listener.cpp src/qt_gui.cpp src/node.cpp src/glviewer.cpp src/parameter_server.cpp src/ros_service_ui.cpp src/misc.cpp src/landmark.cpp src/loop_closing.cpp src/ColorOctomapServer.cpp src/scoped_timer.cpp src/icp.cpp src/matching_result.cpp)
#GraphManager Files
SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} src/graph_manager.cpp src/graph_mgr_io.cpp src/graph_manager2.cpp)
SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} src/transformation_estimation_euclidean.cpp src/transformation_estimation.cpp src/graph_manager2.cpp src/header.cpp)
#Computation of meaningful covariances
SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} src/covariance_estimation.cpp) # src/edge_labeler.cpp)
#Detector Threshold Adapter
SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} src/feature_adjuster.cpp src/aorb.cpp src/features.cpp)

IF (${USE_LINES})
#Line segment detectors 
SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} external/lsd/lsd.cpp)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/lsd)

SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} src/line/lineslam.cpp src/line/motion.cpp src/line/utils.cpp src/line/utils.h src/line/edge_se3_lineendpts.cpp src/line/vertex_lineendpts.cpp src/line/five_point.cpp src/line/essential_mat.cpp src/line/edge_se3_ptxyz.cpp)
ENDIF (${USE_LINES})



IF (${USE_SIFT_GPU})
 	SET(ADDITIONAL_SOURCES ${ADDITIONAL_SOURCES} src/sift_gpu_wrapper.cpp)
ENDIF (${USE_SIFT_GPU})

file(GLOB_RECURSE QT_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS)

include_directories(external/ src/line/ /usr/include/suitesparse/)

##############################################################################
# Binaries
##############################################################################
IF (${USE_GICP})
#  set(ROS_COMPILE_FLAGS ${ROS_COMPILE_FLAGS} -fpermissive)
#  add_library(gicp external/gicp/bfgs_funcs.cpp external/gicp/gicp.cpp external/gicp/optimize.cpp external/gicp/scan.cpp external/gicp/transform.cpp)
ENDIF (${USE_GICP})

add_executable(lineslam ${QT_SOURCES} ${QT_RESOURCES_CPP} ${QT_FORMS_HPP} ${QT_MOC_HPP} ${ADDITIONAL_SOURCES})
add_dependencies(lineslam lineslam_gencpp)




#Use specific version of cxsparse, to avoid linker warning about possible conflicts between versions 2.2.3 (linked from g2o) and 3.1.0
SET(LIBS_LINK ${Line_LIBS}  GL GLU cholmod /usr/lib/libcxsparse.so.2.2.3 ${G2O_LIBRARIES} ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${OpenCV_LIBS})
#-lboost_signals -lrt -loctomap -loctomap_server -loctomath)

IF (${USE_SIFT_GPU})
 	SET(LIBS_LINK ${LIBS_LINK} siftgpu)
ENDIF (${USE_SIFT_GPU})
IF (${USE_GL2PS})
 	SET(LIBS_LINK ${LIBS_LINK} -lgl2ps)
ENDIF (${USE_GL2PS})
IF (${USE_GICP})
 	SET(LIBS_LINK ${LIBS_LINK} gicp ANN gsl gslcblas)
ENDIF (${USE_GICP})
#link libraries
target_link_libraries(lineslam ${LIBS_LINK}  ${catkin_LIBRARIES})














###############################################################################
## Bag Viewer
###############################################################################
#
#file(GLOB QT_FORMS2 RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ui/*.ui)
#file(GLOB QT_RESOURCES2 RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} resources/*.qrc)
#file(GLOB_RECURSE QT_MOC2 RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS include/local/*.hpp src/qtros.h src/bagloader.h src/qt_gui.h src/glviewer.h)
#
#QT4_ADD_RESOURCES(QT_RESOURCES_CPP2 ${QT_RESOURCES2})
#QT4_WRAP_UI(QT_FORMS_HPP2 ${QT_FORMS2})
#QT4_WRAP_CPP(QT_MOC_HPP2 ${QT_MOC2})  
#
#
#SET(ADDITIONAL_SOURCES2 src/result-bag-viewer.cpp src/bagloader.cpp src/qtros.cpp  src/qt_gui.cpp src/glviewer.cpp src/parameter_server.cpp src/misc.cpp src/ColorOctomapServer.cpp src/scoped_timer.cpp)
#add_executable(bagviewer ${QT_SOURCES2} ${QT_RESOURCES_CPP2} ${QT_FORMS_HPP2} ${QT_MOC_HPP2} ${ADDITIONAL_SOURCES2})
#target_link_libraries(bagviewer ${LIBS_LINK})
# 
