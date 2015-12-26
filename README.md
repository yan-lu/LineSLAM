##LineSLAM

Line feature based RGBD SLAM, supporting fusion with point features, developed based on [RGBDSLAM v2](http://felixendres.github.io/rgbdslam_v2/). 

===============
###Dependencies
1. ROS hydro
2. OpenCV 2.4.X
3. [OpenBLAS](http://www.openblas.net/) (build from source)
4. [Armadillo](http://arma.sourceforge.net/)
5. Eigen3
6. [LSD](http://www.ipol.im/pub/art/2012/gjmr-lsd/), [EDLines](http://ceng.anadolu.edu.tr/CV/downloads/downloads.aspx), [levmar](http://users.ics.forth.gr/~lourakis/levmar/) : included in this repo.

===============
###Installation
Refer to http://felixendres.github.io/rgbdslam_v2/

========
###Usage
roslaunch lineslam lineslam.launch

============
###Reference
Yan Lu and Dezhen Song, "Robust RGB-D Odometry Using Point and Line Features" , *IEEE International Conference on Computer Vision (ICCV)*, Santiago, Chile, Dec. 13-16, 2015
