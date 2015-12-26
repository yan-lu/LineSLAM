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


#ifndef G2O_EDGE_SE3_PTXYZ_H_
#define G2O_EDGE_SE3_PTXYZ_H_
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/parameter_camera.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
namespace g2o {
/*! \class EdgeProjectDepth
* \brief 
*/
using namespace Eigen;
class G2O_TYPES_SLAM3D_API EdgeSE3PointXYZ : public BaseBinaryEdge<3, Vector3d, VertexSE3, VertexPointXYZ> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgeSE3PointXYZ();
	virtual bool read(std::istream& is);
	virtual bool write(std::ostream& os) const;
	// return the error estimate as a 3-vector
	void computeError();
	// jacobian
	virtual void setMeasurement(const Vector3d& m){
		_measurement = m;
	}
	virtual bool setMeasurementData(const double* d){
		Eigen::Map<const Vector3d> v(d);
		_measurement = v;
		return true;
	}
	virtual bool getMeasurementData(double* d) const{
		Eigen::Map<Vector3d> v(d);
		v=_measurement;
		return true;
	}
	virtual int measurementDimension() const {return 3;}
	virtual bool setMeasurementFromState() ;
	virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& from,
			OptimizableGraph::Vertex* to) {
		(void) to;
		return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
	}
	virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);
private:
	Eigen::Matrix<double,3,9,Eigen::ColMajor> J; // jacobian before projection
	virtual bool resolveCaches();
	//ParameterCamera* params;
	CacheSE3Offset* cache;
	ParameterSE3Offset* offsetParam;
};
}
#endif
