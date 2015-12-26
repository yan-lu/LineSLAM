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


#include "edge_se3_ptxyz.h"
namespace g2o {
using namespace g2o;
// point to camera projection, monocular
EdgeSE3PointXYZ::EdgeSE3PointXYZ() : BaseBinaryEdge<3, Vector3d, VertexSE3, VertexPointXYZ>() {
	resizeParameters(1);
	installParameter(offsetParam, 0, 0);  
	information().setIdentity();
	cache = 0;
	offsetParam = 0;
	J.fill(0);
	J.block<3,3>(0,0) = -Matrix3d::Identity();
}
bool EdgeSE3PointXYZ::resolveCaches(){
/*	ParameterVector pv(1);
	pv[0]=params;
	resolveCache(cache, (OptimizableGraph::Vertex*)_vertices[0],"CACHE_CAMERA",pv);
	return cache != 0;
*/
	ParameterVector pv(1);
		pv[0]=offsetParam;
		resolveCache(cache, (OptimizableGraph::Vertex*)_vertices[0],"CACHE_SE3_OFFSET",pv);
		return cache != 0;
}
bool EdgeSE3PointXYZ::read(std::istream& is) {
	int pid;
	is >> pid;
	setParameterId(0,pid);
	// measured keypoint
	Vector3d meas;
	for (int i=0; i<3; i++) is >> meas[i];
	setMeasurement(meas);
	// don't need this if we don't use it in error calculation (???)
	// information matrix is the identity for features, could be changed to allow arbitrary covariances
	if (is.bad()) {
		return false;
	}
	for ( int i=0; i<information().rows() && is.good(); i++)
		for (int j=i; j<information().cols() && is.good(); j++){
			is >> information()(i,j);
			if (i!=j)
			information()(j,i)=information()(i,j);
		}
	if (is.bad()) {
	// we overwrite the information matrix
		information().setIdentity();
	}
	return true;
}

bool EdgeSE3PointXYZ::write(std::ostream& os) const {
	os << offsetParam->id() << " ";
	for (int i=0; i<3; i++) os << measurement()[i] << " ";
	for (int i=0; i<information().rows(); i++)
	for (int j=i; j<information().cols(); j++) {
		os << information()(i,j) << " ";
	}
	return os.good();
}

void EdgeSE3PointXYZ::computeError() {
// from cam to point (track)
//VertexSE3 *cam = static_cast<VertexSE3*>(_vertices[0]);
	VertexPointXYZ *point = static_cast<VertexPointXYZ*>(_vertices[1]);
	Vector3d p = cache->w2n() * point->estimate(); // point transformed to cam perspective	
	_error = p - _measurement;
}


bool EdgeSE3PointXYZ::setMeasurementFromState(){
//VertexSE3 *cam = static_cast<VertexSE3*>(_vertices[0]);
	VertexPointXYZ *point = static_cast<VertexPointXYZ*>(_vertices[1]);
// calculate the projection
	const Vector3d& pt = point->estimate();
	Vector3d p = cache->w2n() * pt;
	_measurement = p;
	return true;
}
void EdgeSE3PointXYZ::initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* /*to_*/)
{
	(void) from;
	assert(from.size() == 1 && from.count(_vertices[0]) == 1 && "Can not initialize VertexDepthCam position by VertexTrackXYZ");
	VertexSE3 *cam = dynamic_cast<VertexSE3*>(_vertices[0]);
	VertexPointXYZ *point = dynamic_cast<VertexPointXYZ*>(_vertices[1]);
	
}
}
