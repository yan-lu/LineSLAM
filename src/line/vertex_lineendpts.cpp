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

#include "vertex_lineendpts.h"
#include <stdio.h>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <typeinfo>

namespace g2o {

  bool VertexLineEndpts::read(std::istream& is) {
    Vector6d lv;
    for (int i=0; i<estimateDimension(); i++)
      is >> lv[i];
    setEstimate(lv);
    return true;
  }

  bool VertexLineEndpts::write(std::ostream& os) const {
    Vector6d lv=estimate();
    for (int i=0; i<estimateDimension(); i++){
      os << lv[i] << " ";
    }
    return os.good();
  }


#ifdef G2O_HAVE_OPENGL
   VertexLineEndptsDrawAction::VertexLineEndptsDrawAction(): DrawAction(typeid(VertexLineEndpts).name()){
  }

  bool VertexLineEndptsDrawAction::refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_){
    if (! DrawAction::refreshPropertyPtrs(params_))
      return false;
    if (_previousParams){
      _pointSize = _previousParams->makeProperty<FloatProperty>(_typeName + "::POINT_SIZE", 1.);
    } else {
      _pointSize = 0;
    }
    return true;
  }


  HyperGraphElementAction* VertexLineEndptsDrawAction::operator()(HyperGraph::HyperGraphElement* element, 
                     HyperGraphElementAction::Parameters* params ){

    if (typeid(*element).name()!=_typeName)
      return 0;
    refreshPropertyPtrs(params);
    if (! _previousParams)
      return this;
    
    if (_show && !_show->value())
      return this;
    VertexLineEndpts* that = static_cast<VertexLineEndpts*>(element);
    

    glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT);
    glDisable(GL_LIGHTING);
    glColor3f(0.8f,0.5f,0.3f);
    if (_pointSize) {
      glPointSize(_pointSize->value());
    }
//    glBegin(GL_POINTS);
	glBegin(GL_LINES);
    glVertex3f((float)that->estimate()(0),(float)that->estimate()(1),(float)that->estimate()(2));
	glVertex3f((float)that->estimate()(3),(float)that->estimate()(4),(float)that->estimate()(5));
    glEnd();
    glPopAttrib();

    return this;
  }
#endif

   VertexLineEndptsWriteGnuplotAction:: VertexLineEndptsWriteGnuplotAction() :
    WriteGnuplotAction(typeid( VertexLineEndpts).name())
  {
  }

  HyperGraphElementAction* VertexLineEndptsWriteGnuplotAction::operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ )
  {
    if (typeid(*element).name()!=_typeName)
      return 0;
    WriteGnuplotAction::Parameters* params=static_cast<WriteGnuplotAction::Parameters*>(params_);
    if (!params->os){
      std::cerr << __PRETTY_FUNCTION__ << ": warning, no valid os specified" << std::endl;
      return 0;
    }

    VertexLineEndpts* v = static_cast<VertexLineEndpts*>(element);
    *(params->os) << v->estimate().x() << " " << v->estimate().y() << " " << v->estimate().z() << " " << std::endl;
    return this;
  }

}
