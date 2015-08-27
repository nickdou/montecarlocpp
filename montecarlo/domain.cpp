//
//  domain.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/5/15.
//
//

#include "field.h"
#include "domain.h"
#include "subdomain.h"
#include "boundary.h"
#include "constants.h"
#include <Eigen/Core>
#include <boost/fusion/algorithm/iteration.hpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

namespace fusion = boost::fusion;

Domain::Domain()
{}

Domain::Domain(const Vector3d& gradT)
: gradT_(gradT)
{}

Domain::~Domain()
{}

bool Domain::isInit() const
{
    if (sdomPtrs_.size() == 0) return false;
    for (Subdomain::Pointers::const_iterator s = sdomPtrs_.begin();
         s != sdomPtrs_.end(); ++s)
    {
        if (!(*s)->isInit()) return false;
    }
    return true;
}

bool Domain::isInside(const Vector3d& pos) const
{
    return locate(pos);
}

const Subdomain* Domain::locate(const Vector3d& pos) const
{
    for (Subdomain::Pointers::const_iterator s = sdomPtrs_.begin();
         s != sdomPtrs_.end(); ++s)
    {
        if ((*s)->isInside(pos)) return *s;
    }
    return 0;
}

const Subdomain::Pointers& Domain::sdomPtrs() const
{
    return sdomPtrs_;
}

const Emitter::Pointers& Domain::emitPtrs() const
{
    return emitPtrs_;
}

Vector3d Domain::gradT() const
{
    return gradT_;
}

Domain::Matrix3Xd Domain::checkpoints() const
{
    return checkpoints_;
}

void Domain::checkpoints(const Matrix3Xd& pts)
{
    checkpoints_ = pts;
}

void Domain::info(const std::string& str)
{
    info_ = str;
}

std::ostream& operator<<(std::ostream& os, const Domain& dom)
{
    return os << dom.info_;
}

void Domain::addSdom(const Subdomain* sdom)
{
    sdomPtrs_.push_back(sdom);
    emitPtrs_.insert(emitPtrs_.end(), sdom->emitPtrs().begin(),
                     sdom->emitPtrs().end());
}
void Domain::addSdom(const EmitSubdomain* sdom)
{
    if (sdom->emitWeight() != 0.)
    {
        emitPtrs_.push_back( static_cast<const Emitter*>(sdom) );
    }
    addSdom( static_cast<const Subdomain*>(sdom) );
}

Domain::AddSdomF::AddSdomF(Domain* dom)
: dom_(dom)
{}

template<typename S>
void Domain::AddSdomF::operator()(const S& sdom) const
{
    dom_->addSdom(&sdom);
}

template void Domain::AddSdomF::operator()(const Subdomain& sdom) const;
template void Domain::AddSdomF::operator()(const EmitSubdomain& sdom) const;

BulkDomain::BulkDomain()
: Domain()
{}

BulkDomain::BulkDomain(const Vector3d& corner, const Vector3l& div,
                       double deltaT)
: Domain(Vector3d(-deltaT/corner(0), 0., 0.)),
sdom_(Vector3d::Zero(), corner.asDiagonal(), div,
      Vector3d(-deltaT/corner(0), 0., 0.))
{
    checkpoints(0.5*corner);
    
    std::ostringstream ss;
    ss << "BulkDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << corner.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(corner(0), 0., 0.));
    addSdom(&sdom_);
}

FilmDomain::FilmDomain()
: Domain()
{}

FilmDomain::FilmDomain(const Vector3d& corner, const Vector3l& div,
                       double deltaT)
: Domain(Vector3d(-deltaT/corner(0), 0., 0.)),
sdom_(Vector3d::Zero(), corner.asDiagonal(), div,
      Vector3d(-deltaT/corner(0), 0., 0.))
{
    checkpoints(0.5*corner);
    
    std::ostringstream ss;
    ss << "FilmDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << corner.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(corner(0), 0., 0.));
    addSdom(&sdom_);
}

HexDomain::HexDomain()
: Domain()
{}

HexDomain::HexDomain(const Eigen::Matrix<double, 4, 1>& dim, double deltaT)
: Domain(Vector3d(-deltaT/dim(0), 0., 0.))
{
    Eigen::Matrix<double, 6, 3> mat;
    mat << dim(0),        0.,            0.,
               0.,    dim(1),       -dim(3),
               0., 2.*dim(1),            0.,
               0., 2.*dim(1),        dim(2),
               0.,    dim(1), dim(2)+dim(3),
               0.,        0.,        dim(2);
    sdom_ = Sdom(Vector3d::Zero(), mat.transpose(), 0l,
                 Vector3d(-deltaT/dim(0), 0., 0.));
    
    std::ostringstream ss;
    ss << "HexDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(2, 3);
    pts << 0.5*dim(0), 0.5*dim(1), 0.5*dim(2),
           0.5*dim(0), 1.5*dim(1), 0.5*dim(2);
    checkpoints(pts.transpose());
    
    makePair(sdom_.bdry<0>(), sdom_.bdry<1>(), Vector3d(dim(0), 0., 0.));
    addSdom(&sdom_);
}

PyrDomain::PyrDomain()
: Domain()
{}

PyrDomain::PyrDomain(const Eigen::Matrix<double, 3, 1>& dim, double deltaT)
: Domain(Vector3d(-deltaT/dim(0), 0., 0.))
{
    Eigen::Matrix<double, 4, 3> mat;
    mat << dim(0), 0.5*dim(1), 0.5*dim(2),
               0.,     dim(1),         0.,
               0.,     dim(1),     dim(2),
               0.,         0.,     dim(2);
    sdom_ = Sdom(Vector3d::Zero(), mat.transpose(), 0l,
                 Vector3d(-deltaT/dim(0), 0., 0.));
    
    std::ostringstream ss;
    ss << "PyrDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(1, 3);
    pts << 0.5*dim(0), 0.5*dim(1), 0.5*dim(2);
    checkpoints(pts.transpose());
    
    addSdom(&sdom_);
}

JctDomain::JctDomain()
: Domain()
{}

JctDomain::JctDomain(const Eigen::Matrix<double, 4, 1>& dim,
                     const Eigen::Matrix<long, 4, 1>& div,
                     double deltaT)
: Domain(Vector3d(-deltaT/(2*dim(0)), 0., 0.)),
sdomCont_(Sdom<0>::type(Vector3d(0., 0., 0.),
                        Diagonal(2*dim(0), dim(1), dim(3)),
                        Vector3l(2*div(0), div(1), div(3)),
                        Vector3d(-deltaT/(2*dim(0)), 0., 0.)),
          Sdom<1>::type(Vector3d(0., dim(1), 0.),
                        Diagonal(dim(0), dim(2), dim(3)),
                        Vector3l(div(0), div(2), div(3)),
                        Vector3d(-deltaT/(2*dim(0)), 0., 0.)),
          Sdom<2>::type(Vector3d(dim(0), dim(1), 0.),
                        Diagonal(dim(0), dim(2), dim(3)),
                        Vector3l(div(0), div(2), div(3)),
                        Vector3d(-deltaT/(2*dim(0)), 0., 0.)))
{
    std::ostringstream ss;
    ss << "JctDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(3, 3);
    pts <<     dim(0),        0.5*dim(1), 0.5*dim(3),
           0.5*dim(0), dim(1)+0.5*dim(2), 0.5*dim(3),
           1.5*dim(0), dim(1)+0.5*dim(2), 0.5*dim(3);
    checkpoints(pts.transpose());
    
    makePair(sdom<0>().bdry<4>(), sdom<1>().bdry<1>());
    makePair(sdom<0>().bdry<4>(), sdom<2>().bdry<1>());
    makePair(sdom<1>().bdry<3>(), sdom<2>().bdry<0>());
    
    Vector3d transl(2*dim(0), 0., 0.);
    makePair(sdom<0>().bdry<0>(), sdom<0>().bdry<3>(), transl);
    makePair(sdom<1>().bdry<0>(), sdom<2>().bdry<3>(), transl);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

TeeDomain::TeeDomain()
: Domain()
{}

TeeDomain::TeeDomain(const Eigen::Matrix<double, 5, 1>& dim,
                     const Eigen::Matrix<long, 5, 1>& div,
                     double deltaT)
: Domain(Vector3d(-deltaT/(2.*dim(0) + dim(1)), 0., 0.)),
sdomCont_(Sdom<0>::type(Vector3d::Zero(),
                        Diagonal(dim(0), dim(2), dim(4)),
                        Vector3l(div(0), div(2), div(4)),
                        Vector3d(-deltaT/(2.*dim(0) + dim(1)), 0., 0.)),
          Sdom<1>::type(Vector3d(dim(0), 0., 0.),
                        Diagonal(dim(1), dim(2), dim(4)),
                        Vector3l(div(1), div(2), div(4)),
                        Vector3d(-deltaT/(2.*dim(0) + dim(1)), 0., 0.)),
          Sdom<2>::type(Vector3d(dim(0), dim(2), 0.),
                        Diagonal(dim(1), dim(3), dim(4)),
                        Vector3l(div(1), div(3), div(4)),
                        Vector3d(-deltaT/(2.*dim(0) + dim(1)), 0., 0.)),
          Sdom<3>::type(Vector3d(dim(0) + dim(1), 0., 0.),
                        Diagonal(dim(0), dim(2), dim(4)),
                        Vector3l(div(0), div(2), div(4)),
                        Vector3d(-deltaT/(2.*dim(0) + dim(1)), 0., 0.)))
{
    std::ostringstream ss;
    ss << "TeeDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(4, 3);
    pts <<        0.5*dim(0),        0.5*dim(2), 0.5*dim(4),
           dim(0)+0.5*dim(1),        0.5*dim(2), 0.5*dim(4),
           dim(0)+0.5*dim(1), dim(2)+0.5*dim(3), 0.5*dim(4),
           1.5*dim(0)+dim(1),        0.5*dim(2), 0.5*dim(4);
    
    
    makePair(sdom<0>().bdry<3>(), sdom<1>().bdry<0>());
    makePair(sdom<1>().bdry<4>(), sdom<2>().bdry<1>());
    makePair(sdom<1>().bdry<3>(), sdom<3>().bdry<0>());
    makePair(sdom<0>().bdry<0>(), sdom<3>().bdry<3>(),
             Vector3d(2.*dim(0) + dim(1), 0., 0.));
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

TubeDomain::TubeDomain()
: Domain()
{}

TubeDomain::TubeDomain(const Eigen::Matrix<double, 4, 1>& dim,
                       const Eigen::Matrix<long, 4, 1>& div,
                       double deltaT)
: Domain(Vector3d(-deltaT/dim(0), 0., 0.)),
sdomCont_(Sdom<0>::type(Vector3d(0., dim(1), 0.),
                        Diagonal(dim(0), dim(3), dim(2)),
                        Vector3l(div(0), div(3), div(2)),
                        Vector3d(-deltaT/dim(0), 0., 0.)),
          Sdom<1>::type(Vector3d(0., dim(1), dim(2)),
                        Diagonal(dim(0), dim(3), dim(3)),
                        Vector3l(div(0), div(3), div(3)),
                        Vector3d(-deltaT/dim(0), 0., 0.)),
          Sdom<2>::type(Vector3d(0., 0., dim(2)),
                        Diagonal(dim(0), dim(1), dim(3)),
                        Vector3l(div(0), div(1), div(3)),
                        Vector3d(-deltaT/dim(0), 0., 0.)))
{
    std::ostringstream ss;
    ss << "TubeDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(3, 3);
    pts << 0.5*dim(0), dim(1)+0.5*dim(3),        0.5*dim(2),
           0.5*dim(0), dim(1)+0.5*dim(3), dim(2)+0.5*dim(3),
           0.5*dim(0),        0.5*dim(1), dim(2)+0.5*dim(3);
    checkpoints(pts.transpose());
    
    makePair(sdom<1>().bdry<1>(), sdom<2>().bdry<4>());
    makePair(sdom<1>().bdry<2>(), sdom<0>().bdry<5>());
    
    Vector3d transl(dim(0), 0., 0.);
    makePair(sdom<0>().bdry<0>(), sdom<0>().bdry<3>(), transl);
    makePair(sdom<1>().bdry<0>(), sdom<1>().bdry<3>(), transl);
    makePair(sdom<2>().bdry<0>(), sdom<2>().bdry<3>(), transl);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

OctetDomain::OctetDomain()
: Domain()
{}

OctetDomain::OctetDomain(const Eigen::Matrix<double, 4, 1>& dim,
                         const Eigen::Matrix<long, 4, 1>& div,
                         double deltaT)
: Domain(Vector3d(0., 0., -2.*deltaT/dim(0))),
sdomCont_()
{
    std::ostringstream ss;
    ss << "OctetDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info(ss.str());
    
    double sqrt2 = std::sqrt(2.);
    
    double s = dim(0);
    double a = PI/5.*(dim(1)+dim(2))-(4.-PI)/5.*dim(3);
    double b = a/4.;
    double t = dim(3);
    
    BOOST_ASSERT_MSG(b > t, "Invalid dimensions");
    BOOST_ASSERT_MSG(a > (sqrt2+1.)*(b+t), "Invalid dimensions");
    BOOST_ASSERT_MSG(s/2. > a+(sqrt2+1.)*(b+t)+t, "Invalid dimensions");
    
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
    
    Vector3d o, u, v;
    Matrix3d mat3;
    Matrix3Xd mat4(3, 4);
    Vector3l sdomDiv;
    const Vector3l noDiv(-1l, -1l, -1l);
    const Vector3d gradT(0., 0., -2.*deltaT/dim(0));
    
    Matrix3Xd pts(3, 52);
    long p = 0l;
    
    // Horizontal Top
    {
        // 00: XY Prism
        o << (sqrt2+1.)*b+t, b+t, -a;
        mat4.transpose() <<
            0., 0., a,
            sqrt2*t, 0., 0.,
            s/4.-(2.+sqrt2)/2.*b-(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            s/4.-(2.+sqrt2)/2.*b-t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom< 0>() = Sdom< 0>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 01: XY Prism
        o << (sqrt2+1.)*b+t, b+t, -a-t;
        mat4.transpose() <<
            0., 0., t,
            sqrt2*t, 0., 0.,
            s/4.-(2.+sqrt2)/2.*b-(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            s/4.-(2.+sqrt2)/2.*b-t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom< 1>() = Sdom< 1>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 02: XY Prism
        o << b+t, b+t, -a-t;
        mat4.transpose() <<
            0., 0., t,
            sqrt2*b, 0., 0.,
            s/4.-(2.-sqrt2)/2.*b-t, s/4.-(2.+sqrt2)/2.*b-t, 0.,
            s/4.-b-t, s/4.-b-t, 0.;
        sdom< 2>() = Sdom< 2>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 03: XY Prism
        o << (sqrt2+1.)*b, b, -a;
        mat4.transpose() <<
            0., 0., a,
            (sqrt2+1.)*t, 0., 0.,
            (sqrt2+1.)*t, t, 0.,
            t, t, 0.;
        sdom< 3>() = Sdom< 3>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 04: XY Prism
        o << (sqrt2+1.)*b, b, -a-t;
        mat4.transpose() <<
            0., 0., t,
            (sqrt2+1.)*t, 0., 0.,
            (sqrt2+1.)*t, t, 0.,
            t, t, 0.;
        sdom< 4>() = Sdom< 4>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 05: XY Parallelepiped
        o << b, b, -a-t;
        mat3.transpose() <<
            0., 0., t,
            sqrt2*b, 0., 0.,
            t, t, 0.;
        sdom< 5>() = Sdom< 5>::type(o, mat3, noDiv, gradT);
        
        u << t, 0., 0.;
        pts.col(p++) = o + mat3.col(0)/2. + (mat3.col(2) + u)/3.;
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;
    }

    // Node Top
    {
        // 06: XZ Prism
        o << a, 0., 0.;
        mat4.transpose() <<
            0., b, 0.,
            sqrt2*t, 0., 0.,
            (sqrt2+1.)/2.*b+(sqrt2+1.)*t, 0., -(sqrt2+1.)/2.*b-t,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdom< 6>() = Sdom< 6>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 07: XZ Prism
        o << a, b, 0.;
        mat4.transpose() <<
            0., t, 0.,
            sqrt2*t, 0., 0.,
            (sqrt2+1.)/2.*b+(sqrt2+1.)*t, 0., -(sqrt2+1.)/2.*b-t,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdom< 7>() = Sdom< 7>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 08: XZ Prism
        o << (sqrt2+1.)*(b+t), b, 0.;
        mat4.transpose() <<
            0., t, 0.,
            a-(sqrt2+1.)*(b+t), 0., 0., // CONSTRAINT
            a-(sqrt2+1.)/2.*b-t/sqrt2, 0., -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t,
            0., 0., -a-t;
        sdom< 8>() = Sdom< 8>::type(o, mat4, noDiv(0), gradT);
        
        v << 0., 0., -a;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(1) + v)/2.;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;

        // 09: XZ Prism
        o << b+t, b, -a-t;
        mat4.transpose() <<
            0., t, 0.,
            sqrt2*(b+t), 0., 0.,
            (sqrt2-1.)/2.*b+t/sqrt2, 0., -(sqrt2+1.)/2.*b-t/sqrt2,
            0., 0., -b;
        sdom< 9>() = Sdom< 9>::type(o, mat4, noDiv(0), gradT);
        
        u << sqrt2*b, 0., 0.;
        v << 0., 0., -b+t;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2) + u)/2.;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;
        pts.col(p++) = o + (mat4.col(0) + u + v)/2.;

        // 10: XZ Prism
        o << b+t, b, -a-b-t;
        mat4.transpose() <<
            0., t, 0.,
            (sqrt2-1.)/2.*b+t/sqrt2, 0., -(sqrt2-1.)/2.*b-t/sqrt2,
            (sqrt2-1.)/2.*b, 0., -(sqrt2-1.)/2.*b-sqrt2*t,
            0., 0., -sqrt2*t;
        sdom<10>() = Sdom<10>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 11: XZ Prism
        o << b+t, 0., -a-b-t;
        mat4.transpose() <<
            0., b, 0.,
            (sqrt2-1.)/2.*b+t/sqrt2, 0., -(sqrt2-1.)/2.*b-t/sqrt2,
            (sqrt2-1.)/2.*b, 0., -(sqrt2-1.)/2.*b-sqrt2*t,
            0., 0., -sqrt2*t;
        sdom<11>() = Sdom<11>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 12: XY TriangularPrism
        o << b+t, b, -a-b;
        mat3.transpose() <<
            0., t, 0.,
            -t, 0., 0.,
            0., 0., b-t;  // CONSTRAINT
        sdom<12>() = Sdom<12>::type(o, mat3, noDiv, gradT);
        
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 13: YZ Pyramid
        o << b+t, b, -a-b-t;
        mat4.transpose() <<
            -t, 0., t,
            0., 0., t,
            0., t, t,
            0., t, 0.;
        sdom<13>() = Sdom<13>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + mat4.rowwise().sum()/5.;

        // 14: XY TriangularPrism
        o << b+t, b, -a-b-(sqrt2+1.)*t;
        mat3.transpose() <<
            0., t, 0.,
            -t, 0., t,
            0., 0., sqrt2*t;
        sdom<14>() = Sdom<14>::type(o, mat3, noDiv, gradT);
        
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 15: XY Prism
        o << b+t, 0., -a-b-(sqrt2+1.)*t;
        mat4.transpose() <<
            0., 0., sqrt2*t,
            0., b, 0.,
            -t, b, t,
            -b-t, 0., b+t;
        sdom<15>() = Sdom<15>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;
    }
    
    // Vertical Top
    {
        // 16: XZ Parallelepiped
        o << s/4.+a/2., 0., -s/4.+a/2.;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., b, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(2), div(0);
        sdom<16>() = Sdom<16>::type(o, mat3, sdomDiv, gradT);
        
        pts.col(p++) = o + mat3.rowwise().sum()/2.;

        // 17: XZ Parallelepiped
        o << s/4.+a/2., b, -s/4.+a/2.;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<17>() = Sdom<17>::type(o, mat3, sdomDiv, gradT);
        
        pts.col(p++) = o + mat3.rowwise().sum()/2.;

        // 18: XZ Parallelepiped
        o << s/4.-a/2., b, -s/4.-a/2.;
        mat3.transpose() <<
            a, 0., a,
            0., t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(1), div(3), div(0);
        sdom<18>() = Sdom<18>::type(o, mat3, sdomDiv, gradT);
        
        u << (sqrt2+1.)/2.*b+t/sqrt2, 0., (sqrt2+1.)/2.*b+t/sqrt2;
        pts.col(p++) = o + (mat3.col(1) + mat3.col(2) + u)/2.;
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;

        // 19: XZ Parallelepiped
        o << s/4.-a/2.-t/sqrt2, b, -s/4.-a/2.-t/sqrt2;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<19>() = Sdom<19>::type(o, mat3, sdomDiv, gradT);
        pts.col(p++) = o + mat3.rowwise().sum()/2.;

        // 20: XZ Parallelepiped
        o << s/4.-a/2.-t/sqrt2, 0., -s/4.-a/2.-t/sqrt2;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., b, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(2), div(0);
        sdom<20>() = Sdom<20>::type(o, mat3, sdomDiv, gradT);
        pts.col(p++) = o + mat3.rowwise().sum()/2.;
    }

    // Vertical Bottom
    {
        // 21: XZ Parallelepiped
        o << s/4.+a/2.+t/sqrt2, 0., -s/4.+a/2.+t/sqrt2;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., b, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(2), div(0);
        sdom<21>() = Sdom<21>::type(o, mat3, sdomDiv, gradT);
        
        pts.col(p++) = o + mat3.rowwise().sum()/2.;

        // 22: XZ Parallelepiped
        o << s/4.+a/2.+t/sqrt2, b, -s/4.+a/2.+t/sqrt2;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<22>() = Sdom<22>::type(o, mat3, sdomDiv, gradT);
        
        pts.col(p++) = o + mat3.rowwise().sum()/2.;

        // 23: XZ Parallelepiped
        o << s/4.+a/2., b, -s/4.+a/2.;
        mat3.transpose() <<
            -a, 0., -a,
            0., t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(1), div(3), div(0);
        sdom<23>() = Sdom<23>::type(o, mat3, sdomDiv, gradT);
        
        u << -(sqrt2+1.)/2.*b-t/sqrt2, 0., -(sqrt2+1.)/2.*b-t/sqrt2;
        pts.col(p++) = o + (mat3.col(1) + mat3.col(2) + u)/2.;
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;

        // 24: XZ Parallelepiped
        o << s/4.-a/2., b, -s/4.-a/2.;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<24>() = Sdom<24>::type(o, mat3, sdomDiv, gradT);
        
        pts.col(p++) = o + mat3.rowwise().sum()/2.;

        // 25: XZ Parallelepiped
        o << s/4.-a/2., 0., -s/4.-a/2.;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., b, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(2), div(0);
        sdom<25>() = Sdom<25>::type(o, mat3, sdomDiv, gradT);
        
        pts.col(p++) = o + mat3.rowwise().sum()/2.;
    }

    // Node Bottom
    {
        // 26: XY Prism
        o << s/2.-b-t, 0., -s/2.+a+b+(sqrt2+1.)*t;
        mat4.transpose() <<
            0., 0., -sqrt2*t,
            0., b, 0.,
            t, b, -t,
            b+t, 0., -b-t;
        sdom<26>() = Sdom<26>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 27: XY TriangularPrism
        o << s/2.-b-t, b, -s/2.+a+b+(sqrt2+1.)*t;
        mat3.transpose() <<
            0., t, 0.,
            t, 0., -t,
            0., 0., -sqrt2*t;
        sdom<27>() = Sdom<27>::type(o, mat3, noDiv, gradT);
        
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 28: YZ Pyramid
        o << s/2.-b-t, b, -s/2.+a+b+t;
        mat4.transpose() <<
            t, 0., -t,
            0., 0., -t,
            0., t, -t,
            0., t, 0.;
        sdom<28>() = Sdom<28>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + mat4.rowwise().sum()/5.;

        // 29: XY TriangularPrism
        o << s/2.-b-t, b, -s/2.+a+b;
        mat3.transpose() <<
            0., t, 0.,
            t, 0., 0.,
            0., 0., -b+t;  // CONSTRAINT
        sdom<29>() = Sdom<29>::type(o, mat3, noDiv, gradT);
        
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 30: XZ Prism
        o << s/2.-b-t, 0., -s/2.+a+b+t;
        mat4.transpose() <<
            0., b, 0.,
            -(sqrt2-1.)/2.*b-t/sqrt2, 0., (sqrt2-1.)/2.*b+t/sqrt2,
            -(sqrt2-1.)/2.*b, 0., (sqrt2-1.)/2.*b+sqrt2*t,
            0., 0., sqrt2*t;
        sdom<30>() = Sdom<30>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 31: XZ Prism
        o << s/2.-b-t, b, -s/2.+a+b+t;
        mat4.transpose() <<
            0., t, 0.,
            -(sqrt2-1.)/2.*b-t/sqrt2, 0., (sqrt2-1.)/2.*b+t/sqrt2,
            -(sqrt2-1.)/2.*b, 0., (sqrt2-1.)/2.*b+sqrt2*t,
            0., 0., sqrt2*t;
        sdom<31>() = Sdom<31>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 32: XZ Prism
        o << s/2.-b-t, b, -s/2.+a+t;
        mat4.transpose() <<
            0., t, 0.,
            -sqrt2*(b+t), 0., 0.,
            -(sqrt2-1.)/2.*b-t/sqrt2, 0., (sqrt2+1.)/2.*b+t/sqrt2,
            0., 0., b;
        sdom<32>() = Sdom<32>::type(o, mat4, noDiv(0), gradT);
        
        u << -sqrt2*b, 0., 0.;
        v << 0., 0., b-t;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2) + u)/2.;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;
        pts.col(p++) = o + (mat4.col(0) + u + v)/2.;

        // 33: XZ Prism
        o << s/2.-(sqrt2+1.)*(b+t), b, -s/2.;
        mat4.transpose() <<
            0., t, 0.,
            -a+(sqrt2+1.)*(b+t), 0., 0., // CONSTRAINT
            -a+(sqrt2+1.)/2.*b+t/sqrt2, 0., (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t,
            0., 0., a+t;
        sdom<33>() = Sdom<33>::type(o, mat4, noDiv(0), gradT);
        
        v << 0., 0., a;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(1) + v)/2.;
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;

        // 34: XZ Prism
        o << s/2.-a, b, -s/2.;
        mat4.transpose() <<
            0., t, 0.,
            -sqrt2*t, 0., 0.,
            -(sqrt2+1.)/2.*b-(sqrt2+1.)*t, 0., (sqrt2+1.)/2.*b+t,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdom<34>() = Sdom<34>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 35: XZ Prism
        o << s/2.-a, 0., -s/2.;
        mat4.transpose() <<
            0., b, 0.,
            -sqrt2*t, 0., 0.,
            -(sqrt2+1.)/2.*b-(sqrt2+1.)*t, 0., (sqrt2+1.)/2.*b+t,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdom<35>() = Sdom<35>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;
    }

    // Horizontal Bottom
    {
        // 36: XY Parallelepiped
        o << s/2.-b, b, -s/2.+a+t;
        mat3.transpose() <<
            0., 0., -t,
            -sqrt2*b, 0., 0.,
            -t, t, 0.;
        sdom<36>() = Sdom<36>::type(o, mat3, noDiv, gradT);
        
        u << -t, 0., 0.;
        pts.col(p++) = o + mat3.col(0)/2. + (mat3.col(2) + u)/3.;
        pts.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;

        // 37: XY Prism
        o << s/2.-(sqrt2+1.)*b, b, -s/2.+a+t;
        mat4.transpose() <<
            0., 0., -t,
            -(sqrt2+1.)*t, 0., 0.,
            -(sqrt2+1.)*t, t, 0.,
            -t, t, 0.;
        sdom<37>() = Sdom<37>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 38: XY Prism
        o << s/2.-(sqrt2+1.)*b, b, -s/2.+a;
        mat4.transpose() <<
            0., 0., -a,
            -(sqrt2+1.)*t, 0., 0.,
            -(sqrt2+1.)*t, t, 0.,
            -t, t, 0.;
        sdom<38>() = Sdom<38>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 39: XY Prism
        o << s/2.-b-t, b+t, -s/2.+a+t;
        mat4.transpose() <<
            0., 0., -t,
            -sqrt2*b, 0., 0.,
            -s/4.+(2.-sqrt2)/2.*b+t, s/4.-(2.+sqrt2)/2.*b-t, 0.,
            -s/4.+b+t, s/4.-b-t, 0.;
        sdom<39>() = Sdom<39>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 40: XY Prism
        o << s/2.-(sqrt2+1.)*b-t, b+t, -s/2.+a+t;
        mat4.transpose() <<
            0., 0., -t,
            -sqrt2*t, 0., 0.,
            -s/4.+(2.+sqrt2)/2.*b+(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            -s/4.+(2.+sqrt2)/2.*b+t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom<40>() = Sdom<40>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 41: XY Prism
        o << s/2.-(sqrt2+1.)*b-t, b+t, -s/2.+a;
        mat4.transpose() <<
            0., 0., -a,
            -sqrt2*t, 0., 0.,
            -s/4.+(2.+sqrt2)/2.*b+(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            -s/4.+(2.+sqrt2)/2.*b+t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom<41>() = Sdom<41>::type(o, mat4, noDiv(0), gradT);
        
        pts.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;
    }
    
    // InterBoundary Pairs
    {
        makePair(sdom< 0>().bdry<0>(), sdom< 1>().bdry<1>());
        makePair(sdom< 1>().bdry<5>(), sdom< 2>().bdry<3>());
        makePair(sdom< 0>().bdry<2>(), sdom< 3>().bdry<4>());
        makePair(sdom< 1>().bdry<2>(), sdom< 4>().bdry<4>());
        makePair(sdom< 2>().bdry<2>(), sdom< 5>().bdry<5>());
        makePair(sdom< 3>().bdry<0>(), sdom< 4>().bdry<1>());
        makePair(sdom< 4>().bdry<5>(), sdom< 5>().bdry<4>());
        
        makePair(sdom< 3>().bdry<3>(), sdom< 8>().bdry<5>());
        makePair(sdom< 4>().bdry<3>(), sdom< 8>().bdry<5>());
        makePair(sdom< 4>().bdry<0>(), sdom< 9>().bdry<2>());
        makePair(sdom< 5>().bdry<0>(), sdom< 9>().bdry<2>());
        makePair(sdom< 5>().bdry<0>(), sdom<12>().bdry<4>());
        
        makePair(sdom< 6>().bdry<1>(), sdom< 7>().bdry<0>());
        makePair(sdom< 7>().bdry<5>(), sdom< 8>().bdry<3>());
        makePair(sdom< 9>().bdry<4>(), sdom<10>().bdry<2>());
        makePair(sdom<10>().bdry<0>(), sdom<11>().bdry<1>());
        makePair(sdom< 9>().bdry<5>(), sdom<12>().bdry<1>());
        makePair(sdom< 9>().bdry<5>(), sdom<13>().bdry<0>());
        makePair(sdom<10>().bdry<5>(), sdom<14>().bdry<1>());
        makePair(sdom<11>().bdry<5>(), sdom<15>().bdry<2>());
        makePair(sdom<12>().bdry<2>(), sdom<13>().bdry<2>());
        makePair(sdom<13>().bdry<4>(), sdom<14>().bdry<4>());
        makePair(sdom<14>().bdry<0>(), sdom<15>().bdry<3>());
        
        makePair(sdom< 6>().bdry<4>(), sdom<16>().bdry<5>());
        makePair(sdom< 7>().bdry<4>(), sdom<17>().bdry<5>());
        makePair(sdom< 8>().bdry<4>(), sdom<18>().bdry<5>());
        makePair(sdom< 9>().bdry<3>(), sdom<18>().bdry<5>());
        makePair(sdom<10>().bdry<3>(), sdom<19>().bdry<5>());
        makePair(sdom<11>().bdry<3>(), sdom<20>().bdry<5>());
        
        makePair(sdom<16>().bdry<4>(), sdom<17>().bdry<1>());
        makePair(sdom<17>().bdry<0>(), sdom<18>().bdry<3>());
        makePair(sdom<18>().bdry<0>(), sdom<19>().bdry<3>());
        makePair(sdom<19>().bdry<1>(), sdom<20>().bdry<4>());
        
        makePair(sdom<16>().bdry<2>(), sdom<21>().bdry<2>());
        makePair(sdom<17>().bdry<2>(), sdom<22>().bdry<2>());
        makePair(sdom<18>().bdry<2>(), sdom<23>().bdry<2>());
        makePair(sdom<19>().bdry<2>(), sdom<24>().bdry<2>());
        makePair(sdom<20>().bdry<2>(), sdom<25>().bdry<2>());
        
        makePair(sdom<21>().bdry<4>(), sdom<22>().bdry<1>());
        makePair(sdom<22>().bdry<3>(), sdom<23>().bdry<0>());
        makePair(sdom<23>().bdry<3>(), sdom<24>().bdry<0>());
        makePair(sdom<24>().bdry<1>(), sdom<25>().bdry<4>());
        
        makePair(sdom<21>().bdry<5>(), sdom<30>().bdry<3>());
        makePair(sdom<22>().bdry<5>(), sdom<31>().bdry<3>());
        makePair(sdom<23>().bdry<5>(), sdom<32>().bdry<3>());
        makePair(sdom<23>().bdry<5>(), sdom<33>().bdry<4>());
        makePair(sdom<24>().bdry<5>(), sdom<34>().bdry<4>());
        makePair(sdom<25>().bdry<5>(), sdom<35>().bdry<4>());
        
        makePair(sdom<26>().bdry<3>(), sdom<27>().bdry<0>());
        makePair(sdom<27>().bdry<4>(), sdom<28>().bdry<4>());
        makePair(sdom<28>().bdry<2>(), sdom<29>().bdry<2>());
        makePair(sdom<26>().bdry<2>(), sdom<30>().bdry<5>());
        makePair(sdom<27>().bdry<1>(), sdom<31>().bdry<5>());
        makePair(sdom<28>().bdry<0>(), sdom<32>().bdry<5>());
        makePair(sdom<29>().bdry<1>(), sdom<32>().bdry<5>());
        makePair(sdom<30>().bdry<1>(), sdom<31>().bdry<0>());
        makePair(sdom<31>().bdry<2>(), sdom<32>().bdry<4>());
        makePair(sdom<33>().bdry<3>(), sdom<34>().bdry<5>());
        makePair(sdom<34>().bdry<0>(), sdom<35>().bdry<1>());
        
        makePair(sdom<29>().bdry<4>(), sdom<36>().bdry<0>());
        makePair(sdom<32>().bdry<2>(), sdom<36>().bdry<0>());
        makePair(sdom<32>().bdry<2>(), sdom<37>().bdry<0>());
        makePair(sdom<33>().bdry<5>(), sdom<37>().bdry<3>());
        makePair(sdom<33>().bdry<5>(), sdom<38>().bdry<3>());
        
        makePair(sdom<36>().bdry<4>(), sdom<37>().bdry<5>());
        makePair(sdom<37>().bdry<1>(), sdom<38>().bdry<0>());
        makePair(sdom<36>().bdry<5>(), sdom<39>().bdry<2>());
        makePair(sdom<37>().bdry<4>(), sdom<40>().bdry<2>());
        makePair(sdom<38>().bdry<4>(), sdom<41>().bdry<2>());
        makePair(sdom<39>().bdry<3>(), sdom<40>().bdry<5>());
        makePair(sdom<40>().bdry<1>(), sdom<41>().bdry<0>());
    }
    
    // PeriBoundary Pairs
    {
        Vector3d transl = Vector3d(s/2., 0., -s/2.);
        Matrix3d rot = Diagonal(-1., 1., 1.);
        
        makePair(sdom< 0>().bdry<1>(), sdom<41>().bdry<1>(), transl, rot);
        makePair(sdom< 3>().bdry<1>(), sdom<38>().bdry<1>(), transl, rot);
        makePair(sdom< 6>().bdry<2>(), sdom<35>().bdry<2>(), transl, rot);
        makePair(sdom< 7>().bdry<2>(), sdom<34>().bdry<2>(), transl, rot);
        makePair(sdom< 8>().bdry<2>(), sdom<33>().bdry<2>(), transl, rot);
    }
    
    BOOST_ASSERT_MSG(p == pts.cols(), "Incorrect number of checkpoints");
    checkpoints(pts);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

template<typename T, int N>
OctetDomain::AverageF<T, N>::AverageF(const OctetDomain& dom)
:dom_(&dom)
{
    BOOST_ASSERT_MSG(dom.isInit(), "Domain is not initialized");
    
    for (int i = 16; i < 26; ++i)
    {
        sdomAvg_.push_back( dom.sdomPtrs().at(i) );
    }
}

template<typename T, int N>
typename OctetDomain::AverageF<T, N>::VectorNT
OctetDomain::AverageF<T, N>::operator()(const Subdomain* sdom,
                                        const Vector3l& index) const
{
    Subdomain::Pointers::const_iterator beg = sdomAvg_.begin();
    Subdomain::Pointers::const_iterator end = sdomAvg_.end();
    
    double scalar = 0.;
    if (index(2) == 0l && std::find(beg, end, sdom) != end)
    {
        scalar = sdom->cellVol(index);
    }
    return VectorNT::Constant(std::max(1, N), scalar);
}

template<typename T, int N>
typename OctetDomain::AverageF<T, N>::VectorNT
OctetDomain::AverageF<T, N>::operator()(const Field<T, N>& fld) const
{
    Field<T, N> weight(dom_, *this);
    return fld.average(weight);
}

template class OctetDomain::AverageF<double, Eigen::Dynamic>;
template class OctetDomain::AverageF<double, 1>;
template class OctetDomain::AverageF<double, 2>;
template class OctetDomain::AverageF<double, 3>;
template class OctetDomain::AverageF<double, 4>;

//OctetDomain::OctetDomain(const Eigen::Matrix<double, 5, 1>& dim,
//                         const Eigen::Matrix<long, 5, 1>& div,
//                         double deltaT)
//: Domain(Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//sdomCont_(
//          Sdom< 0>::type(Vector3d(-dim(4), 0., -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 1>::type(Vector3d(-dim(4), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 2>::type(Vector3d(0., dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(2), dim(4), dim(0)),
//                         Vector3l(div(2), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 3>::type(Vector3d(dim(2), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 4>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(2)-dim(4), dim(4), dim(0)),
//                         Vector3l(div(2)-div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 5>::type(Vector3d(2.*dim(2), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 6>::type(Vector3d(2.*dim(2), 0., -dim(0)-dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom< 7>::type(Vector3d(0., dim(3), -dim(2)-dim(4)),
//                         Diagonal(dim(2), dim(4), dim(4)),
//                         Vector3l(div(2), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 8>::type(Vector3d(dim(2), dim(3), -dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 9>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(2)-dim(4)),
//                         Diagonal(dim(2)-dim(4), dim(4), dim(4)),
//                         Vector3l(div(2)-div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<10>::type(Vector3d(2.*dim(2), dim(3), -dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<11>::type(Vector3d(2.*dim(2), 0., -dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(3), dim(4)),
//                         Vector3l(div(4), div(3), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<12>::type(Vector3d(0., dim(3)+dim(4), -dim(2)-dim(4)),
//                         Diagonal(dim(2), dim(1), dim(4)),
//                         Vector3l(div(2), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<13>::type(Vector3d(dim(2), dim(3)+dim(4), -dim(2)-dim(4)),
//                         Diagonal(dim(4), dim(1), dim(4)),
//                         Vector3l(div(4), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom<14>::type(Vector3d(dim(2), dim(3), -dim(2)),
//                         Diagonal(dim(4), dim(4), 2.*dim(2)),
//                         Vector3l(div(4), div(4), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<15>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(2)),
//                         Diagonal(dim(2)-dim(4), dim(4), 2.*dim(2)),
//                         Vector3l(div(2)-div(4), div(4), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<16>::type(Vector3d(2.*dim(2), dim(3), -dim(2)),
//                         Diagonal(dim(4), dim(4), 2.*dim(2)),
//                         Vector3l(div(4), div(4), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<17>::type(Vector3d(2.*dim(2), 0., -dim(2)),
//                         Diagonal(dim(4), dim(3), 2.*dim(2)),
//                         Vector3l(div(4), div(3), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<18>::type(Vector3d(dim(2), dim(3)+dim(4), -dim(2)),
//                         Diagonal(dim(4), dim(1), 2.*dim(2)),
//                         Vector3l(div(4), div(1), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom<19>::type(Vector3d(0., dim(3), dim(2)),
//                         Diagonal(dim(2), dim(4), dim(4)),
//                         Vector3l(div(2), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<20>::type(Vector3d(dim(2), dim(3), dim(2)),
//                         Diagonal(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<21>::type(Vector3d(dim(2)+dim(4), dim(3), dim(2)),
//                         Diagonal(dim(2)-dim(4), dim(4), dim(4)),
//                         Vector3l(div(2)-div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<22>::type(Vector3d(2.*dim(2), dim(3), dim(2)),
//                         Diagonal(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<23>::type(Vector3d(2.*dim(2), 0., dim(2)),
//                         Diagonal(dim(4), dim(3), dim(4)),
//                         Vector3l(div(4), div(3), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<24>::type(Vector3d(0., dim(3)+dim(4), dim(2)),
//                         Diagonal(dim(2), dim(1), dim(4)),
//                         Vector3l(div(2), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<25>::type(Vector3d(dim(2), dim(3)+dim(4), dim(2)),
//                         Diagonal(dim(4), dim(1), dim(4)),
//                         Vector3l(div(4), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom<26>::type(Vector3d(-dim(4), 0., dim(2)+dim(4)),
//                         Diagonal(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<27>::type(Vector3d(-dim(4), dim(3), dim(2)+dim(4)),
//                         Diagonal(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<28>::type(Vector3d(0., dim(3), dim(2)+dim(4)),
//                         Diagonal(dim(2), dim(4), dim(0)),
//                         Vector3l(div(2), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<29>::type(Vector3d(dim(2), dim(3), dim(2)+dim(4)),
//                         Diagonal(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<30>::type(Vector3d(dim(2)+dim(4), dim(3), dim(2)+dim(4)),
//                         Diagonal(dim(2)-dim(4), dim(4), dim(0)),
//                         Vector3l(div(2)-div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<31>::type(Vector3d(2.*dim(2), dim(3), dim(2)+dim(4)),
//                         Diagonal(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<32>::type(Vector3d(2.*dim(2), 0., dim(2)+dim(4)),
//                         Diagonal(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))))
//{
//    std::ostringstream ss;
//    ss << "OctetDomain " << static_cast<Domain*>(this) << std::endl;
//    ss << "  dim: " << dim.transpose() << std::endl;
//    ss << "  div: " << div.transpose() << std::endl;
//    ss << "  dT:  " << deltaT;
//    info(ss.str());
//    
//    Eigen::Matrix<double, Eigen::Dynamic, 3> pts(33, 3);
//    pts <<    -0.5*dim(4),               0.5*dim(3), -0.5*dim(0)-dim(2)-dim(4),
//    -0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
//    0.5*dim(2),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
//    dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
//    1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
//    2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
//    2.0*dim(2)+0.5*dim(4),               0.5*dim(3), -0.5*dim(0)-dim(2)-dim(4),
//    
//    0.5*dim(2),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
//    dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
//    1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
//    2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
//    2.0*dim(2)+0.5*dim(4),               0.5*dim(3),        -dim(2)-0.5*dim(4),
//    0.5*dim(2), 0.5*dim(1)+dim(3)+dim(4),        -dim(2)-0.5*dim(4),
//    dim(2)+0.5*dim(4), 0.5*dim(1)+dim(3)+dim(4),        -dim(2)-0.5*dim(4),
//    
//    dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),                        0.,
//    1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),                        0.,
//    2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),                        0.,
//    2.0*dim(2)+0.5*dim(4),               0.5*dim(3),                        0.,
//    dim(2)+0.5*dim(4), 0.5*dim(1)+dim(3)+dim(4),                        0.,
//    
//    0.5*dim(2),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
//    dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
//    1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
//    2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
//    2.0*dim(2)+0.5*dim(4),               0.5*dim(3),         dim(2)+0.5*dim(4),
//    0.5*dim(2), 0.5*dim(1)+dim(3)+dim(4),         dim(2)+0.5*dim(4),
//    dim(2)+0.5*dim(4), 0.5*dim(1)+dim(3)+dim(4),         dim(2)+0.5*dim(4),
//    
//    -0.5*dim(4),               0.5*dim(3),  0.5*dim(0)+dim(2)+dim(4),
//    -0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
//    0.5*dim(2),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
//    dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
//    1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
//    2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
//    2.0*dim(2)+0.5*dim(4),               0.5*dim(3),  0.5*dim(0)+dim(2)+dim(4);
//    checkpoints(pts.transpose());
//    
//    makePair(sdom< 0>().bdry<4>(), sdom< 1>().bdry<1>());
//    makePair(sdom< 1>().bdry<3>(), sdom< 2>().bdry<0>());
//    makePair(sdom< 2>().bdry<3>(), sdom< 3>().bdry<0>());
//    makePair(sdom< 3>().bdry<3>(), sdom< 4>().bdry<0>());
//    makePair(sdom< 4>().bdry<3>(), sdom< 5>().bdry<0>());
//    makePair(sdom< 5>().bdry<1>(), sdom< 6>().bdry<4>());
//    
//    makePair(sdom< 2>().bdry<5>(), sdom< 7>().bdry<2>());
//    makePair(sdom< 3>().bdry<5>(), sdom< 8>().bdry<2>());
//    makePair(sdom< 4>().bdry<5>(), sdom< 9>().bdry<2>());
//    makePair(sdom< 5>().bdry<5>(), sdom<10>().bdry<2>());
//    makePair(sdom< 6>().bdry<5>(), sdom<11>().bdry<2>());
//    
//    makePair(sdom< 7>().bdry<3>(), sdom< 8>().bdry<0>());
//    makePair(sdom< 8>().bdry<3>(), sdom< 9>().bdry<0>());
//    makePair(sdom< 9>().bdry<3>(), sdom<10>().bdry<0>());
//    makePair(sdom<10>().bdry<1>(), sdom<11>().bdry<4>());
//    makePair(sdom< 7>().bdry<4>(), sdom<12>().bdry<1>());
//    makePair(sdom< 8>().bdry<4>(), sdom<13>().bdry<1>());
//    makePair(sdom<12>().bdry<3>(), sdom<13>().bdry<0>());
//    
//    makePair(sdom< 8>().bdry<5>(), sdom<14>().bdry<2>());
//    makePair(sdom< 9>().bdry<5>(), sdom<15>().bdry<2>());
//    makePair(sdom<10>().bdry<5>(), sdom<16>().bdry<2>());
//    makePair(sdom<11>().bdry<5>(), sdom<17>().bdry<2>());
//    makePair(sdom<13>().bdry<5>(), sdom<18>().bdry<2>());
//    
//    makePair(sdom<14>().bdry<3>(), sdom<15>().bdry<0>());
//    makePair(sdom<15>().bdry<3>(), sdom<16>().bdry<0>());
//    makePair(sdom<16>().bdry<1>(), sdom<17>().bdry<4>());
//    makePair(sdom<14>().bdry<4>(), sdom<18>().bdry<1>());
//    
//    makePair(sdom<14>().bdry<5>(), sdom<20>().bdry<2>());
//    makePair(sdom<15>().bdry<5>(), sdom<21>().bdry<2>());
//    makePair(sdom<16>().bdry<5>(), sdom<22>().bdry<2>());
//    makePair(sdom<17>().bdry<5>(), sdom<23>().bdry<2>());
//    makePair(sdom<18>().bdry<5>(), sdom<25>().bdry<2>());
//    
//    makePair(sdom<19>().bdry<3>(), sdom<20>().bdry<0>());
//    makePair(sdom<20>().bdry<3>(), sdom<21>().bdry<0>());
//    makePair(sdom<21>().bdry<3>(), sdom<22>().bdry<0>());
//    makePair(sdom<22>().bdry<1>(), sdom<23>().bdry<4>());
//    makePair(sdom<19>().bdry<4>(), sdom<24>().bdry<1>());
//    makePair(sdom<20>().bdry<4>(), sdom<25>().bdry<1>());
//    makePair(sdom<24>().bdry<3>(), sdom<25>().bdry<0>());
//    
//    makePair(sdom<19>().bdry<5>(), sdom<28>().bdry<2>());
//    makePair(sdom<20>().bdry<5>(), sdom<29>().bdry<2>());
//    makePair(sdom<21>().bdry<5>(), sdom<30>().bdry<2>());
//    makePair(sdom<22>().bdry<5>(), sdom<31>().bdry<2>());
//    makePair(sdom<23>().bdry<5>(), sdom<32>().bdry<2>());
//    
//    makePair(sdom<26>().bdry<4>(), sdom<27>().bdry<1>());
//    makePair(sdom<27>().bdry<3>(), sdom<28>().bdry<0>());
//    makePair(sdom<28>().bdry<3>(), sdom<29>().bdry<0>());
//    makePair(sdom<29>().bdry<3>(), sdom<30>().bdry<0>());
//    makePair(sdom<30>().bdry<3>(), sdom<31>().bdry<0>());
//    makePair(sdom<31>().bdry<1>(), sdom<32>().bdry<4>());
//    
//    Vector3d transl(0., 0., 2.*(dim(0)+dim(2)+dim(4)));
//    makePair(sdom< 0>().bdry<2>(), sdom<26>().bdry<5>(), transl);
//    makePair(sdom< 1>().bdry<2>(), sdom<27>().bdry<5>(), transl);
//    makePair(sdom< 2>().bdry<2>(), sdom<28>().bdry<5>(), transl);
//    makePair(sdom< 3>().bdry<2>(), sdom<29>().bdry<5>(), transl);
//    makePair(sdom< 4>().bdry<2>(), sdom<30>().bdry<5>(), transl);
//    makePair(sdom< 5>().bdry<2>(), sdom<31>().bdry<5>(), transl);
//    makePair(sdom< 6>().bdry<2>(), sdom<32>().bdry<5>(), transl);
//    
//    fusion::for_each(sdomCont_, AddSdomF(this));
//}

