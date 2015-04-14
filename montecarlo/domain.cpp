//
//  domain.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/5/15.
//
//

#include "domain.h"
#include "boundary.h"
#include <Eigen/Core>
#include <boost/fusion/algorithm/iteration.hpp>
#include <sstream>
#include <string>

namespace fusion = boost::fusion;

TeeDomain::TeeDomain(const Eigen::Matrix<double, 5, 1>& dim,
                     const Eigen::Matrix<long, 5, 1>& div,
                     double deltaT)
: sdomCont_(Sdom<0>::type(Vector3d::Zero(),
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
    info( ss.str() );
    
    centers_ <<        0.5*dim(0),        0.5*dim(2), 0.5*dim(4),
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

TubeDomain::TubeDomain(const Eigen::Matrix<double, 4, 1>& dim,
                       const Eigen::Matrix<long, 4, 1>& div,
                       double deltaT)
: sdomCont_(Sdom<0>::type(Vector3d(0., dim(1), 0.),
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
    info( ss.str() );
    
    centers_ << 0.5*dim(0), dim(1)+0.5*dim(3),        0.5*dim(2),
                0.5*dim(0), dim(1)+0.5*dim(3), dim(2)+0.5*dim(3),
                0.5*dim(0),        0.5*dim(1), dim(2)+0.5*dim(3);
    
    makePair(sdom<1>().bdry<1>(), sdom<2>().bdry<4>());
    makePair(sdom<1>().bdry<2>(), sdom<0>().bdry<5>());
    
    Vector3d transl(dim(0), 0., 0.);
    makePair(sdom<0>().bdry<0>(), sdom<0>().bdry<3>(), transl);
    makePair(sdom<1>().bdry<0>(), sdom<1>().bdry<3>(), transl);
    makePair(sdom<2>().bdry<0>(), sdom<2>().bdry<3>(), transl);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

OctetDomain::OctetDomain(const Eigen::Matrix<double, 5, 1>& dim,
                         const Eigen::Matrix<long, 5, 1>& div,
                         double deltaT)
: sdomCont_(
Sdom< 0>::type(Vector3d(-dim(4), 0., -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(4), dim(3), dim(0)),
             Vector3l(div(4), div(3), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 1>::type(Vector3d(-dim(4), dim(3), -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(4), dim(4), dim(0)),
             Vector3l(div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 2>::type(Vector3d(0., dim(3), -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(2), dim(4), dim(0)),
             Vector3l(div(2), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 3>::type(Vector3d(dim(2), dim(3), -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(4), dim(4), dim(0)),
             Vector3l(div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 4>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(2)-dim(4), dim(4), dim(0)),
             Vector3l(div(2)-div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 5>::type(Vector3d(2.*dim(2), dim(3), -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(4), dim(4), dim(0)),
             Vector3l(div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 6>::type(Vector3d(2.*dim(2), 0., -dim(0)-dim(2)-dim(4)),
             Diagonal(dim(4), dim(3), dim(0)),
             Vector3l(div(4), div(3), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),

Sdom< 7>::type(Vector3d(0., dim(3), -dim(2)-dim(4)),
             Diagonal(dim(2), dim(4), dim(4)),
             Vector3l(div(2), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 8>::type(Vector3d(dim(2), dim(3), -dim(2)-dim(4)),
             Diagonal(dim(4), dim(4), dim(4)),
             Vector3l(div(4), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom< 9>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(2)-dim(4)),
             Diagonal(dim(2)-dim(4), dim(4), dim(4)),
             Vector3l(div(2)-div(4), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<10>::type(Vector3d(2.*dim(2), dim(3), -dim(2)-dim(4)),
             Diagonal(dim(4), dim(4), dim(4)),
             Vector3l(div(4), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<11>::type(Vector3d(2.*dim(2), 0., -dim(2)-dim(4)),
             Diagonal(dim(4), dim(3), dim(4)),
             Vector3l(div(4), div(3), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<12>::type(Vector3d(0., dim(3)+dim(4), -dim(2)-dim(4)),
             Diagonal(dim(2), dim(1), dim(4)),
             Vector3l(div(2), div(1), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<13>::type(Vector3d(dim(2), dim(3)+dim(4), -dim(2)-dim(4)),
             Diagonal(dim(4), dim(1), dim(4)),
             Vector3l(div(4), div(1), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),

Sdom<14>::type(Vector3d(dim(2), dim(3), -dim(2)),
             Diagonal(dim(4), dim(4), 2.*dim(2)),
             Vector3l(div(4), div(4), 2*div(2)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<15>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(2)),
             Diagonal(dim(2)-dim(4), dim(4), 2.*dim(2)),
             Vector3l(div(2)-div(4), div(4), 2*div(2)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<16>::type(Vector3d(2.*dim(2), dim(3), -dim(2)),
             Diagonal(dim(4), dim(4), 2.*dim(2)),
             Vector3l(div(4), div(4), 2*div(2)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<17>::type(Vector3d(2.*dim(2), 0., -dim(2)),
             Diagonal(dim(4), dim(3), 2.*dim(2)),
             Vector3l(div(4), div(3), 2*div(2)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<18>::type(Vector3d(dim(2), dim(3)+dim(4), -dim(2)),
             Diagonal(dim(4), dim(1), 2.*dim(2)),
             Vector3l(div(4), div(1), 2*div(2)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),

Sdom<19>::type(Vector3d(0., dim(3), dim(2)),
             Diagonal(dim(2), dim(4), dim(4)),
             Vector3l(div(2), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<20>::type(Vector3d(dim(2), dim(3), dim(2)),
             Diagonal(dim(4), dim(4), dim(4)),
             Vector3l(div(4), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<21>::type(Vector3d(dim(2)+dim(4), dim(3), dim(2)),
             Diagonal(dim(2)-dim(4), dim(4), dim(4)),
             Vector3l(div(2)-div(4), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<22>::type(Vector3d(2.*dim(2), dim(3), dim(2)),
             Diagonal(dim(4), dim(4), dim(4)),
             Vector3l(div(4), div(4), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<23>::type(Vector3d(2.*dim(2), 0., dim(2)),
             Diagonal(dim(4), dim(3), dim(4)),
             Vector3l(div(4), div(3), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<24>::type(Vector3d(0., dim(3)+dim(4), dim(2)),
             Diagonal(dim(2), dim(1), dim(4)),
             Vector3l(div(2), div(1), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<25>::type(Vector3d(dim(2), dim(3)+dim(4), dim(2)),
             Diagonal(dim(4), dim(1), dim(4)),
             Vector3l(div(4), div(1), div(4)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),

Sdom<26>::type(Vector3d(-dim(4), 0., dim(2)+dim(4)),
             Diagonal(dim(4), dim(3), dim(0)),
             Vector3l(div(4), div(3), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<27>::type(Vector3d(-dim(4), dim(3), dim(2)+dim(4)),
             Diagonal(dim(4), dim(4), dim(0)),
             Vector3l(div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<28>::type(Vector3d(0., dim(3), dim(2)+dim(4)),
             Diagonal(dim(2), dim(4), dim(0)),
             Vector3l(div(2), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<29>::type(Vector3d(dim(2), dim(3), dim(2)+dim(4)),
             Diagonal(dim(4), dim(4), dim(0)),
             Vector3l(div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<30>::type(Vector3d(dim(2)+dim(4), dim(3), dim(2)+dim(4)),
             Diagonal(dim(2)-dim(4), dim(4), dim(0)),
             Vector3l(div(2)-div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<31>::type(Vector3d(2.*dim(2), dim(3), dim(2)+dim(4)),
             Diagonal(dim(4), dim(4), dim(0)),
             Vector3l(div(4), div(4), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
Sdom<32>::type(Vector3d(2.*dim(2), 0., dim(2)+dim(4)),
             Diagonal(dim(4), dim(3), dim(0)),
             Vector3l(div(4), div(3), div(0)),
             Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4)))))),
centers_(33, 3)
{
    std::ostringstream ss;
    ss << "OctetDomain " << static_cast<Domain*>(this) << std::endl;
    ss << "  dim: " << dim.transpose() << std::endl;
    ss << "  div: " << div.transpose() << std::endl;
    ss << "  dT:  " << deltaT;
    info( ss.str() );
    
    centers_<< -0.5*dim(4),               0.5*dim(3), -0.5*dim(0)-dim(2)-dim(4),
               -0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
                0.5*dim(2),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
         dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
     1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
     2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4), -0.5*dim(0)-dim(2)-dim(4),
     2.0*dim(2)+0.5*dim(4),               0.5*dim(3), -0.5*dim(0)-dim(2)-dim(4),
    
                0.5*dim(2),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
         dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
     1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
     2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),        -dim(2)-0.5*dim(4),
     2.0*dim(2)+0.5*dim(4),               0.5*dim(3),        -dim(2)-0.5*dim(4),
                0.5*dim(2), 0.5*dim(1)+dim(3)+dim(4),        -dim(2)-0.5*dim(4),
         dim(2)+0.5*dim(4), 0.5*dim(1)+dim(3)+dim(4),        -dim(2)-0.5*dim(4),
    
         dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),                        0.,
     1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),                        0.,
     2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),                        0.,
     2.0*dim(2)+0.5*dim(4),               0.5*dim(3),                        0.,
         dim(2)+0.5*dim(4), 0.5*dim(1)+dim(3)+dim(4),                        0.,
    
                0.5*dim(2),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
         dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
     1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
     2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),         dim(2)+0.5*dim(4),
     2.0*dim(2)+0.5*dim(4),               0.5*dim(3),         dim(2)+0.5*dim(4),
                0.5*dim(2), 0.5*dim(1)+dim(3)+dim(4),         dim(2)+0.5*dim(4),
         dim(2)+0.5*dim(4), 0.5*dim(1)+dim(3)+dim(4),         dim(2)+0.5*dim(4),
    
               -0.5*dim(4),               0.5*dim(3),  0.5*dim(0)+dim(2)+dim(4),
               -0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
                0.5*dim(2),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
         dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
     1.5*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
     2.0*dim(2)+0.5*dim(4),        dim(3)+0.5*dim(4),  0.5*dim(0)+dim(2)+dim(4),
     2.0*dim(2)+0.5*dim(4),               0.5*dim(3),  0.5*dim(0)+dim(2)+dim(4);
    
    makePair(sdom< 0>().bdry<4>(), sdom< 1>().bdry<1>());
    makePair(sdom< 1>().bdry<3>(), sdom< 2>().bdry<0>());
    makePair(sdom< 2>().bdry<3>(), sdom< 3>().bdry<0>());
    makePair(sdom< 3>().bdry<3>(), sdom< 4>().bdry<0>());
    makePair(sdom< 4>().bdry<3>(), sdom< 5>().bdry<0>());
    makePair(sdom< 5>().bdry<1>(), sdom< 6>().bdry<4>());
    
    makePair(sdom< 2>().bdry<5>(), sdom< 7>().bdry<2>());
    makePair(sdom< 3>().bdry<5>(), sdom< 8>().bdry<2>());
    makePair(sdom< 4>().bdry<5>(), sdom< 9>().bdry<2>());
    makePair(sdom< 5>().bdry<5>(), sdom<10>().bdry<2>());
    makePair(sdom< 6>().bdry<5>(), sdom<11>().bdry<2>());
    
    makePair(sdom< 7>().bdry<3>(), sdom< 8>().bdry<0>());
    makePair(sdom< 8>().bdry<3>(), sdom< 9>().bdry<0>());
    makePair(sdom< 9>().bdry<3>(), sdom<10>().bdry<0>());
    makePair(sdom<10>().bdry<1>(), sdom<11>().bdry<4>());
    makePair(sdom< 7>().bdry<4>(), sdom<12>().bdry<1>());
    makePair(sdom< 8>().bdry<4>(), sdom<13>().bdry<1>());
    makePair(sdom<12>().bdry<3>(), sdom<13>().bdry<0>());
    
    makePair(sdom< 8>().bdry<5>(), sdom<14>().bdry<2>());
    makePair(sdom< 9>().bdry<5>(), sdom<15>().bdry<2>());
    makePair(sdom<10>().bdry<5>(), sdom<16>().bdry<2>());
    makePair(sdom<11>().bdry<5>(), sdom<17>().bdry<2>());
    makePair(sdom<13>().bdry<5>(), sdom<18>().bdry<2>());
    
    makePair(sdom<14>().bdry<3>(), sdom<15>().bdry<0>());
    makePair(sdom<15>().bdry<3>(), sdom<16>().bdry<0>());
    makePair(sdom<16>().bdry<1>(), sdom<17>().bdry<4>());
    makePair(sdom<14>().bdry<4>(), sdom<18>().bdry<1>());
    
    makePair(sdom<14>().bdry<5>(), sdom<20>().bdry<2>());
    makePair(sdom<15>().bdry<5>(), sdom<21>().bdry<2>());
    makePair(sdom<16>().bdry<5>(), sdom<22>().bdry<2>());
    makePair(sdom<17>().bdry<5>(), sdom<23>().bdry<2>());
    makePair(sdom<18>().bdry<5>(), sdom<25>().bdry<2>());
    
    makePair(sdom<19>().bdry<3>(), sdom<20>().bdry<0>());
    makePair(sdom<20>().bdry<3>(), sdom<21>().bdry<0>());
    makePair(sdom<21>().bdry<3>(), sdom<22>().bdry<0>());
    makePair(sdom<22>().bdry<1>(), sdom<23>().bdry<4>());
    makePair(sdom<19>().bdry<4>(), sdom<24>().bdry<1>());
    makePair(sdom<20>().bdry<4>(), sdom<25>().bdry<1>());
    makePair(sdom<24>().bdry<3>(), sdom<25>().bdry<0>());
    
    makePair(sdom<19>().bdry<5>(), sdom<28>().bdry<2>());
    makePair(sdom<20>().bdry<5>(), sdom<29>().bdry<2>());
    makePair(sdom<21>().bdry<5>(), sdom<30>().bdry<2>());
    makePair(sdom<22>().bdry<5>(), sdom<31>().bdry<2>());
    makePair(sdom<23>().bdry<5>(), sdom<32>().bdry<2>());
    
    makePair(sdom<26>().bdry<4>(), sdom<27>().bdry<1>());
    makePair(sdom<27>().bdry<3>(), sdom<28>().bdry<0>());
    makePair(sdom<28>().bdry<3>(), sdom<29>().bdry<0>());
    makePair(sdom<29>().bdry<3>(), sdom<30>().bdry<0>());
    makePair(sdom<30>().bdry<3>(), sdom<31>().bdry<0>());
    makePair(sdom<31>().bdry<1>(), sdom<32>().bdry<4>());
    
    Vector3d transl(0., 0., 2.*(dim(0)+dim(2)+dim(4)));
    makePair(sdom< 0>().bdry<2>(), sdom<26>().bdry<5>(), transl);
    makePair(sdom< 1>().bdry<2>(), sdom<27>().bdry<5>(), transl);
    makePair(sdom< 2>().bdry<2>(), sdom<28>().bdry<5>(), transl);
    makePair(sdom< 3>().bdry<2>(), sdom<29>().bdry<5>(), transl);
    makePair(sdom< 4>().bdry<2>(), sdom<30>().bdry<5>(), transl);
    makePair(sdom< 5>().bdry<2>(), sdom<31>().bdry<5>(), transl);
    makePair(sdom< 6>().bdry<2>(), sdom<32>().bdry<5>(), transl);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

