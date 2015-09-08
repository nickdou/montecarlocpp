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

//----------------------------------------
//  Domain
//----------------------------------------

Domain::Domain()
{}

Domain::Domain(const Domain& dom)
{}

Domain& Domain::operator=(const Domain& dom)
{
    return *this;
}

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

ArrayXXd Domain::average(const ArrayXXd& data) const
{
    return data;
}

std::ostream& operator<<(std::ostream& os, const Domain& dom)
{
    return os << dom.info();
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

void Domain::AddSdomF::operator()(const Subdomain& bdry) const
{
    dom_->addSdom(&bdry);
}

void Domain::AddSdomF::operator()(const EmitSubdomain& bdry) const
{
    dom_->addSdom(&bdry);
}

//----------------------------------------
//  Domain implementations
//----------------------------------------

BulkDomain::BulkDomain()
: Domain()
{}

std::string BulkDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "BulkDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  div: " << div_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void BulkDomain::init()
{
    makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(dim_(0), 0., 0.));
    addSdom(&sdom_);
}

BulkDomain::BulkDomain(const Vector3d& dim, const Vector3l& div, double dT)
: Domain(), dim_(dim), div_(div), dT_(dT),
sdom_(Vector3d::Zero(), dim.asDiagonal(), div, Vector3d(-dT/dim(0), 0., 0.))
{
    init();
}

BulkDomain::BulkDomain(const BulkDomain& dom)
: Domain(), sdom_(dom.sdom_), dim_(dom.dim_), div_(dom.div_), dT_(dom.dT_)
{
    init();
}

BulkDomain& BulkDomain::operator=(const BulkDomain& dom)
{
    Domain::operator=(dom);
    sdom_ = dom.sdom_;
    dim_ = dom.dim_;
    div_ = dom.div_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd BulkDomain::checkpoints() const
{
    return 0.5*dim_;
}

FilmDomain::FilmDomain()
: Domain()
{}

std::string FilmDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "FilmDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  div: " << div_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void FilmDomain::init()
{
    makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(dim_(0), 0., 0.));
    addSdom(&sdom_);
}

FilmDomain::FilmDomain(const Vector3d& dim, const Vector3l& div, double dT)
: Domain(), dim_(dim), div_(div), dT_(dT),
sdom_(Vector3d::Zero(), dim.asDiagonal(), div, Vector3d(-dT/dim(0), 0., 0.))
{
    init();
}

FilmDomain::FilmDomain(const FilmDomain& dom)
: Domain(), sdom_(dom.sdom_), dim_(dom.dim_), div_(dom.div_), dT_(dom.dT_)
{
    init();
}

FilmDomain& FilmDomain::operator=(const FilmDomain& dom)
{
    Domain::operator=(dom);
    sdom_ = dom.sdom_;
    dim_ = dom.dim_;
    div_ = dom.div_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd FilmDomain::checkpoints() const
{
    return 0.5*dim_;
}

HexDomain::HexDomain()
: Domain()
{}

std::string HexDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "HexDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void HexDomain::init()
{
    makePair(sdom_.bdry<0>(), sdom_.bdry<1>(), Vector3d(dim_(0), 0., 0.));
    addSdom(&sdom_);
}

HexDomain::HexDomain(const Vector4d& dim, double dT)
: Domain(), dim_(dim), dT_(dT)
{
    Matrix3Xd mat(3, 6);
    mat.transpose() <<
        dim(0),        0.,            0.,
            0.,    dim(1),       -dim(3),
            0., 2.*dim(1),            0.,
            0., 2.*dim(1),        dim(2),
            0.,    dim(1), dim(2)+dim(3),
            0.,        0.,        dim(2);
    sdom_ = Sdom(Vector3d::Zero(), mat, 0, Vector3d(-dT/dim(0), 0., 0.));
}

HexDomain::HexDomain(const HexDomain& dom)
: Domain(), sdom_(dom.sdom_), dim_(dom.dim_), dT_(dom.dT_)
{
    init();
}

HexDomain& HexDomain::operator=(const HexDomain& dom)
{
    Domain::operator=(dom);
    sdom_ = dom.sdom_;
    dim_ = dom.dim_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd HexDomain::checkpoints() const
{
    Matrix3Xd pts(3, 2);
    pts.transpose() <<
        0.5*dim_(0), 0.5*dim_(1), 0.5*dim_(2),
        0.5*dim_(0), 1.5*dim_(1), 0.5*dim_(2);
    
    return pts;
}

PyrDomain::PyrDomain()
: Domain()
{}

std::string PyrDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "PyrDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void PyrDomain::init()
{
    addSdom(&sdom_);
}

PyrDomain::PyrDomain(const Vector3d& dim, double dT)
: Domain(), dim_(dim), dT_(dT)
{
    Matrix3Xd mat(3,4);
    mat.transpose() <<
        dim(0), 0.5*dim(1), 0.5*dim(2),
            0.,     dim(1),         0.,
            0.,     dim(1),     dim(2),
            0.,         0.,     dim(2);
    sdom_ = Sdom(Vector3d::Zero(), mat, 0, Vector3d(-dT/dim(0), 0., 0.));
}

PyrDomain::PyrDomain(const PyrDomain& dom)
: Domain(), sdom_(dom.sdom_), dim_(dom.dim_), dT_(dom.dT_)
{
    init();
}

PyrDomain& PyrDomain::operator=(const PyrDomain& dom)
{
    Domain::operator=(dom);
    sdom_ = dom.sdom_;
    dim_ = dom.dim_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd PyrDomain::checkpoints() const
{
    return 0.5*dim_;
}

JctDomain::JctDomain()
: Domain()
{}

std::string JctDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "JctDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  div: " << div_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void JctDomain::init()
{
    makePair(sdom<0>().bdry<4>(), sdom<1>().bdry<1>());
    makePair(sdom<0>().bdry<4>(), sdom<2>().bdry<1>());
    makePair(sdom<1>().bdry<3>(), sdom<2>().bdry<0>());
    
    Vector3d transl(2.*dim_(0), 0., 0.);
    makePair(sdom<0>().bdry<0>(), sdom<0>().bdry<3>(), transl);
    makePair(sdom<1>().bdry<0>(), sdom<2>().bdry<3>(), transl);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

JctDomain::JctDomain(const Vector4d& dim, const Vector4l& div, double dT)
: Domain(),
sdomCont_(Sdom<0>::type(Vector3d(0., 0., 0.),
                        Diagonal3d(2.*dim(0), dim(1), dim(3)),
                        Vector3l(2*div(0), div(1), div(3)),
                        Vector3d(-dT/(2.*dim(0)), 0., 0.)),
          Sdom<1>::type(Vector3d(0., dim(1), 0.),
                        Diagonal3d(dim(0), dim(2), dim(3)),
                        Vector3l(div(0), div(2), div(3)),
                        Vector3d(-dT/(2.*dim(0)), 0., 0.)),
          Sdom<2>::type(Vector3d(dim(0), dim(1), 0.),
                        Diagonal3d(dim(0), dim(2), dim(3)),
                        Vector3l(div(0), div(2), div(3)),
                        Vector3d(-dT/(2.*dim(0)), 0., 0.))),
dim_(dim), div_(div), dT_(dT)
{
    init();
}

JctDomain::JctDomain(const JctDomain& dom)
: Domain(), sdomCont_(dom.sdomCont_),
dim_(dom.dim_), div_(dom.div_), dT_(dom.dT_)
{
    init();
}

JctDomain& JctDomain::operator=(const JctDomain& dom)
{
    Domain::operator=(dom);
    sdomCont_ = dom.sdomCont_;
    dim_ = dom.dim_;
    div_ = dom.div_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd JctDomain::checkpoints() const
{
    Matrix3Xd pts(3, 3);
    pts.transpose() <<
            dim_(0),         0.5*dim_(1), 0.5*dim_(3),
        0.5*dim_(0), dim_(1)+0.5*dim_(2), 0.5*dim_(3),
        1.5*dim_(0), dim_(1)+0.5*dim_(2), 0.5*dim_(3);
    
    return pts;
}

TeeDomain::TeeDomain()
: Domain()
{}

std::string TeeDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "TeeDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  div: " << div_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void TeeDomain::init()
{
    makePair(sdom<0>().bdry<3>(), sdom<1>().bdry<0>());
    makePair(sdom<1>().bdry<4>(), sdom<2>().bdry<1>());
    makePair(sdom<1>().bdry<3>(), sdom<3>().bdry<0>());
    makePair(sdom<0>().bdry<0>(), sdom<3>().bdry<3>(),
             Vector3d(2.*dim_(0) + dim_(1), 0., 0.));
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

TeeDomain::TeeDomain(const Vector5d& dim, const Vector5l& div, double dT)
: Domain(),
sdomCont_(Sdom<0>::type(Vector3d::Zero(),
                        Diagonal3d(dim(0), dim(2), dim(4)),
                        Vector3l(div(0), div(2), div(4)),
                        Vector3d(-dT/(2.*dim(0) + dim(1)), 0., 0.)),
          Sdom<1>::type(Vector3d(dim(0), 0., 0.),
                        Diagonal3d(dim(1), dim(2), dim(4)),
                        Vector3l(div(1), div(2), div(4)),
                        Vector3d(-dT/(2.*dim(0) + dim(1)), 0., 0.)),
          Sdom<2>::type(Vector3d(dim(0), dim(2), 0.),
                        Diagonal3d(dim(1), dim(3), dim(4)),
                        Vector3l(div(1), div(3), div(4)),
                        Vector3d(-dT/(2.*dim(0) + dim(1)), 0., 0.)),
          Sdom<3>::type(Vector3d(dim(0) + dim(1), 0., 0.),
                        Diagonal3d(dim(0), dim(2), dim(4)),
                        Vector3l(div(0), div(2), div(4)),
                        Vector3d(-dT/(2.*dim(0) + dim(1)), 0., 0.))),
dim_(dim), div_(div), dT_(dT)
{
    init();
}

TeeDomain::TeeDomain(const TeeDomain& dom)
: Domain(), sdomCont_(dom.sdomCont_),
dim_(dom.dim_), div_(dom.div_), dT_(dom.dT_)
{
    init();
}

TeeDomain& TeeDomain::operator=(const TeeDomain& dom)
{
    Domain::operator=(dom);
    sdomCont_ = dom.sdomCont_;
    dim_ = dom.dim_;
    div_ = dom.div_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd TeeDomain::checkpoints() const
{
    Matrix3Xd pts(3, 4);
    pts.transpose() <<
                0.5*dim_(0),         0.5*dim_(2), 0.5*dim_(4),
        dim_(0)+0.5*dim_(1),         0.5*dim_(2), 0.5*dim_(4),
        dim_(0)+0.5*dim_(1), dim_(2)+0.5*dim_(3), 0.5*dim_(4),
        1.5*dim_(0)+dim_(1),         0.5*dim_(2), 0.5*dim_(4);
    
    return pts;
}

TubeDomain::TubeDomain()
: Domain()
{}

std::string TubeDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "TubeDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  div: " << div_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void TubeDomain::init()
{
    makePair(sdom<1>().bdry<1>(), sdom<2>().bdry<4>());
    makePair(sdom<1>().bdry<2>(), sdom<0>().bdry<5>());
    
    Vector3d transl(dim_(0), 0., 0.);
    makePair(sdom<0>().bdry<0>(), sdom<0>().bdry<3>(), transl);
    makePair(sdom<1>().bdry<0>(), sdom<1>().bdry<3>(), transl);
    makePair(sdom<2>().bdry<0>(), sdom<2>().bdry<3>(), transl);
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

TubeDomain::TubeDomain(const Vector4d& dim, const Vector4l& div, double dT)
: Domain(),
sdomCont_(Sdom<0>::type(Vector3d(0., dim(1), 0.),
                        Diagonal3d(dim(0), dim(3), dim(2)),
                        Vector3l(div(0), div(3), div(2)),
                        Vector3d(-dT/dim(0), 0., 0.)),
          Sdom<1>::type(Vector3d(0., dim(1), dim(2)),
                        Diagonal3d(dim(0), dim(3), dim(3)),
                        Vector3l(div(0), div(3), div(3)),
                        Vector3d(-dT/dim(0), 0., 0.)),
          Sdom<2>::type(Vector3d(0., 0., dim(2)),
                        Diagonal3d(dim(0), dim(1), dim(3)),
                        Vector3l(div(0), div(1), div(3)),
                        Vector3d(-dT/dim(0), 0., 0.))),
dim_(dim), div_(div), dT_(dT)
{
    init();
}

TubeDomain::TubeDomain(const TubeDomain& dom)
: Domain(), sdomCont_(dom.sdomCont_),
dim_(dom.dim_), div_(dom.div_), dT_(dom.dT_)
{
    init();
}

TubeDomain& TubeDomain::operator=(const TubeDomain& dom)
{
    Domain::operator=(dom);
    sdomCont_ = dom.sdomCont_;
    dim_ = dom.dim_;
    div_ = dom.div_;
    dT_ = dom.dT_;
    
    init();
    return *this;
}

Matrix3Xd TubeDomain::checkpoints() const
{
    Matrix3Xd pts(3, 3);
    pts.transpose() <<
        0.5*dim_(0), dim_(1)+0.5*dim_(3),         0.5*dim_(2),
        0.5*dim_(0), dim_(1)+0.5*dim_(3), dim_(2)+0.5*dim_(3),
        0.5*dim_(0),         0.5*dim_(1), dim_(2)+0.5*dim_(3);
    
    return pts;
}

//----------------------------------------
//  Nanotruss domain
//----------------------------------------

OctetDomain::OctetDomain()
: Domain()
{}

std::string OctetDomain::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "OctetDomain " << static_cast<const Domain*>(this) << std::endl;
    ss << "  dim: " << dim_.transpose().format(fmt) << std::endl;
    ss << "  div: " << div_.transpose().format(fmt) << std::endl;
    ss << "  dT:  " << dT_;
    return ss.str();
}

void OctetDomain::init()
{
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
        Vector3d transl = Vector3d(dim_(0)/2., 0., -dim_(0)/2.);
        Matrix3d rot = Diagonal3d(-1., 1., 1.);
        
        makePair(sdom< 0>().bdry<1>(), sdom<41>().bdry<1>(), transl, rot);
        makePair(sdom< 3>().bdry<1>(), sdom<38>().bdry<1>(), transl, rot);
        makePair(sdom< 6>().bdry<2>(), sdom<35>().bdry<2>(), transl, rot);
        makePair(sdom< 7>().bdry<2>(), sdom<34>().bdry<2>(), transl, rot);
        makePair(sdom< 8>().bdry<2>(), sdom<33>().bdry<2>(), transl, rot);
    }
    
    fusion::for_each(sdomCont_, AddSdomF(this));
}

OctetDomain::OctetDomain(const Vector4d& dim, const Vector4l& div, double dT)
: Domain(), dim_(dim), div_(div), dT_(dT), pts_(3, 52)
{
    double sqrt2 = std::sqrt(2.);
    double s = dim(0);
    double a = PI/5.*(dim(1)+dim(2))-(4.-PI)/5.*dim(3);
    double b = a/4.;
    double t = dim(3);
    
    BOOST_ASSERT_MSG(b > t, "Invalid dimensions");
    BOOST_ASSERT_MSG(a > (sqrt2+1.)*(b+t), "Invalid dimensions");
    BOOST_ASSERT_MSG(s/2. > a+(sqrt2+1.)*(b+t)+t, "Invalid dimensions");
    
    Vector3d o, u, v;
    Matrix3d mat3;
    Matrix3Xd mat4(3, 4);
    Vector3l sdomDiv;
    const Vector3l noDiv = Vector3l::Constant(-1);
    const Vector3d noGrad = Vector3d::Zero();
    const Vector3d gradT(dT/(s-2.*a-2.*(sqrt2+1.)*b-2.*(2.+sqrt2)*t), 0.,
                         -dT/(s-2.*a-2.*(sqrt2+1.)*b-2.*(2.+sqrt2)*t));
    
    long p = 0;
    
    // Horizontal Top
    {
        // 00: XY Prism
        o << (sqrt2+1.)*b+t, b+t, -a;
        mat4.transpose() <<
            0., 0., a,
            sqrt2*t, 0., 0.,
            s/4.-(2.+sqrt2)/2.*b-(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            s/4.-(2.+sqrt2)/2.*b-t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom< 0>() = Sdom< 0>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 01: XY Prism
        o << (sqrt2+1.)*b+t, b+t, -a-t;
        mat4.transpose() <<
            0., 0., t,
            sqrt2*t, 0., 0.,
            s/4.-(2.+sqrt2)/2.*b-(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            s/4.-(2.+sqrt2)/2.*b-t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom< 1>() = Sdom< 1>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 02: XY Prism
        o << b+t, b+t, -a-t;
        mat4.transpose() <<
            0., 0., t,
            sqrt2*b, 0., 0.,
            s/4.-(2.-sqrt2)/2.*b-t, s/4.-(2.+sqrt2)/2.*b-t, 0.,
            s/4.-b-t, s/4.-b-t, 0.;
        sdom< 2>() = Sdom< 2>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 03: XY Prism
        o << (sqrt2+1.)*b, b, -a;
        mat4.transpose() <<
            0., 0., a,
            (sqrt2+1.)*t, 0., 0.,
            (sqrt2+1.)*t, t, 0.,
            t, t, 0.;
        sdom< 3>() = Sdom< 3>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 04: XY Prism
        o << (sqrt2+1.)*b, b, -a-t;
        mat4.transpose() <<
            0., 0., t,
            (sqrt2+1.)*t, 0., 0.,
            (sqrt2+1.)*t, t, 0.,
            t, t, 0.;
        sdom< 4>() = Sdom< 4>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 05: XY Parallelepiped
        o << b, b, -a-t;
        mat3.transpose() <<
            0., 0., t,
            sqrt2*b, 0., 0.,
            t, t, 0.;
        sdom< 5>() = Sdom< 5>::type(o, mat3, noDiv, noGrad);
        
        u << t, 0., 0.;
        pts_.col(p++) = o + mat3.col(0)/2. + (mat3.col(2) + u)/3.;
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;
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
        sdom< 6>() = Sdom< 6>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 07: XZ Prism
        o << a, b, 0.;
        mat4.transpose() <<
            0., t, 0.,
            sqrt2*t, 0., 0.,
            (sqrt2+1.)/2.*b+(sqrt2+1.)*t, 0., -(sqrt2+1.)/2.*b-t,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdom< 7>() = Sdom< 7>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 08: XZ Prism
        o << (sqrt2+1.)*(b+t), b, 0.;
        mat4.transpose() <<
            0., t, 0.,
            a-(sqrt2+1.)*(b+t), 0., 0., // CONSTRAINT
            a-(sqrt2+1.)/2.*b-t/sqrt2, 0., -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t,
            0., 0., -a-t;
        sdom< 8>() = Sdom< 8>::type(o, mat4, noDiv(0), noGrad);
        
        v << 0., 0., -a;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(1) + v)/2.;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;

        // 09: XZ Prism
        o << b+t, b, -a-t;
        mat4.transpose() <<
            0., t, 0.,
            sqrt2*(b+t), 0., 0.,
            (sqrt2-1.)/2.*b+t/sqrt2, 0., -(sqrt2+1.)/2.*b-t/sqrt2,
            0., 0., -b;
        sdom< 9>() = Sdom< 9>::type(o, mat4, noDiv(0), noGrad);
        
        u << sqrt2*b, 0., 0.;
        v << 0., 0., -b+t;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2) + u)/2.;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;
        pts_.col(p++) = o + (mat4.col(0) + u + v)/2.;

        // 10: XZ Prism
        o << b+t, b, -a-b-t;
        mat4.transpose() <<
            0., t, 0.,
            (sqrt2-1.)/2.*b+t/sqrt2, 0., -(sqrt2-1.)/2.*b-t/sqrt2,
            (sqrt2-1.)/2.*b, 0., -(sqrt2-1.)/2.*b-sqrt2*t,
            0., 0., -sqrt2*t;
        sdom<10>() = Sdom<10>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 11: XZ Prism
        o << b+t, 0., -a-b-t;
        mat4.transpose() <<
            0., b, 0.,
            (sqrt2-1.)/2.*b+t/sqrt2, 0., -(sqrt2-1.)/2.*b-t/sqrt2,
            (sqrt2-1.)/2.*b, 0., -(sqrt2-1.)/2.*b-sqrt2*t,
            0., 0., -sqrt2*t;
        sdom<11>() = Sdom<11>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 12: XY TriangularPrism
        o << b+t, b, -a-b;
        mat3.transpose() <<
            0., t, 0.,
            -t, 0., 0.,
            0., 0., b-t;  // CONSTRAINT
        sdom<12>() = Sdom<12>::type(o, mat3, noDiv, noGrad);
        
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 13: YZ Pyramid
        o << b+t, b, -a-b-t;
        mat4.transpose() <<
            -t, 0., t,
            0., 0., t,
            0., t, t,
            0., t, 0.;
        sdom<13>() = Sdom<13>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + mat4.rowwise().sum()/5.;

        // 14: XY TriangularPrism
        o << b+t, b, -a-b-(sqrt2+1.)*t;
        mat3.transpose() <<
            0., t, 0.,
            -t, 0., t,
            0., 0., sqrt2*t;
        sdom<14>() = Sdom<14>::type(o, mat3, noDiv, noGrad);
        
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 15: XY Prism
        o << b+t, 0., -a-b-(sqrt2+1.)*t;
        mat4.transpose() <<
            0., 0., sqrt2*t,
            0., b, 0.,
            -t, b, t,
            -b-t, 0., b+t;
        sdom<15>() = Sdom<15>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;
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
        
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;

        // 17: XZ Parallelepiped
        o << s/4.+a/2., b, -s/4.+a/2.;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<17>() = Sdom<17>::type(o, mat3, sdomDiv, gradT);
        
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;

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
        pts_.col(p++) = o + (mat3.col(1) + mat3.col(2) + u)/2.;
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;

        // 19: XZ Parallelepiped
        o << s/4.-a/2.-t/sqrt2, b, -s/4.-a/2.-t/sqrt2;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<19>() = Sdom<19>::type(o, mat3, sdomDiv, gradT);
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;

        // 20: XZ Parallelepiped
        o << s/4.-a/2.-t/sqrt2, 0., -s/4.-a/2.-t/sqrt2;
        mat3.transpose() <<
            t/sqrt2, 0., t/sqrt2,
            0., b, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(2), div(0);
        sdom<20>() = Sdom<20>::type(o, mat3, sdomDiv, gradT);
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;
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
        
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;

        // 22: XZ Parallelepiped
        o << s/4.+a/2.+t/sqrt2, b, -s/4.+a/2.+t/sqrt2;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<22>() = Sdom<22>::type(o, mat3, sdomDiv, gradT);
        
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;

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
        pts_.col(p++) = o + (mat3.col(1) + mat3.col(2) + u)/2.;
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;

        // 24: XZ Parallelepiped
        o << s/4.-a/2., b, -s/4.-a/2.;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., t, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(3), div(0);
        sdom<24>() = Sdom<24>::type(o, mat3, sdomDiv, gradT);
        
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;

        // 25: XZ Parallelepiped
        o << s/4.-a/2., 0., -s/4.-a/2.;
        mat3.transpose() <<
            -t/sqrt2, 0., -t/sqrt2,
            0., b, 0.,
            s/4.-a/2.-(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            -s/4.+a/2.+(sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdomDiv << div(3), div(2), div(0);
        sdom<25>() = Sdom<25>::type(o, mat3, sdomDiv, gradT);
        
        pts_.col(p++) = o + mat3.rowwise().sum()/2.;
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
        sdom<26>() = Sdom<26>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 27: XY TriangularPrism
        o << s/2.-b-t, b, -s/2.+a+b+(sqrt2+1.)*t;
        mat3.transpose() <<
            0., t, 0.,
            t, 0., -t,
            0., 0., -sqrt2*t;
        sdom<27>() = Sdom<27>::type(o, mat3, noDiv, noGrad);
        
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 28: YZ Pyramid
        o << s/2.-b-t, b, -s/2.+a+b+t;
        mat4.transpose() <<
            t, 0., -t,
            0., 0., -t,
            0., t, -t,
            0., t, 0.;
        sdom<28>() = Sdom<28>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + mat4.rowwise().sum()/5.;

        // 29: XY TriangularPrism
        o << s/2.-b-t, b, -s/2.+a+b;
        mat3.transpose() <<
            0., t, 0.,
            t, 0., 0.,
            0., 0., -b+t;  // CONSTRAINT
        sdom<29>() = Sdom<29>::type(o, mat3, noDiv, noGrad);
        
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1))/3. + mat3.col(2)/2.;

        // 30: XZ Prism
        o << s/2.-b-t, 0., -s/2.+a+b+t;
        mat4.transpose() <<
            0., b, 0.,
            -(sqrt2-1.)/2.*b-t/sqrt2, 0., (sqrt2-1.)/2.*b+t/sqrt2,
            -(sqrt2-1.)/2.*b, 0., (sqrt2-1.)/2.*b+sqrt2*t,
            0., 0., sqrt2*t;
        sdom<30>() = Sdom<30>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 31: XZ Prism
        o << s/2.-b-t, b, -s/2.+a+b+t;
        mat4.transpose() <<
            0., t, 0.,
            -(sqrt2-1.)/2.*b-t/sqrt2, 0., (sqrt2-1.)/2.*b+t/sqrt2,
            -(sqrt2-1.)/2.*b, 0., (sqrt2-1.)/2.*b+sqrt2*t,
            0., 0., sqrt2*t;
        sdom<31>() = Sdom<31>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 32: XZ Prism
        o << s/2.-b-t, b, -s/2.+a+t;
        mat4.transpose() <<
            0., t, 0.,
            -sqrt2*(b+t), 0., 0.,
            -(sqrt2-1.)/2.*b-t/sqrt2, 0., (sqrt2+1.)/2.*b+t/sqrt2,
            0., 0., b;
        sdom<32>() = Sdom<32>::type(o, mat4, noDiv(0), noGrad);
        
        u << -sqrt2*b, 0., 0.;
        v << 0., 0., b-t;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2) + u)/2.;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;
        pts_.col(p++) = o + (mat4.col(0) + u + v)/2.;

        // 33: XZ Prism
        o << s/2.-(sqrt2+1.)*(b+t), b, -s/2.;
        mat4.transpose() <<
            0., t, 0.,
            -a+(sqrt2+1.)*(b+t), 0., 0., // CONSTRAINT
            -a+(sqrt2+1.)/2.*b+t/sqrt2, 0., (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t,
            0., 0., a+t;
        sdom<33>() = Sdom<33>::type(o, mat4, noDiv(0), noGrad);
        
        v << 0., 0., a;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(1) + v)/2.;
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2) + v)/2.;

        // 34: XZ Prism
        o << s/2.-a, b, -s/2.;
        mat4.transpose() <<
            0., t, 0.,
            -sqrt2*t, 0., 0.,
            -(sqrt2+1.)/2.*b-(sqrt2+1.)*t, 0., (sqrt2+1.)/2.*b+t,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdom<34>() = Sdom<34>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 35: XZ Prism
        o << s/2.-a, 0., -s/2.;
        mat4.transpose() <<
            0., b, 0.,
            -sqrt2*t, 0., 0.,
            -(sqrt2+1.)/2.*b-(sqrt2+1.)*t, 0., (sqrt2+1.)/2.*b+t,
            -(sqrt2+1.)/2.*b-(2.+sqrt2)/2.*t, 0.,
            (sqrt2+1.)/2.*b+(2.+sqrt2)/2.*t;
        sdom<35>() = Sdom<35>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;
    }

    // Horizontal Bottom
    {
        // 36: XY Parallelepiped
        o << s/2.-b, b, -s/2.+a+t;
        mat3.transpose() <<
            0., 0., -t,
            -sqrt2*b, 0., 0.,
            -t, t, 0.;
        sdom<36>() = Sdom<36>::type(o, mat3, noDiv, noGrad);
        
        u << -t, 0., 0.;
        pts_.col(p++) = o + mat3.col(0)/2. + (mat3.col(2) + u)/3.;
        pts_.col(p++) = o + (mat3.col(0) + mat3.col(1) + mat3.col(2) + u)/2.;

        // 37: XY Prism
        o << s/2.-(sqrt2+1.)*b, b, -s/2.+a+t;
        mat4.transpose() <<
            0., 0., -t,
            -(sqrt2+1.)*t, 0., 0.,
            -(sqrt2+1.)*t, t, 0.,
            -t, t, 0.;
        sdom<37>() = Sdom<37>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 38: XY Prism
        o << s/2.-(sqrt2+1.)*b, b, -s/2.+a;
        mat4.transpose() <<
            0., 0., -a,
            -(sqrt2+1.)*t, 0., 0.,
            -(sqrt2+1.)*t, t, 0.,
            -t, t, 0.;
        sdom<38>() = Sdom<38>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 39: XY Prism
        o << s/2.-b-t, b+t, -s/2.+a+t;
        mat4.transpose() <<
            0., 0., -t,
            -sqrt2*b, 0., 0.,
            -s/4.+(2.-sqrt2)/2.*b+t, s/4.-(2.+sqrt2)/2.*b-t, 0.,
            -s/4.+b+t, s/4.-b-t, 0.;
        sdom<39>() = Sdom<39>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 40: XY Prism
        o << s/2.-(sqrt2+1.)*b-t, b+t, -s/2.+a+t;
        mat4.transpose() <<
            0., 0., -t,
            -sqrt2*t, 0., 0.,
            -s/4.+(2.+sqrt2)/2.*b+(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            -s/4.+(2.+sqrt2)/2.*b+t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom<40>() = Sdom<40>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;

        // 41: XY Prism
        o << s/2.-(sqrt2+1.)*b-t, b+t, -s/2.+a;
        mat4.transpose() <<
            0., 0., -a,
            -sqrt2*t, 0., 0.,
            -s/4.+(2.+sqrt2)/2.*b+(2.-sqrt2)/2.*t, s/4.-(2.+sqrt2)/2.*(b+t), 0.,
            -s/4.+(2.+sqrt2)/2.*b+t, s/4.-(2.+sqrt2)/2.*b-t, 0.;
        sdom<41>() = Sdom<41>::type(o, mat4, noDiv(0), noGrad);
        
        pts_.col(p++) = o + (mat4.col(0) + mat4.col(2))/2.;
    }
    BOOST_ASSERT_MSG(p == pts_.cols(), "Incorrect number of checkpoints");
    
    init();
}

OctetDomain::OctetDomain(const OctetDomain& dom)
: Domain(), sdomCont_(dom.sdomCont_),
dim_(dom.dim_), div_(dom.div_), dT_(dom.dT_), pts_(dom.pts_)
{
    init();
}

OctetDomain& OctetDomain::operator=(const OctetDomain& dom)
{
    Domain::operator=(dom);
    sdomCont_ = dom.sdomCont_;
    dim_ = dom.dim_;
    div_ = dom.div_;
    dT_ = dom.dT_;
    pts_ = dom.pts_;
    
    init();
    return *this;
}

Matrix3Xd OctetDomain::checkpoints() const
{
    return pts_;
}

ArrayXXd OctetDomain::average(const ArrayXXd& data) const
{
    Eigen::Array<double, 1, Eigen::Dynamic> weight;
    weight = Field(1, this, WeightF(*this)).data().row(0);
    return (data.rowwise() * weight).rowwise().sum() / weight.sum();
}

OctetDomain::WeightF::WeightF(const OctetDomain& dom)
{
    BOOST_ASSERT_MSG(dom.isInit(), "Domain is not initialized");
    
    for (int i = 16; i < 26; ++i)
    {
        sdomAvg_.push_back( dom.sdomPtrs().at(i) );
    }
}

VectorXd OctetDomain::WeightF::operator()(const Subdomain* sdom,
                                          const Vector3l& index) const
{
    Subdomain::Pointers::const_iterator beg = sdomAvg_.begin();
    Subdomain::Pointers::const_iterator end = sdomAvg_.end();
    
    if (index(2) == 0 && std::find(beg, end, sdom) != end)
    {
        return VectorXd::Constant(1, sdom->cellVol(index));
    }
    return VectorXd::Zero(1);
}

//OctetDomain::OctetDomain(const Eigen::Matrix<double, 5, 1>& dim,
//                         const Eigen::Matrix<long, 5, 1>& div,
//                         double deltaT)
//: Domain(Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//sdomCont_(
//          Sdom< 0>::type(Vector3d(-dim(4), 0., -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 1>::type(Vector3d(-dim(4), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 2>::type(Vector3d(0., dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(2), dim(4), dim(0)),
//                         Vector3l(div(2), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 3>::type(Vector3d(dim(2), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 4>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(2)-dim(4), dim(4), dim(0)),
//                         Vector3l(div(2)-div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 5>::type(Vector3d(2.*dim(2), dim(3), -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 6>::type(Vector3d(2.*dim(2), 0., -dim(0)-dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom< 7>::type(Vector3d(0., dim(3), -dim(2)-dim(4)),
//                         Diagonal3d(dim(2), dim(4), dim(4)),
//                         Vector3l(div(2), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 8>::type(Vector3d(dim(2), dim(3), -dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom< 9>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(2)-dim(4)),
//                         Diagonal3d(dim(2)-dim(4), dim(4), dim(4)),
//                         Vector3l(div(2)-div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<10>::type(Vector3d(2.*dim(2), dim(3), -dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<11>::type(Vector3d(2.*dim(2), 0., -dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(3), dim(4)),
//                         Vector3l(div(4), div(3), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<12>::type(Vector3d(0., dim(3)+dim(4), -dim(2)-dim(4)),
//                         Diagonal3d(dim(2), dim(1), dim(4)),
//                         Vector3l(div(2), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<13>::type(Vector3d(dim(2), dim(3)+dim(4), -dim(2)-dim(4)),
//                         Diagonal3d(dim(4), dim(1), dim(4)),
//                         Vector3l(div(4), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom<14>::type(Vector3d(dim(2), dim(3), -dim(2)),
//                         Diagonal3d(dim(4), dim(4), 2.*dim(2)),
//                         Vector3l(div(4), div(4), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<15>::type(Vector3d(dim(2)+dim(4), dim(3), -dim(2)),
//                         Diagonal3d(dim(2)-dim(4), dim(4), 2.*dim(2)),
//                         Vector3l(div(2)-div(4), div(4), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<16>::type(Vector3d(2.*dim(2), dim(3), -dim(2)),
//                         Diagonal3d(dim(4), dim(4), 2.*dim(2)),
//                         Vector3l(div(4), div(4), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<17>::type(Vector3d(2.*dim(2), 0., -dim(2)),
//                         Diagonal3d(dim(4), dim(3), 2.*dim(2)),
//                         Vector3l(div(4), div(3), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<18>::type(Vector3d(dim(2), dim(3)+dim(4), -dim(2)),
//                         Diagonal3d(dim(4), dim(1), 2.*dim(2)),
//                         Vector3l(div(4), div(1), 2*div(2)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom<19>::type(Vector3d(0., dim(3), dim(2)),
//                         Diagonal3d(dim(2), dim(4), dim(4)),
//                         Vector3l(div(2), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<20>::type(Vector3d(dim(2), dim(3), dim(2)),
//                         Diagonal3d(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<21>::type(Vector3d(dim(2)+dim(4), dim(3), dim(2)),
//                         Diagonal3d(dim(2)-dim(4), dim(4), dim(4)),
//                         Vector3l(div(2)-div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<22>::type(Vector3d(2.*dim(2), dim(3), dim(2)),
//                         Diagonal3d(dim(4), dim(4), dim(4)),
//                         Vector3l(div(4), div(4), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<23>::type(Vector3d(2.*dim(2), 0., dim(2)),
//                         Diagonal3d(dim(4), dim(3), dim(4)),
//                         Vector3l(div(4), div(3), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<24>::type(Vector3d(0., dim(3)+dim(4), dim(2)),
//                         Diagonal3d(dim(2), dim(1), dim(4)),
//                         Vector3l(div(2), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<25>::type(Vector3d(dim(2), dim(3)+dim(4), dim(2)),
//                         Diagonal3d(dim(4), dim(1), dim(4)),
//                         Vector3l(div(4), div(1), div(4)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          
//          Sdom<26>::type(Vector3d(-dim(4), 0., dim(2)+dim(4)),
//                         Diagonal3d(dim(4), dim(3), dim(0)),
//                         Vector3l(div(4), div(3), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<27>::type(Vector3d(-dim(4), dim(3), dim(2)+dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<28>::type(Vector3d(0., dim(3), dim(2)+dim(4)),
//                         Diagonal3d(dim(2), dim(4), dim(0)),
//                         Vector3l(div(2), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<29>::type(Vector3d(dim(2), dim(3), dim(2)+dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<30>::type(Vector3d(dim(2)+dim(4), dim(3), dim(2)+dim(4)),
//                         Diagonal3d(dim(2)-dim(4), dim(4), dim(0)),
//                         Vector3l(div(2)-div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<31>::type(Vector3d(2.*dim(2), dim(3), dim(2)+dim(4)),
//                         Diagonal3d(dim(4), dim(4), dim(0)),
//                         Vector3l(div(4), div(4), div(0)),
//                         Vector3d(0., 0., -deltaT/(2.*(dim(0)+dim(2)+dim(4))))),
//          Sdom<32>::type(Vector3d(2.*dim(2), 0., dim(2)+dim(4)),
//                         Diagonal3d(dim(4), dim(3), dim(0)),
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

