//
//  subdomain.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 8/7/15.
//
//

#include "subdomain.h"
#include "boundary.h"
#include "phonon.h"
#include "random.h"
#include "constants.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <cmath>

//----------------------------------------
//  Helper functions
//----------------------------------------

namespace
{
    Matrix3d rotMatrix(const Vector3d& n)
    {
        typedef Eigen::Quaternion<double> Quatd;
        return Quatd::FromTwoVectors(Vector3d::UnitZ(), n).matrix();
    }
}

//----------------------------------------
//  Subdomain
//----------------------------------------

Subdomain::Subdomain()
: vol_(0)
{}

Subdomain::Subdomain(double vol, const Vector3d& o, const Matrix3d& mat,
                     const Vector3l& div)
: vol_(vol), o_(o), mat_(mat), inv_(mat.inverse()),
div_(div), shape_(div.cwiseMax(1l)), max_(div.cwiseMax(1l) - Vector3l::Ones()),
eps_(100. * Dbl::epsilon() * mat.colwise().norm().minCoeff())
{
    int dim = static_cast<int>( (div.array() > 0).count() );
    if (dim == 0)
    {
        accum_ = -1;
    }
    else if (dim == 1)
    {
        Vector3l::Index dir;
        div_.maxCoeff(&dir);
        accum_ = static_cast<int>(dir);
    }
    else
    {
        accum_ = 3;
    }
}

Subdomain::Subdomain(const Subdomain& sdom)
: vol_(sdom.vol_), o_(sdom.o_), mat_(sdom.mat_), inv_(sdom.inv_),
div_(sdom.div_), shape_(sdom.shape_), max_(sdom.max_), accum_(sdom.accum_),
eps_(sdom.eps_)
{}

Subdomain& Subdomain::operator=(const Subdomain& sdom)
{
    vol_ = sdom.vol_;
    o_ = sdom.o_;
    mat_ = sdom.mat_;
    inv_ = sdom.inv_;
    div_ = sdom.div_;
    shape_ = sdom.shape_;
    max_ = sdom.max_;
    accum_ = sdom.accum_;
    eps_ = sdom.eps_;
    
    return *this;
}

Subdomain::~Subdomain()
{}

bool Subdomain::isInit() const
{
    if (vol_ == 0 || bdryPtrs_.size() == 0) return false;
    for (Boundary::Pointers::const_iterator b = bdryPtrs_.begin();
         b != bdryPtrs_.end(); ++b)
    {
        if (!(*b)->isInit()) return false;
    }
    return true;
}

bool Subdomain::isInside(const Vector3d& pos) const
{
    for (Boundary::Pointers::const_iterator b = bdryPtrs_.begin();
         b != bdryPtrs_.end(); ++b)
    {
        if ((*b)->distance(pos) < -eps_) return false;
    }
    return true;
}

const Boundary::Pointers& Subdomain::bdryPtrs() const
{
    return bdryPtrs_;
}

const Emitter::Pointers& Subdomain::emitPtrs() const
{
    return emitPtrs_;
}

const Vector3d& Subdomain::origin() const
{
    return o_;
}

const Matrix3d& Subdomain::matrix() const
{
    return mat_;
}

const Vector3l& Subdomain::shape() const
{
    return shape_;
}

int Subdomain::accumFlag() const
{
    return accum_;
}

Vector3d Subdomain::coord(const Vector3d& pos) const
{
    return div_.cast<double>().cwiseProduct(inv_ * (pos - o_));
}

Vector3l Subdomain::coord2index(const Vector3d& coord) const
{
    Vector3l index(static_cast<long>( std::floor(coord(0)) ),
                   static_cast<long>( std::floor(coord(1)) ),
                   static_cast<long>( std::floor(coord(2)) ));
    return index.cwiseMax(0).cwiseMin(max_);
}

const Boundary* Subdomain::advect(Phonon& phn, double vel) const
{
    double minDist = phn.scatNext();
    const Boundary* newBdry = 0;
    
    BOOST_ASSERT_MSG(minDist > 0., "Scattering distance not set");
    
    for (Boundary::Pointers::const_iterator b = bdryPtrs_.begin();
         b != bdryPtrs_.end(); ++b)
    {
        if ((*b)->normal().dot(phn.dir()) >= 0.) continue;
        
        double dist = (*b)->distance(phn);
        if (dist < minDist)
        {
            minDist = dist;
            newBdry = *b;
        }
    }
    phn.move(minDist, vel);
    
    if (minDist < -eps_ || !isInside( phn.pos() ))
    {
        phn.kill();
        return 0;
    }
    
    return newBdry;
}

double Subdomain::vol() const
{
    return vol_;
}

void Subdomain::addBdry(Boundary* bdry)
{
    bdry->sdom(this);
    bdryPtrs_.push_back(bdry);
}

void Subdomain::addBdry(EmitBoundary* bdry)
{
    addBdry( static_cast<Boundary*>(bdry) );
    if (bdry->emitWeight() != 0.)
    {
        emitPtrs_.push_back( static_cast<Emitter*>(bdry) );
    }
}

Subdomain::AddBdryF::AddBdryF(Subdomain* sdom)
: sdom_(sdom)
{}

void Subdomain::AddBdryF::operator()(Boundary& bdry) const
{
    sdom_->addBdry(&bdry);
}

void Subdomain::AddBdryF::operator()(EmitBoundary& bdry) const
{
    sdom_->addBdry(&bdry);
}

EmitSubdomain::EmitSubdomain()
: Subdomain()
{}

EmitSubdomain::EmitSubdomain(double vol, const Vector3d& o, const Matrix3d& mat,
                             const Vector3l& div, const Vector3d& gradT)
: Subdomain(vol, o, mat, div), gradT_(gradT),
rot_( rotMatrix(gradT.normalized()) )
{}

EmitSubdomain::~EmitSubdomain()
{}

const Subdomain* EmitSubdomain::emitSdom() const
{
    return this;
}
const Boundary* EmitSubdomain::emitBdry() const
{
    return 0;
}

double EmitSubdomain::emitWeight() const
{
    return 2. * vol() * gradT_.norm();
}

Vector3d EmitSubdomain::drawDir(Rng& gen) const
{
    return rot_ * drawAniso(gen, true);
}

bool EmitSubdomain::emitSign(const Vector3d&, const Vector3d& dir) const
{
    return dir.dot(gradT_) < 0.;
}

//----------------------------------------
//  Subdomain Implementations
//----------------------------------------

double ParallelepipedImpl::cellVol(const Vector3l&,
                                   const Vector3l& shape, double vol)
{
    return vol / shape.prod();
}

Vector3d ParallelepipedImpl::drawPos(const Vector3d& o,
                                     const Matrix3d& mat, Rng& gen)
{
    static UniformDist01 dist; // [0, 1)
    Vector3d coord(dist(gen), dist(gen), dist(gen));
    return o + mat * coord;
}

double TriangularPrismImpl::cellVol(const Vector3l& index,
                                    const Vector3l& shape, double vol)
{
    Eigen::Vector2d index2 = index.head<2>().cast<double>();
    Eigen::Vector2d shape2 = shape.head<2>().cast<double>();
    double f0 = 1. - index2.cwiseQuotient(shape2).sum();
    if (f0 <= 0.) return 0.;
    
    Eigen::Matrix<long, 2, 1> corner(1l, 1l);
    double f1 = f0 - corner.cast<double>().cwiseQuotient(shape2).sum();
    if (f1 >= 0.) return vol / shape.prod();
    
    typedef Eigen::Matrix<long, 2, 2> Matrix2l;
    static const Matrix2l pts = Matrix2l::Identity();
    
    double frac = std::pow(f0, 2);
    for (int i = 0; i < 2; i++)
    {
        corner = pts.col(i);
        int sign = std::pow(-1., corner.cast<int>().sum());
        double f = f0 - corner.cast<double>().cwiseQuotient(shape2).sum();
        if (f > 0.) frac += sign * std::pow(f, 2);
    }
    return vol * frac / (2.*shape(2));
}

Vector3d TriangularPrismImpl::drawPos(const Vector3d& o,
                                      const Matrix3d& mat, Rng& gen)
{
    static UniformDist01 dist; // [0, 1)
    Vector3d coord(dist(gen), dist(gen), dist(gen));
    if (coord(0) + coord(1) > 1.)
    {
        coord(0) = 1. - coord(0);
        coord(1) = 1. - coord(1);
    }
    return o + mat * coord;
}

double TetrahedronImpl::cellVol(const Vector3l& index,
                                const Vector3l& shape, double vol)
{
    Vector3d index3 = index.cast<double>();
    Vector3d shape3 = shape.cast<double>();
    double f0 = 1. - index3.cwiseQuotient(shape3).sum();
    if (f0 <= 0.) return 0.;
    
    Vector3l corner(1l, 1l, 1l);
    double f1 = f0 - corner.cast<double>().cwiseQuotient(shape3).sum();
    if (f1 >= 0.) return vol / shape.prod();
    
    typedef Eigen::Matrix<long, 3, 6> Matrix36l;
    static const Matrix36l pts = (Matrix36l() <<
                                  1l, 0l, 0l, 0l, 1l, 1l,
                                  0l, 1l, 0l, 1l, 0l, 1l,
                                  0l, 0l, 1l, 1l, 1l, 0l).finished();
    
    double frac = std::pow(f0, 3);
    for (int i = 0; i < 6; i++)
    {
        corner = pts.col(i);
        int sign = std::pow(-1., corner.cast<int>().sum());
        double f = f0 - corner.cast<double>().cwiseQuotient(shape3).sum();
        if (f > 0.) frac += sign * std::pow(f, 3);
    }
    return vol * frac / 6.;
}

Vector3d TetrahedronImpl::drawPos(const Vector3d& o,
                                  const Matrix3d& mat, Rng& gen)
{
    static UniformDist01 dist; // [0, 1)
    Vector3d coord(dist(gen), dist(gen), dist(gen));
    
    if (coord(0) + coord(1) > 1.)
    {
        coord(0) = 1. - coord(0);
        coord(1) = 1. - coord(1);
    }
    
    if (coord(1) + coord(2) > 1.)
    {
        double tmp = coord(2);
        coord(2) = 1. - coord(0) - coord(1);
        coord(1) = 1. - tmp;
    }
    else if (coord.sum() > 1.)
    {
        double tmp = coord(2);
        coord(2) = coord.sum() - 1.;
        coord(0) = 1. - coord(1) - tmp;
    }
    
    return o + mat * coord;
}

Matrix3d PrismImpl::matBase(const Matrix3Xd& mat)
{
    long N = mat.cols();
    return (Matrix3d() << mat.col(1), mat.col(N - 1), mat.col(0)).finished();
}

Eigen::VectorXd PrismImpl::volume(const Matrix3Xd& mat)
{
    long N = mat.cols();
    Eigen::VectorXd vol(N - 2);
    for (long i = 0; i < N - 2; ++i)
    {
        vol(i) = mat.col(i).cross( mat.col(i + 1) ).dot( mat.col(0) ) / 2.;
    }
    return vol;
}

double PrismImpl::cellVol(const Vector3l&, const Vector3l&, double vol)
{
    return vol;
}

Vector3d PrismImpl::drawPos(const Vector3d& o, const Matrix3Xd& mat,
                            const DiscreteDist& volDist, Rng& gen)
{
    long ind = volDist(gen);
    Matrix3d local;
    local << mat.col(ind+1), mat.col(ind+2), mat.col(0);
    
    return TriangularPrismImpl::drawPos(o, local, gen);
}

Matrix3d PyramidImpl::matBase(const Matrix3Xd& mat)
{
    long N = mat.cols();
    return (Matrix3d() << mat.col(1), mat.col(N - 1), mat.col(0)).finished();
}

Eigen::VectorXd PyramidImpl::volume(const Matrix3Xd& mat)
{
    long N = mat.cols();
    Eigen::VectorXd vol(N - 2);
    for (long i = 0; i < N - 2; ++i)
    {
        vol(i) = mat.col(i).cross( mat.col(i + 1) ).dot( mat.col(0) ) / 6.;
    }
    return vol;
}

double PyramidImpl::cellVol(const Vector3l&, const Vector3l&, double vol)
{
    return vol;
}

Vector3d PyramidImpl::drawPos(const Vector3d& o, const Matrix3Xd& mat,
                            const DiscreteDist& volDist, Rng& gen)
{
    long ind = volDist(gen);
    Matrix3d local;
    local << mat.col(ind+1), mat.col(ind+2), mat.col(0);
    
    return TetrahedronImpl::drawPos(o, local, gen);
}
