//
//  boundary.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/29/15.
//
//

#include "subdomain.h"
#include "boundary.h"
#include "phonon.h"
#include "random.h"
#include "constants.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <sstream>
#include <string>
#include <cmath>


//----------------------------------------
//  Helper functions
//----------------------------------------

namespace
{
    Matrix3d reflMatrix(const Vector3d& n)
    {
        return Matrix3d::Identity() - 2.*(n*n.transpose());
    }

    Matrix3d rotMatrix(const Vector3d& n)
    {
        typedef Eigen::Quaternion<double> Quatd;
        return Quatd::FromTwoVectors(Vector3d::UnitZ(), n).matrix();
    }
}

//----------------------------------------
//  Boundary
//----------------------------------------

Boundary::Boundary()
: sdom_(0)
{}

Boundary::Boundary(const Vector3d& o, const Vector3d& n)
: sdom_(0), plane_(n.normalized(), o)
{}

Boundary::Boundary(const Vector3d& o, const Shape& s)
: sdom_(0), plane_(s.normal(), o)
{
    BOOST_ASSERT_MSG(s.isInit(), "Shape not initialized");
}

Boundary::Boundary(const Boundary& bdry)
: sdom_(0), plane_(bdry.plane_)
{}

Boundary& Boundary::operator=(const Boundary& bdry)
{
    sdom_ = 0;
    plane_ = bdry.plane_;
    return *this;
}

Boundary::~Boundary()
{}

bool Boundary::isInit() const
{
    return sdom_ != 0;
}

const Subdomain* Boundary::sdom() const
{
    return sdom_;
}

void Boundary::sdom(const Subdomain* s)
{
    sdom_ = s;
}

Vector3d Boundary::normal() const
{
    return plane_.normal();
}

double Boundary::offset() const
{
    return plane_.offset();
}

Vector3d Boundary::projection(const Vector3d& pos) const
{
    return plane_.projection(pos);
}

double Boundary::distance(const Vector3d& pos) const
{
    return plane_.signedDistance(pos);
}

double Boundary::distance(const Phonon& phn) const
{
    return phn.line().intersection(plane_);
}

//----------------------------------------
//  Shape
//----------------------------------------

Boundary::Shape::~Shape()
{}

Parallelogram::Parallelogram()
: i_(Vector3d::Zero()), j_(Vector3d::Zero())
{}

Parallelogram::Parallelogram(const Vector3d& i, const Vector3d& j)
: i_(i), j_(j)
{}

std::string Parallelogram::type() const
{
    return std::string("P");
}

bool Parallelogram::isInit() const
{
    return area() != 0.;
}

Vector3d Parallelogram::normal() const
{
    return i_.cross(j_).normalized();
}

double Parallelogram::area() const
{
    return i_.cross(j_).norm();
}

Vector3d Parallelogram::drawPos(Rng& gen) const
{
    UniformDist01 dist; // [0, 1)
    double r1 = dist(gen), r2 = dist(gen);
    return r1*i_ + r2*j_;
}

Triangle::Triangle()
: i_(Vector3d::Zero()), j_(Vector3d::Zero())
{}

Triangle::Triangle(const Vector3d& i, const Vector3d& j)
: i_(i), j_(j)
{}

std::string Triangle::type() const
{
    return std::string("T");
}

bool Triangle::isInit() const
{
    return area() != 0.;
}

Vector3d Triangle::normal() const
{
    return i_.cross(j_).normalized();
}

double Triangle::area() const
{
    return i_.cross(j_).norm() / 2.;
}

Vector3d Triangle::drawPos(Rng& gen) const
{
    UniformDist01 dist; // [0, 1)
    double r1 = dist(gen), r2 = dist(gen);
    return r1 + r2 < 1. ? r1*i_ + r2*j_ : (1. - r1)*i_ + (1. - r2)*j_;
}

template<int N>
Polygon<N>::Polygon()
{
    verts_.setZero(3, N - 1);
    areas_.setZero(N - 2);
}

template<int N>
Polygon<N>::Polygon(const Matrix3Xd& verts)
: verts_(verts), areas_(static_cast<long>(N - 2))
{
    BOOST_ASSERT_MSG(verts_.cols() == N - 1, "Incorrect number of vertices");
    
    for (int n = 0; n < N - 2; ++n)
    {
        Vector3d cross = verts_.col(n).cross( verts_.col(n + 1) );
        areas_(n) = cross.norm() / 2.;
        
        BOOST_ASSERT_MSG(areas_(n) > Dbl::min(), "Area too small");
        BOOST_ASSERT_MSG(cross.normalized().isApprox(normal()),
                         "Normals are inconsistent");
    }
    
    double* beg = areas_.data();
    double* end = beg + N - 2;
    areaDist_ = DiscreteDist(beg, end);
}

template<int N>
std::string Polygon<N>::type() const
{
    std::ostringstream ss;
    ss << N;
    return ss.str();
}

template<int N>
bool Polygon<N>::isInit() const
{
    return area() != 0.;
}

template<int N>
Vector3d Polygon<N>::normal() const
{
    return verts_.col(0).cross( verts_.col(1) ).normalized();
}

template<int N>
double Polygon<N>::area() const
{
    return areas_.sum();
}

template<int N>
Vector3d Polygon<N>::drawPos(Rng& gen) const
{
    long n = areaDist_(gen);
    Vector3d i = verts_.col(n), j = verts_.col(n + 1);
    UniformDist01 dist; // [0, 1)
    double r1 = dist(gen), r2 = dist(gen);
    return r1 + r2 < 1. ? r1*i + r2*j : (1. - r1)*i + (1. - r2)*j;
}

template class Polygon<4>;
template class Polygon<5>;
template class Polygon<6>;
template class Polygon<7>;
template class Polygon<8>;
template class Polygon<9>;

//----------------------------------------
//  Non-emitting boundaries
//----------------------------------------

SpecBoundary::SpecBoundary()
: Boundary()
{}

SpecBoundary::SpecBoundary(const Vector3d& o, const Vector3d& n)
: Boundary(o, n), refl_( reflMatrix(normal()) )
{}

SpecBoundary::SpecBoundary(const Vector3d& o, const Shape& s, const double T)
: Boundary(o, s), refl_( reflMatrix(normal()) )
{
    BOOST_ASSERT_MSG(T == 0., "Cannot specify temperature");
}

std::string SpecBoundary::type() const
{
    return std::string("Spec");
}

const Boundary* SpecBoundary::scatter(Phonon& phn, Rng&) const
{
    phn.dir(refl_.selfadjointView<Eigen::Upper>() * phn.dir(), false);
    return this;
}

DiffBoundary::DiffBoundary()
: Boundary()
{}

DiffBoundary::DiffBoundary(const Vector3d& o, const Vector3d& n)
: Boundary(o, n), rot_( rotMatrix(normal()) )
{}

DiffBoundary::DiffBoundary(const Vector3d& o, const Shape& s, const double T)
: Boundary(o, s), rot_( rotMatrix(normal()) )
{
    BOOST_ASSERT_MSG(T == 0., "Cannot specify temperature");
}

std::string DiffBoundary::type() const
{
    return std::string("Diff");
}

const Boundary* DiffBoundary::scatter(Phonon& phn, Rng& gen) const
{
    phn.dir(rot_ * drawAniso(gen, false), true);
    return this;
}

InterBoundary::InterBoundary()
: Boundary()
{}

InterBoundary::InterBoundary(const Vector3d& o, const Vector3d& n)
: Boundary(o, n)
{}

InterBoundary::InterBoundary(const Vector3d& o, const Shape& s, const double T)
: Boundary(o, s)
{
    BOOST_ASSERT_MSG(T == 0., "Cannot specify temperature");
}

InterBoundary::InterBoundary(const InterBoundary& bdry)
: Boundary(bdry), pairs_()
{}

InterBoundary& InterBoundary::operator=(const InterBoundary& bdry)
{
    Boundary::operator=(bdry);
    pairs_.clear();
    return *this;
}

bool InterBoundary::isInit() const
{
    return Boundary::isInit() && !pairs_.empty();
}

std::string InterBoundary::type() const
{
    return std::string("Inter");
}

const Boundary* InterBoundary::scatter(Phonon& phn, Rng&) const
{
    if (pairs_.size() == 1) return pairs_.front();
    for (Boundary::Pointers::const_iterator b = pairs_.begin();
         b != pairs_.end(); ++b)
    {
        if ((*b)->sdom()->isInside( phn.pos() )) return *b;
    }
    phn.kill();
    return 0;
}

void makePair(InterBoundary& bdry1, InterBoundary& bdry2)
{
    BOOST_ASSERT_MSG(bdry1.normal().isApprox( -bdry2.normal() ),
                     "Boundary normals not antiparallel");
    BOOST_ASSERT_MSG(isApprox(bdry1.offset(), -bdry2.offset()),
                     "Boundaries planes not the same");
    bdry1.pairs_.push_back( &bdry2 );
    bdry2.pairs_.push_back( &bdry1 );
}

//----------------------------------------
//  Emitting boundaries
//----------------------------------------

Emitter::~Emitter()
{}

Phonon Emitter::emit(const Phonon::Prop& prop, Rng& gen) const
{
    BOOST_ASSERT_MSG(emitWeight() > 0., "No phonons to emit");
    Vector3d pos = drawPos(gen);
    Vector3d dir = drawDir(gen);
    bool sign = emitSign(pos, dir);
    return Phonon(sign, prop, pos, dir);
}

EmitBoundary::EmitBoundary()
: Boundary()
{}

EmitBoundary::EmitBoundary(const Vector3d& o, const Shape& s, const double T)
: Boundary(o, s), rot_( rotMatrix(normal()) ), o_(o), T_(T)
{}

EmitBoundary::~EmitBoundary()
{}

bool EmitBoundary::isInit() const
{
    return Boundary::isInit();
}

const Subdomain* EmitBoundary::emitSdom() const
{
    return sdom();
}

const Boundary* EmitBoundary::emitBdry() const
{
    return this;
}

double EmitBoundary::emitWeight() const
{
    return shape().area() * std::abs(T_);
}

Vector3d EmitBoundary::drawPos(Rng& gen) const
{
    return o_ + shape().drawPos(gen);
}

Vector3d EmitBoundary::drawDir(Rng& gen) const
{
    return rot_ * drawAniso(gen, false);
}

bool EmitBoundary::emitSign(const Vector3d&, const Vector3d&) const
{
    return T_ >= 0.;
}

template<typename S>
IsotBoundary<S>::IsotBoundary()
: EmitBoundary()
{}

template<typename S>
IsotBoundary<S>::IsotBoundary(const Vector3d& o, const S& s, const double T)
: EmitBoundary(o, s, T), shape_(s)
{}

template<typename S>
const Boundary::Shape& IsotBoundary<S>::shape() const
{
    return shape_;
}

template<typename S>
std::string IsotBoundary<S>::type() const
{
    return std::string("Isot") + shape().type();
}

template<typename S>
const Boundary* IsotBoundary<S>::scatter(Phonon& phn, Rng&) const
{
    phn.kill();
    return this;
}

template class IsotBoundary< Parallelogram >;
template class IsotBoundary< Triangle >;
template class IsotBoundary< Polygon<4> >;
template class IsotBoundary< Polygon<5> >;
template class IsotBoundary< Polygon<6> >;
template class IsotBoundary< Polygon<7> >;
template class IsotBoundary< Polygon<8> >;
template class IsotBoundary< Polygon<9> >;

template<typename S>
PeriBoundary<S>::PeriBoundary()
: EmitBoundary()
{}

template<typename S>
PeriBoundary<S>::PeriBoundary(const Vector3d& o, const S& s, const double T)
: EmitBoundary(o, s, T), shape_(s), pair_(0)
{}

template<typename S>
PeriBoundary<S>::PeriBoundary(const PeriBoundary& bdry)
: EmitBoundary(bdry), shape_(bdry.shape_), pair_(0),
rot_(Matrix3d::Zero()), transl_(Vector3d::Zero())
{}

template<typename S>
PeriBoundary<S>& PeriBoundary<S>::operator=(const PeriBoundary& bdry)
{
    EmitBoundary::operator=(bdry);
    shape_ = bdry.shape_;
    pair_ = 0;
    rot_ = Matrix3d::Zero();
    transl_ = Vector3d::Zero();
    return *this;
}

template<typename S>
bool PeriBoundary<S>::isInit() const
{
    return EmitBoundary::isInit() && pair_ != 0;
}

template<typename S>
const Boundary::Shape& PeriBoundary<S>::shape() const
{
    return shape_;
}

template<typename S>
std::string PeriBoundary<S>::type() const
{
    return std::string("Peri") + shape().type();
}

template<typename S>
const Boundary* PeriBoundary<S>::scatter(Phonon& phn, Rng&) const
{
    phn.pos(rot_ * phn.pos() + transl_);
    phn.dir(rot_ * phn.dir(), false); // CHECK THIS!!
    return pair_;
}

template<typename S>
void makePair(PeriBoundary<S>& bdry1, PeriBoundary<S>& bdry2,
              const Vector3d& transl, const Matrix3d& rot)
{
    BOOST_ASSERT_MSG(bdry1.pair_ == 0 && bdry2.pair_ == 0,
                     "Boundary already paired");
    BOOST_ASSERT_MSG(Matrix3d::Identity().isApprox(rot * rot.transpose()),
                     "Rotation matrix must be orthogonal");
    BOOST_ASSERT_MSG((rot * bdry1.normal()).isApprox( -bdry2.normal() ),
                     "Rotation matrix incorrect");
    Vector3d transform = rot * bdry1.normal() * (-bdry1.offset()) + transl;
    BOOST_ASSERT_MSG(isApprox(bdry2.normal().dot(transform), -bdry2.offset()),
                     "Translation vector incorrect");
    
    double T1 = bdry1.T_;
    double T2 = bdry2.T_;
    bdry1.T_ = T1 - T2;
    bdry2.T_ = T2 - T1;
    
    bdry1.rot_ = rot;
    bdry2.rot_ = rot.transpose();
    bdry1.transl_ = transl;
    bdry2.transl_ = -rot.transpose() * transl;
    
    bdry1.pair_ = &bdry2;
    bdry2.pair_ = &bdry1;
}

template class PeriBoundary< Parallelogram >;
template class PeriBoundary< Triangle >;
template class PeriBoundary< Polygon<4> >;
template class PeriBoundary< Polygon<5> >;
template class PeriBoundary< Polygon<6> >;
template class PeriBoundary< Polygon<7> >;
template class PeriBoundary< Polygon<8> >;
template class PeriBoundary< Polygon<9> >;

template void makePair(PeriBoundary< Parallelogram >&,
                       PeriBoundary< Parallelogram >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Triangle >&,
                       PeriBoundary< Triangle >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Polygon<4> >&,
                       PeriBoundary< Polygon<4> >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Polygon<5> >&,
                       PeriBoundary< Polygon<5> >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Polygon<6> >&,
                       PeriBoundary< Polygon<6> >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Polygon<7> >&,
                       PeriBoundary< Polygon<7> >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Polygon<8> >&,
                       PeriBoundary< Polygon<8> >&,
                       const Vector3d&, const Matrix3d&);
template void makePair(PeriBoundary< Polygon<9> >&,
                       PeriBoundary< Polygon<9> >&,
                       const Vector3d&, const Matrix3d&);


