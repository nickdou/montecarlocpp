//
//  boundary.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/29/15.
//
//

#ifndef __montecarlocpp__boundary__
#define __montecarlocpp__boundary__

#include "phonon.h"
#include "tools.h"
#include "random.h"
#include "constants.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/type_traits.hpp>
#include <cmath>

class Subdomain;

class Boundary {
public:
    class Shape {
    protected:
        typedef Eigen::Vector3d Vector3d;
    public:
        virtual ~Shape() {}
        virtual bool isInit() const = 0;
        virtual Vector3d normal() const = 0;
        virtual double area() const = 0;
        virtual Vector3d drawPos(Rng& gen) const = 0;
    };
protected:
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
    typedef Eigen::Hyperplane<double, 3> Plane3d;
private:
    const Subdomain* sdom_;
    Plane3d plane_;
public:
    Boundary() : sdom_(0) {}
    Boundary(const Vector3d& o, const Vector3d& n)
    : sdom_(0), plane_(n.normalized(), o) {}
    Boundary(const Vector3d& o, const Shape& s)
    : sdom_(0), plane_(s.normal(), o) {
        BOOST_ASSERT_MSG(s.isInit(), "Shape not initialized");
    }
    Boundary(const Boundary& bdry) : sdom_(0), plane_(bdry.plane_) {}
    Boundary& operator=(const Boundary& bdry) {
        sdom_ = 0;
        plane_ = bdry.plane_;
        return *this;
    }
    virtual ~Boundary() {}
    virtual bool isInit() const { return sdom_ != 0; }
    const Subdomain* sdom() const { return sdom_; }
    void sdom(const Subdomain* s) { sdom_ = s; }
    Vector3d normal() const { return plane_.normal(); }
    double offset() const { return plane_.offset(); }
    Vector3d projection(const Vector3d& pos) const {
        return plane_.projection(pos);
    }
    double distance(const Vector3d& pos) const {
        return plane_.signedDistance(pos);
    }
    double distance(const Phonon& phn) const {
        return phn.line().intersection(plane_);
    }
    virtual const Boundary* scatter(Phonon& phn, Rng& gen) const = 0;
};

class Parallelogram : public Boundary::Shape {
private:
    Vector3d i_, j_;
public:
    Parallelogram() : i_(Vector3d::Zero()), j_(Vector3d::Zero()) {}
    Parallelogram(const Vector3d& i, const Vector3d& j) : i_(i), j_(j) {}
    bool isInit() const {
        return i_ != Vector3d::Zero() && j_ != Vector3d::Zero();
    }
    Vector3d normal() const {
        return i_.cross(j_).normalized();
    }
    double area() const {
        return i_.cross(j_).norm();
    }
    Vector3d drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        double r1 = dist(gen), r2 = dist(gen);
        return r1*i_ + r2*j_;
    }
};

class Triangle : public Boundary::Shape {
private:
    Vector3d i_, j_;
public:
    Triangle() : i_(Vector3d::Zero()), j_(Vector3d::Zero()) {}
    Triangle(const Vector3d& i, const Vector3d& j) : i_(i), j_(j) {}
    bool isInit() const {
        return i_ != Vector3d::Zero() && j_ != Vector3d::Zero();
    }
    Vector3d normal() const {
        return i_.cross(j_).normalized();
    }
    double area() const {
        return i_.cross(j_).norm() / 2.;
    }
    Vector3d drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        double r1 = dist(gen), r2 = dist(gen);
        return r1 + r2 < 1. ? r1*i_ + r2*j_ : (1. - r1)*i_ + (1. - r2)*j_;
    }
};

class SpecBoundary : public Boundary {
private:
    Matrix3d refl_;
public:
    SpecBoundary() : Boundary() {}
    SpecBoundary(const Vector3d& o, const Vector3d& n)
    : Boundary(o, n), refl_(reflMatrix(normal())) {}
    SpecBoundary(const Vector3d& o, const Shape& s, const double T)
    : Boundary(o, s), refl_(reflMatrix(normal())) {
        BOOST_ASSERT_MSG(T == 0., "Cannot specify temperature");
    }
    const Boundary* scatter(Phonon& phn, Rng&) const {
        phn.dir<false>( refl_.selfadjointView<Eigen::Upper>() * phn.dir() );
        return this;
    }
};

class DiffBoundary : public Boundary {
private:
    Matrix3d rot_;
public:
    DiffBoundary() : Boundary() {}
    DiffBoundary(const Vector3d& o, const Vector3d& n)
    : Boundary(o, n), rot_(rotMatrix(normal())) {}
    DiffBoundary(const Vector3d& o, const Shape& s, const double T)
    : Boundary(o, s), rot_(rotMatrix(normal())) {
        BOOST_ASSERT_MSG(T == 0., "Cannot specify temperature");
    }
    const Boundary* scatter(Phonon& phn, Rng& gen) const {
        phn.dir<true>( rot_ * drawAniso<false>(gen) );
        return this;
    }
};

class InterBoundary : public Boundary {
private:
    const InterBoundary* pair_;
public:
    InterBoundary() : Boundary(), pair_(0) {}
    InterBoundary(const Vector3d& o, const Vector3d& n)
    : Boundary(o, n), pair_(0) {}
    InterBoundary(const Vector3d& o, const Shape& s, const double T)
    : Boundary(o, s), pair_(0) {
        BOOST_ASSERT_MSG(T == 0., "Cannot specify temperature");
    }
    InterBoundary(const InterBoundary& bdry) : Boundary(bdry), pair_(0) {}
    InterBoundary& operator=(const InterBoundary& bdry) {
        Boundary::operator=(bdry);
        pair_ = 0;
        return *this;
    }
    bool isInit() const {
        return Boundary::isInit() && pair_ != 0;
    }
    const Boundary* scatter(Phonon&, Rng&) const {
        return pair_;
    }
    friend void makePair(InterBoundary& bdry1, InterBoundary& bdry2) {
        BOOST_ASSERT_MSG(bdry1.pair_ == 0 && bdry2.pair_ == 0,
                         "Boundary already paired");
        BOOST_ASSERT_MSG(bdry1.normal().isApprox(-bdry2.normal()),
                         "Boundary normals not antiparallel");
        BOOST_ASSERT_MSG(isApprox(bdry1.offset(), -bdry2.offset()),
                         "Boundaries planes not the same");
        bdry1.pair_ = &bdry2;
        bdry2.pair_ = &bdry1;
    }
};

class Emitter {
public:
    typedef boost::is_base_of<Emitter, boost::mpl::_> IsBaseL;
    virtual ~Emitter() {}
    virtual const Subdomain* emitSdom() const = 0;
    virtual const Boundary* emitBdry() const = 0;
    virtual double emitWeight() const = 0;
    Phonon emit(const Phonon::Prop& prop, Rng& gen) const {
        BOOST_ASSERT_MSG(emitWeight() >= Dbl::min(), "No phonons to emit");
        Eigen::Vector3d pos = drawPos(gen);
        Eigen::Vector3d dir = drawDir(gen);
        bool sign = emitSign(pos, dir);
        return Phonon(prop, pos, dir, sign);
    };
protected:
    virtual Eigen::Vector3d drawPos(Rng& gen) const = 0;
    virtual Eigen::Vector3d drawDir(Rng& gen) const = 0;
    virtual bool emitSign(const Eigen::Vector3d& pos,
                          const Eigen::Vector3d& dir) const = 0;
};

template<typename S>
class EmitBoundary : public Boundary, public Emitter {
private:
    typedef Eigen::Vector3d Vector3d;
protected:
    Vector3d o_;
    S shape_;
    double T_;
private:
    Matrix3d rot_;
public:
    EmitBoundary() : Boundary() {}
    EmitBoundary(const Vector3d& o, const S& s, const double T)
    : Boundary(o, s), o_(o), shape_(s), T_(T), rot_(rotMatrix(normal())) {}
    virtual ~EmitBoundary() {}
    virtual bool isInit() const { return Boundary::isInit(); }
    const Subdomain* emitSdom() const { return sdom(); }
    const Boundary* emitBdry() const { return this; }
    double emitWeight() const {
        return shape_.area() * std::abs(T_);
    }
private:
    Vector3d drawPos(Rng& gen) const {
        return o_ + shape_.drawPos(gen);
    }
    Vector3d drawDir(Rng& gen) const {
        return rot_ * drawAniso<false>(gen);
    }
    bool emitSign(const Vector3d&, const Vector3d&) const {
        return T_ > 0.;
    }
};

template<typename S>
class IsotBoundary : public EmitBoundary<S> {
private:
    typedef Eigen::Vector3d Vector3d;
public:
    IsotBoundary() : EmitBoundary<S>() {}
    IsotBoundary(const Vector3d& o, const S& s, const double T)
    : EmitBoundary<S>(o, s, T) {}
    const Boundary* scatter(Phonon& phn, Rng&) const {
        phn.kill();
        return this;
    }
};

template<typename S>
class PeriBoundary : public EmitBoundary<S> {
private:
    typedef Eigen::Vector3d Vector3d;
    PeriBoundary* pair_;
    Vector3d transl_;
public:
    PeriBoundary() : EmitBoundary<S>(), pair_(0) {}
    PeriBoundary(const Vector3d& o, const S& s, const double T)
    : EmitBoundary<S>(o, s, T), pair_(0) {}
    PeriBoundary(const PeriBoundary& bdry)
    : EmitBoundary<S>(bdry), pair_(0) {}
    PeriBoundary& operator=(const PeriBoundary& peri) {
        EmitBoundary<S>::operator=(peri);
        pair_ = 0;
        transl_ = Vector3d::Zero();
        return *this;
    }
    bool isInit() const {
        return EmitBoundary<S>::isInit() && pair_ != 0;
    }
    const Boundary* scatter(Phonon& phn, Rng&) const {
        phn.pos(phn.pos() + transl_);
        return pair_;
    }
    friend void makePair(PeriBoundary& bdry1, PeriBoundary& bdry2,
                         const Vector3d& t) {
        BOOST_ASSERT_MSG(bdry1.pair_ == 0 && bdry2.pair_ == 0,
                         "Boundary already paired");
        BOOST_ASSERT_MSG(bdry1.normal().isApprox(-bdry2.normal()),
                         "Boundary normals not antiparallel");
        double distance = bdry1.offset() + bdry2.offset();
        BOOST_ASSERT_MSG(distance > 0.,
                         "Boundary normals not inward facing");
        BOOST_ASSERT_MSG(isApprox(t.dot(bdry1.normal()), distance),
                         "Translation vector must span boundary planes");
        double T1 = bdry1.T_;
        double T2 = bdry2.T_;
        bdry1.T_ = T1 - T2;
        bdry2.T_ = T2 - T1;
        bdry1.pair_ = &bdry2;
        bdry2.pair_ = &bdry1;
        bdry1.transl_ = t;
        bdry2.transl_ = -t;
    }
};


#endif
