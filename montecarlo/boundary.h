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
#include "random.h"
#include "constants.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <boost/static_assert.hpp>
#include <vector>

using Eigen::Vector3d;
using Eigen::Matrix3d;

class Subdomain;

//----------------------------------------
//  Boundary
//----------------------------------------

class Boundary
{
public:
    class Shape;
    typedef std::vector<const Boundary*> Pointers;
    
private:
    const Subdomain* sdom_;
    Eigen::Hyperplane<double, 3> plane_;
    
public:
    Boundary();
    Boundary(const Vector3d& o, const Vector3d& n);
    Boundary(const Vector3d& o, const Shape& s);
    Boundary(const Boundary& bdry);
    Boundary& operator=(const Boundary& bdry);
    virtual ~Boundary();
    
    virtual bool isInit() const;
    const Subdomain* sdom() const;
    void sdom(const Subdomain* s);
    
    Vector3d normal() const;
    double offset() const;
    Vector3d projection(const Vector3d& pos) const;
    double distance(const Vector3d& pos) const;
    double distance(const Phonon& phn) const;
    
    virtual const Boundary* scatter(Phonon& phn, Rng& gen) const = 0;
};

//----------------------------------------
//  Shapes
//----------------------------------------

class Boundary::Shape
{
public:
    virtual ~Shape();
    virtual bool isInit() const = 0;
    virtual Vector3d normal() const = 0;
    virtual double area() const = 0;
    virtual Vector3d drawPos(Rng& gen) const = 0;
};

class Parallelogram : public Boundary::Shape
{
private:
    Vector3d i_, j_;
    
public:
    Parallelogram();
    Parallelogram(const Vector3d& i, const Vector3d& j);
    
    bool isInit() const;
    Vector3d normal() const;
    double area() const;
    Vector3d drawPos(Rng& gen) const;
};

class Triangle : public Boundary::Shape
{
private:
    Vector3d i_, j_;
    
public:
    Triangle();
    Triangle(const Vector3d& i, const Vector3d& j);
    
    bool isInit() const;
    Vector3d normal() const;
    double area() const;
    Vector3d drawPos(Rng& gen) const;
};

template<int N>
class Polygon : public Boundary::Shape
{
private:
    BOOST_STATIC_ASSERT_MSG(N > 3, "Polygon must have more than 3 sides");
    
    Eigen::Matrix<double, 3, N - 1> verts_;
    Eigen::Matrix<double, 1, N - 2> areas_;
    DiscreteDist areaDist_;
    
public:
    Polygon();
    Polygon(const Eigen::Matrix<double, 3, N - 1>& v);
    
    bool isInit() const;
    Vector3d normal() const;
    double area() const;
    Vector3d drawPos(Rng& gen) const;
};

//----------------------------------------
//  Non-emitting boundaries
//----------------------------------------

class SpecBoundary : public Boundary
{
private:
    Matrix3d refl_;
    
public:
    SpecBoundary();
    SpecBoundary(const Vector3d& o, const Vector3d& n);
    SpecBoundary(const Vector3d& o, const Shape& s, const double T);
    
    const Boundary* scatter(Phonon& phn, Rng&) const;
};

class DiffBoundary : public Boundary
{
private:
    Matrix3d rot_;
    
public:
    DiffBoundary();
    DiffBoundary(const Vector3d& o, const Vector3d& n);
    DiffBoundary(const Vector3d& o, const Shape& s, const double T);
    
    const Boundary* scatter(Phonon& phn, Rng& gen) const;
};

class InterBoundary : public Boundary
{
private:
    Pointers pairs_;
    
public:
    InterBoundary();
    InterBoundary(const Vector3d& o, const Vector3d& n);
    InterBoundary(const Vector3d& o, const Shape& s, const double T);
    InterBoundary(const InterBoundary& bdry);
    InterBoundary& operator=(const InterBoundary& bdry);
    
    bool isInit() const;
    const Boundary* scatter(Phonon&, Rng&) const;
    
    friend void makePair(InterBoundary& bdry1, InterBoundary& bdry2);
};

//----------------------------------------
//  Emitting boundaries
//----------------------------------------

class Emitter
{
public:
    typedef std::vector< const Emitter* > Pointers;
    
    virtual ~Emitter();
    virtual const Subdomain* emitSdom() const = 0;
    virtual const Boundary* emitBdry() const = 0;
    virtual double emitWeight() const = 0;
    Phonon emit(const Phonon::Prop& prop, Rng& gen) const;
    
private:
    virtual Vector3d drawPos(Rng& gen) const = 0;
    virtual Vector3d drawDir(Rng& gen) const = 0;
    virtual bool emitSign(const Vector3d& pos, const Vector3d& dir) const = 0;
};

class EmitBoundary : public Boundary, public Emitter
{
private:
    Matrix3d rot_;
    Vector3d o_;
    
protected:
    double T_;

public:
    EmitBoundary();
    EmitBoundary(const Vector3d& o, const Shape& s, const double T);
    virtual ~EmitBoundary();
    
    virtual bool isInit() const;
    
    const Subdomain* emitSdom() const;
    const Boundary* emitBdry() const;
    double emitWeight() const;
    
private:
    Vector3d drawPos(Rng& gen) const;
    Vector3d drawDir(Rng& gen) const;
    bool emitSign(const Vector3d&, const Vector3d&) const;
    
    virtual const Shape& shape() const = 0;
};

template<typename S>
class IsotBoundary : public EmitBoundary
{
private:
    S shape_;
    
    const Shape& shape() const;
    
public:
    IsotBoundary();
    IsotBoundary(const Vector3d& o, const S& s, const double T);
    
    const Boundary* scatter(Phonon& phn, Rng&) const;
};

template<typename S>
class PeriBoundary;

template<typename S>
void makePair(PeriBoundary<S>& bdry1, PeriBoundary<S>& bdry2,
              const Vector3d& t, const Matrix3d& r = Matrix3d::Identity());

template<typename S>
class PeriBoundary : public EmitBoundary
{
private:
    S shape_;
    
    const PeriBoundary* pair_;
    Matrix3d rot_;
    Vector3d transl_;
    
    const Shape& shape() const;
    
public:
    PeriBoundary();
    PeriBoundary(const Vector3d& o, const S& s, const double T);
    PeriBoundary(const PeriBoundary& bdry);
    PeriBoundary& operator=(const PeriBoundary& peri);
    
    bool isInit() const;
    
    const Boundary* scatter(Phonon& phn, Rng&) const;
    
    friend void makePair<>(PeriBoundary& bdry1, PeriBoundary& bdry2,
                           const Vector3d& t, const Matrix3d& r);
};


#endif
