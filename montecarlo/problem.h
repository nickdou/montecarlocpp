//
//  problem.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#ifndef __montecarlocpp__problem__
#define __montecarlocpp__problem__

#include "field.h"
#include "domain.h"
#include "subdomain.h"
#include "material.h"
#include "phonon.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/optional.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/assert.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

using Eigen::Vector3d;
using Eigen::Matrix3d;

class Clock
{
private:
    std::time_t start_;
    
public:
    Clock();

    std::string stopwatch();
    
    static std::string timestamp();
};

class Progress
{
private:
    long tot_, count_, div_, next_, esc_;
    std::vector<long> vec_;
    Clock clk_;
    
public:
    Progress();
    Progress(long tot, long div);
    
    void clock(const Clock& clk);
    
    long count() const;
    long esc() const;
    
    long incrCount();
    long incrEsc();
    
    void count(long n);
};

template<typename Derived>
struct ProbTraits
{};

template<typename Derived>
class Problem;

template<typename Derived>
std::ostream& operator<<(std::ostream& os, const Problem<Derived>& prob);

template<typename Derived>
class Problem
{
public:
    typedef typename ProbTraits<Derived>::Solution Solution;
    
private:
    const Material* mat_;
    const Domain* dom_;
    
protected:
    virtual std::string info() const;
    
public:
    Problem();
    Problem(const Material* mat, const Domain* dom);
    virtual ~Problem();
    
    const Material* mat() const;
    const Domain* dom() const;
    
    virtual Progress initProgress() const = 0;
    virtual Solution solve(Rng& gen, Progress* prog) const = 0;
    
    friend std::ostream& operator<< <>(std::ostream& os, const Problem& prob);
};

class TrajProblem;

template<>
struct ProbTraits<TrajProblem>
{
    typedef TrkPhonon::Array3Xd Solution;
};

class TrajProblem : public Problem<TrajProblem>
{
public:
    typedef Problem<TrajProblem> Base;
    typedef TrkPhonon::Array3Xd Solution;
    
private:
    static const long loopFactor_ = 100;
    
    long maxscat_, maxloop_;
    boost::optional<Phonon::Prop> prop_;
    boost::optional<Vector3d> pos_, dir_;
    
    std::string info() const;
    
public:
    TrajProblem();
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop,
                const Vector3d& pos, const Vector3d& dir,
                long maxscat = 100l, long maxloop = 0l);
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop, const Vector3d& pos,
                long maxscat = 100l, long maxloop = 0l);
    TrajProblem(const Material* mat, const Domain* dom,
                const Vector3d& pos, const Vector3d& dir,
                long maxscat = 100l, long maxloop = 0l);
    TrajProblem(const Material* mat, const Domain* dom,
                const Vector3d& pos,
                long maxscat = 100l, long maxloop = 0l);
    TrajProblem(const Material* mat, const Domain* dom,
                long maxscat = 100l, long maxloop = 0l);
    
    Progress initProgress() const;
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

template<typename T, int N>
struct CellVolF
{
    typedef Eigen::Matrix<double, N, 1> VectorNT;
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    
    VectorNT operator()(const Subdomain* sdom, const Vector3l& index) const;
};

template<typename T, int N>
struct AccumF
{
    typedef Eigen::Matrix<T, N, 1> VectorNT;
    
    virtual VectorNT operator()(const Phonon& before,
                                const Phonon& after) const = 0;
    virtual VectorNT operator()(const VectorNT& elem) const = 0;
};

template<typename Derived>
class FieldProblem : public Problem<Derived>
{
public:
    typedef Problem<Derived> Base;
    typedef typename ProbTraits<Derived>::Solution Solution;
    typedef typename ProbTraits<Derived>::FieldAccumF FieldAccumF;
    
    typedef typename Solution::Type Type;
    static const int Num = Solution::Num;
    
    typedef Eigen::Matrix<Type, Num, 1> VectorNT;
    BOOST_MPL_ASSERT((boost::is_same< Solution, Field<Type, Num> >));
    
private:
    static const long loopFactor_ = 100;
    long nemit_, maxscat_, maxloop_;
    
protected:
    std::string info() const;
    
public:
    FieldProblem();
    FieldProblem(const Material* mat, const Domain* dom,
                 long nemit, long maxscat, long maxloop);
    virtual ~FieldProblem();
    
    Progress initProgress() const;
    Solution initSolution() const;
    
    Solution solve(Rng& gen, Progress* prog = 0) const;
    
private:
    virtual VectorNT postMult() const = 0;
    virtual FieldAccumF accumFun() const = 0;
};

class TempProblem;

struct TempAccumF : public AccumF<double, 1>
{
    VectorNT operator()(const Phonon& before, const Phonon& after) const;
    VectorNT operator()(const VectorNT& elem) const;
};

template<>
struct ProbTraits<TempProblem>
{
    typedef Field<double, 1> Solution;
    typedef TempAccumF FieldAccumF;
};

class TempProblem : public FieldProblem<TempProblem>
{
public:
    typedef FieldProblem<TempProblem> Base;
    
private:
    std::string info() const;
    
public:
    TempProblem();
    TempProblem(const Material* mat, const Domain* dom,
                long nemit, long maxscat, long maxloop = 0);
    
    Base::VectorNT postMult() const;
    Base::FieldAccumF accumFun() const;
};

class FluxProblem;

struct FluxAccumF : public AccumF<double, 1>
{
private:
    Vector3d dir_;
    
public:
    FluxAccumF(const Vector3d& dir);
    
    VectorNT operator()(const Phonon& before, const Phonon& after) const;
    VectorNT operator()(const VectorNT& elem) const;
};

template<>
struct ProbTraits<FluxProblem>
{
    typedef Field<double, 1> Solution;
    typedef FluxAccumF FieldAccumF;
};

class FluxProblem : public FieldProblem<FluxProblem>
{
public:
    typedef FieldProblem<FluxProblem> Base;
    
private:
    Vector3d dir_;
    
    std::string info() const;
    
public:
    FluxProblem();
    FluxProblem(const Material* mat, const Domain* dom,
                const Vector3d& dir,
                long nemit, long maxscat, long maxloop = 0);
    
    Base::VectorNT postMult() const;
    Base::FieldAccumF accumFun() const;
};

class MultiProblem;

struct MultiAccumF : public AccumF<double, 4>
{
private:
    Matrix3d inv_;
    
public:
    MultiAccumF(const Matrix3d& rot);
    
    VectorNT operator()(const Phonon& before, const Phonon& after) const;
    VectorNT operator()(const VectorNT& elem) const;
};

template<>
struct ProbTraits<MultiProblem>
{
    typedef Field<double, 4> Solution;
    typedef MultiAccumF FieldAccumF;
};

class MultiProblem : public FieldProblem<MultiProblem>
{
public:
    typedef FieldProblem<MultiProblem> Base;
    
private:
    Matrix3d rot_;
    
    std::string info() const;
    
public:
    MultiProblem();
    MultiProblem(const Material* mat, const Domain* dom,
                 const Matrix3d& rot,
                 long nemit, long maxscat, long maxloop = 0);
    MultiProblem(const Material* mat, const Domain* dom,
                 long nemit, long maxscat, long maxloop = 0);
    
    Base::VectorNT postMult() const;
    Base::FieldAccumF accumFun() const;
};

template<typename F>
struct CumF : public AccumF<double, Eigen::Dynamic>
{
private:
    long size_, step_;
    F fun_;
    
public:
    CumF(long size, long step, const F& fun);
    
    VectorNT operator()(const Phonon& before, const Phonon& after) const;
    VectorNT operator()(const VectorNT& elem) const;
};

class CumTempProblem;

struct CumTempAccumF : public CumF<TempAccumF>
{
    CumTempAccumF(long step, long size, const TempAccumF& fun);
};

template<>
struct ProbTraits<CumTempProblem>
{
    typedef Field<double, Eigen::Dynamic> Solution;
    typedef CumTempAccumF FieldAccumF;
};

class CumTempProblem : public FieldProblem<CumTempProblem>
{
public:
    typedef FieldProblem<CumTempProblem> Base;
    
private:
    long size_, step_;
    
    std::string info() const;
    
public:
    CumTempProblem();
    CumTempProblem(const Material* mat, const Domain* dom,
                   long nemit, long size, long maxscat, long maxloop = 0);
    
    Base::VectorNT postMult() const;
    Base::FieldAccumF accumFun() const;
};

class CumFluxProblem;

struct CumFluxAccumF : public CumF<FluxAccumF>
{
    CumFluxAccumF(long step, long size, const FluxAccumF& fun);
};

template<>
struct ProbTraits<CumFluxProblem>
{
    typedef Field<double, Eigen::Dynamic> Solution;
    typedef CumFluxAccumF FieldAccumF;
};

class CumFluxProblem : public FieldProblem<CumFluxProblem>
{
public:
    typedef FieldProblem<CumFluxProblem> Base;
    
private:
    Vector3d dir_;
    long size_, step_;
    
    std::string info() const;
    
public:
    CumFluxProblem();
    CumFluxProblem(const Material* mat, const Domain* dom,
                   const Vector3d& dir,
                   long nemit, long size, long maxscat, long maxloop = 0);
    
    Base::VectorNT postMult() const;
    Base::FieldAccumF accumFun() const;
};

#endif
