//
//  problem.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#ifndef __montecarlocpp__problem__
#define __montecarlocpp__problem__

#include "phonon.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/optional.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::ArrayXXd;

typedef Eigen::Matrix<long, 3, 1> Vector3l;
typedef Eigen::Matrix<long, Eigen::Dynamic, 1> VectorXl;

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
};

class Material;
class Domain;

class Problem
{
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
    virtual ArrayXXd solve(Rng& gen, Progress* prog) const = 0;
    
    friend std::ostream& operator<<(std::ostream& os, const Problem& prob);
};

class TrajProblem : public Problem
{
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
                long maxscat = 100, long maxloop = 0);
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop, const Vector3d& pos,
                long maxscat = 100, long maxloop = 0);
    TrajProblem(const Material* mat, const Domain* dom,
                const Vector3d& pos, const Vector3d& dir,
                long maxscat = 100, long maxloop = 0);
    TrajProblem(const Material* mat, const Domain* dom,
                const Vector3d& pos,
                long maxscat = 100, long maxloop = 0);
    TrajProblem(const Material* mat, const Domain* dom,
                long maxscat = 100, long maxloop = 0);
    
    Progress initProgress() const;
    ArrayXXd solve(Rng& gen, Progress* prog) const;
};

class FieldProblem : public Problem
{
private:
    static const long loopFactor_ = 100;
    long nemit_, maxscat_, maxloop_;
    double power_;
    VectorXl emitPdf_;
    
protected:
    std::string info() const;
    
public:
    FieldProblem();
    FieldProblem(const Material* mat, const Domain* dom,
                 long nemit, long maxscat, long maxloop);
    virtual ~FieldProblem();
    
    Progress initProgress() const;
    ArrayXXd initSolution() const;
    
    ArrayXXd solve(Rng& gen, Progress* prog) const;
    
private:
    virtual long rows() const = 0;
    virtual VectorXd accumAmt(const Phonon& before, const Phonon& after) const = 0;
    virtual ArrayXXd postProc(const ArrayXXd& data) const = 0;
};

class Subdomain;

class CellVolF
{
public:
    VectorXd operator()(const Subdomain* sdom, const Vector3l& index) const;
};

class TempProblem : public FieldProblem
{
private:
    std::string info() const;
    
public:
    TempProblem();
    TempProblem(const Material* mat, const Domain* dom,
                long nemit, long maxscat, long maxloop = 0);
    
private:
    long rows() const;
    VectorXd accumAmt(const Phonon& before, const Phonon& after) const;
    ArrayXXd postProc(const ArrayXXd& data) const;
};

class FluxProblem : public FieldProblem
{
private:
    std::string info() const;
    
public:
    FluxProblem();
    FluxProblem(const Material* mat, const Domain* dom,
                long nemit, long maxscat, long maxloop = 0);
    
private:
    long rows() const;
    VectorXd accumAmt(const Phonon& before, const Phonon& after) const;
    ArrayXXd postProc(const ArrayXXd& data) const;
};

class MultiProblem : public FieldProblem
{
private:
    std::string info() const;
    
public:
    MultiProblem();
    MultiProblem(const Material* mat, const Domain* dom,
                 long nemit, long maxscat, long maxloop = 0);
    
private:
    long rows() const;
    VectorXd accumAmt(const Phonon& before, const Phonon& after) const;
    ArrayXXd postProc(const ArrayXXd& data) const;
};

class CumTempProblem : public FieldProblem
{
private:
    long size_, step_;
    
    std::string info() const;
    
public:
    CumTempProblem();
    CumTempProblem(const Material* mat, const Domain* dom,
                   long nemit, long size, long maxscat, long maxloop = 0);
    
private:
    long rows() const;
    VectorXd accumAmt(const Phonon& before, const Phonon& after) const;
    ArrayXXd postProc(const ArrayXXd& data) const;
};

class CumFluxProblem : public FieldProblem
{
private:
    long size_, step_;
    
    std::string info() const;
    
public:
    CumFluxProblem();
    CumFluxProblem(const Material* mat, const Domain* dom,
                   long nemit, long size, long maxscat, long maxloop = 0);
    
private:
    long rows() const;
    VectorXd accumAmt(const Phonon& before, const Phonon& after) const;
    ArrayXXd postProc(const ArrayXXd& data) const;
};

#endif
