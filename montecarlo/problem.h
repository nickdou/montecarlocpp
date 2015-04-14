//
//  problem.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#ifndef __montecarlocpp__problem__
#define __montecarlocpp__problem__

#include "domain.h"
#include "grid.h"
#include "data.h"
#include "material.h"
#include "phonon.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/optional.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/assert.hpp>
#include <boost/mpl/assert.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cmath>

class Clock {
private:
    std::time_t start_;
public:
    Clock() : start_() {
        std::time(&start_);
    }
    static std::string timestamp() {
        static const int size = 80;
        std::time_t now;
        char buffer[size];
        std::time(&now);
        std::strftime(buffer, size, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buffer);
    }
    std::string stopwatch() {
        std::time_t now;
        std::time(&now);
        long diff = static_cast<long>(std::difftime(now, start_));
        std::ostringstream ss;
        ss << std::setfill('0');
        ss << std::setw(2) << diff / 3600 << ':';
        ss << std::setw(2) << diff / 60 % 60 << ':';
        ss << std::setw(2) << diff % 60;
        return ss.str();
    }
};

class Progress {
private:
    long tot_, count_, div_, next_;
    std::vector<long> vec_;
    Clock clk_;
    std::ostream* os_;
    void initVec() {
        BOOST_ASSERT_MSG(tot_ >= div_, "Too many divisions");
        for (long i = 0; i < div_; ++i) {
            vec_.push_back(tot_ * (i+1) / div_);
        }
    }
public:
    Progress()
    : tot_(0), count_(0), div_(0), next_(0), vec_(), clk_(), os_(&std::cout) {}
    Progress(long tot, long div)
    : tot_(tot), count_(0), div_(div), next_(0), vec_(), clk_(), os_(&std::cout)
    {
        initVec();
    }
    void div(long d) {
        BOOST_ASSERT_MSG(count_ == 0, "Cannot change divisions");
        div_ = d;
        initVec();
    }
    void clock(const Clock& clk) { clk_ = clk; }
    void ostream(std::ostream* os) { os_ = os; }
    long increment() {
        if (tot_ == 0 || next_ >= div_) return count_;
        count_++;
        if (count_ == vec_.at(next_)) {
            next_++;
            *os_ << clk_.stopwatch() << ' ';
            *os_ << '[' << std::string(next_, '|');
            *os_ << std::string(div_ - next_, '-') << ']' << std::endl;
        }
        return count_;
    }
};

template<typename Derived>
struct ProblemTraits {};

template<typename Derived>
class Problem {
public:
    typedef typename ProblemTraits<Derived>::Solution Solution;
protected:
    const Material* mat_;
    const Domain* dom_;
    virtual std::string info() const {
        std::ostringstream ss;
        ss << "  mat:     " << mat_ << std::endl;
        ss << "  dom:     " << dom_;
        return ss.str();
    }
public:
    Problem() : mat_(0), dom_(0) {}
    Problem(const Material* mat, const Domain* dom) : mat_(mat), dom_(dom) {
        BOOST_ASSERT_MSG(dom_->isInit(), "Domain setup not complete");
    };
    virtual ~Problem() {}
    virtual Progress initProgress() const = 0;
    virtual Solution solve(Rng& gen, Progress* prog = 0) const = 0;
    friend std::ostream& operator<<(std::ostream& os, const Problem& prob) {
        return os << prob.info();
    }
};

class TrajProblem;

template<>
struct ProblemTraits<TrajProblem> {
    typedef TrkPhonon::Trajectory Solution;
};

class TrajProblem : public Problem<TrajProblem> {
public:
    typedef Problem<TrajProblem> Base;
private:
    static const long loopFactor_ = 100;
    long maxscat_, maxloop_;
    boost::optional<Phonon::Prop> prop_;
    boost::optional<Eigen::Vector3d> pos_, dir_;
    std::string info() const {
        std::ostringstream ss;
        ss << "TrajProblem " << this << std::endl;
        ss << Base::info() << std::endl;
        if (prop_) ss << "  prop:    " << prop_->w() << ", " <<
                                          prop_->p() << std::endl;
        if (pos_)  ss << "  pos:     " << pos_->transpose() << std::endl;
        if (dir_)  ss << "  dir:     " << dir_->transpose() << std::endl;
        ss << "  maxscat: " << maxscat_ << std::endl;
        ss << "  maxloop: " << maxloop_;
        return ss.str();
    }
public:
    TrajProblem() : Base() {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop,
                const Eigen::Vector3d& pos, const Eigen::Vector3d& dir,
                long maxscat = 100, long maxloop = 0)
    : Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop),
    prop_(prop), pos_(pos), dir_(dir)
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop, const Eigen::Vector3d& pos,
                long maxscat = 100, long maxloop = 0)
    : Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop),
    prop_(prop), pos_(pos), dir_()
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Eigen::Vector3d& pos, const Eigen::Vector3d& dir,
                long maxscat = 100, long maxloop = 0)
    : Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop),
    prop_(), pos_(pos), dir_(dir)
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Eigen::Vector3d& pos,
                long maxscat = 100, long maxloop = 0)
    : Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop),
    prop_(), pos_(pos), dir_()
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                long maxscat = 100, long maxloop = 0)
    : Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop),
    prop_(), pos_(), dir_()
    {}
    Progress initProgress() const {
        return Progress(maxscat_, std::min(10l, maxscat_));
    }
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

template<typename Derived>
class FieldProblem : public Problem<Derived> {
public:
    typedef Problem<Derived> Base;
    typedef typename ProblemTraits<Derived>::Factor Factor;
    typedef typename ProblemTraits<Derived>::AccumF AccumF;
    typedef typename Base::Solution::Type Type;
    BOOST_MPL_ASSERT((boost::is_same< typename Base::Solution, Field<Type> >));
protected:
    static const long loopFactor_ = 100;
    long nemit_, maxscat_, maxloop_;
    virtual std::string info() const {
        std::ostringstream ss;
        ss << Base::info() << std::endl;
        ss << "  nemit:   " << nemit_ << std::endl;
        ss << "  maxscat: " << maxscat_ << std::endl;
        ss << "  maxloop: " << maxloop_;
        return ss.str();
    }
public:
    FieldProblem() {}
    FieldProblem(const Material* mat, const Domain* dom,
                 long nemit, long maxscat, long maxloop)
    : Base(mat, dom), nemit_(nemit), maxscat_(maxscat), maxloop_(maxloop)
    {}
    ~FieldProblem() {}
    Progress initProgress() const {
        return Progress(nemit_, std::min(20l, nemit_));
    }
    Field<Type> initField() const {
        Field<Type> field;
        AddRegionF add(&field, initElem());
        const Domain::SdomPtrs& sdomPtrs = Base::dom_->sdomPtrs();
        std::for_each(sdomPtrs.begin(), sdomPtrs.end(), add);
        return field;
    }
    virtual Type initElem() const = 0;
private:
    struct AddRegionF {
        Field<Type>* field;
        Type value;
        AddRegionF(Field<Type>* fld, const Type& val) : field(fld), value(val)
        {}
        void operator()(const Subdomain* sdom) {
            typedef typename Field<Type>::value_type Value;
            field->insert( Value(sdom, sdom->initData<Type>(value)) );
        }
    };
protected:
    Field<Type> solveField(const AccumF& fun, const Factor& fac,
                           Rng& gen, Progress* prog) const;
};

class TempProblem;
struct TempAccumF;

template<>
struct ProblemTraits<TempProblem> {
    typedef Field<double> Solution;
    typedef double Factor;
    typedef TempAccumF AccumF;
};

class TempProblem : public FieldProblem<TempProblem> {
public:
    typedef FieldProblem<TempProblem> Base;
private:
    std::string info() const {
        std::ostringstream ss;
        ss << "TempProblem " << this << std::endl;
        ss << Base::info();
        return ss.str();
    }
public:
    TempProblem() : Base() {}
    TempProblem(const Material* mat, const Domain* dom,
                long nemit, long maxscat, long maxloop = 0)
    : Base(mat, dom, nemit, maxscat, maxloop)
    {}
    Type initElem() const { return 0.; }
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

class FluxProblem;
struct FluxAccumF;

template<>
struct ProblemTraits<FluxProblem> {
    typedef Field<double> Solution;
    typedef double Factor;
    typedef FluxAccumF AccumF;
};

class FluxProblem : public FieldProblem<FluxProblem> {
public:
    typedef FieldProblem<FluxProblem> Base;
private:
    Eigen::Vector3d dir_;
    std::string info() const {
        std::ostringstream ss;
        ss << "FluxProblem " << this << std::endl;
        ss << Base::info() << std::endl;
        ss << "  dir:     " << dir_.transpose();
        return ss.str();
    }
public:
    FluxProblem() : Base() {}
    FluxProblem(const Material* mat, const Domain* dom,
                const Eigen::Vector3d& dir,
                long nemit, long maxscat, long maxloop = 0)
    : Base(mat, dom, nemit, maxscat, maxloop), dir_(dir)
    {}
    Type initElem() const { return 0.; }
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

class MultiProblem;
struct MultiAccumF;

template<>
struct ProblemTraits<MultiProblem> {
    typedef Field<Eigen::Array4d> Solution;
    typedef Eigen::Array4d Factor;
    typedef MultiAccumF AccumF;
};

class MultiProblem : public FieldProblem<MultiProblem> {
public:
    typedef FieldProblem<MultiProblem> Base;
private:
    Eigen::Matrix3d rot_, inv_;
    std::string info() const {
        std::ostringstream ss;
        ss << "FluxProblem " << this << std::endl;
        ss << Base::info() << std::endl;
        ss << "  rot:     " << rot_.row(0) << ' ' <<
                               rot_.row(1) << ' ' <<
                               rot_.row(2);
        return ss.str();
    }
public:
    MultiProblem() : Base() {}
    MultiProblem(const Material* mat, const Domain* dom,
                 const Eigen::Matrix3d& rot,
                 long nemit, long maxscat, long maxloop = 0)
    : Base(mat, dom, nemit, maxscat, maxloop), rot_(rot), inv_(rot.transpose())
    {
        BOOST_ASSERT_MSG(Eigen::Matrix3d::Identity().isApprox(inv_*rot),
                         "Direction matrix must be unitary");
    }
    Type initElem() const { return Eigen::Array4d::Zero(); }
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

class CumTempProblem;
struct CumTempAccumF;

template<>
struct ProblemTraits<CumTempProblem> {
    typedef Field<Eigen::ArrayXd> Solution;
    typedef double Factor;
    typedef CumTempAccumF AccumF;
};

class CumTempProblem : public FieldProblem<CumTempProblem> {
public:
    typedef FieldProblem<CumTempProblem> Base;
private:
    long size_, step_;
    std::string info() const {
        std::ostringstream ss;
        ss << "CumTempProblem " << this << std::endl;
        ss << Base::info() << std::endl;
        ss << "  step:    " << step_;
        return ss.str();
    }
public:
    CumTempProblem() : Base() {}
    CumTempProblem(const Material* mat, const Domain* dom,
                   long nemit, long size, long maxscat, long maxloop = 0)
    : Base(mat, dom, nemit, maxscat, maxloop),
    size_(size), step_(maxscat%size == 0 ? maxscat/size : maxscat/size + 1)
    {}
    Type initElem() const { return Eigen::ArrayXd::Zero(size_); }
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

class CumFluxProblem;
struct CumFluxAccumF;

template<>
struct ProblemTraits<CumFluxProblem> {
    typedef Field<Eigen::ArrayXd> Solution;
    typedef double Factor;
    typedef CumFluxAccumF AccumF;
};

class CumFluxProblem : public FieldProblem<CumFluxProblem> {
public:
    typedef FieldProblem<CumFluxProblem> Base;
private:
    Eigen::Vector3d dir_;
    long size_, step_;
    std::string info() const {
        std::ostringstream ss;
        ss << "CumTempProblem " << this << std::endl;
        ss << Base::info() << std::endl;
        ss << "  step:    " << step_ << std::endl;
        ss << "  dir:     " << dir_.transpose();
        return ss.str();
    }
public:
    CumFluxProblem() : Base() {}
    CumFluxProblem(const Material* mat, const Domain* dom,
                   const Eigen::Vector3d& dir,
                   long nemit, long size, long maxscat, long maxloop = 0)
    : Base(mat, dom, nemit, maxscat, maxloop), dir_(dir),
    size_(size), step_(maxscat%size == 0 ? maxscat/size : maxscat/size + 1)
    {}
    Type initElem() const { return Eigen::ArrayXd::Zero(size_); }
    Solution solve(Rng& gen, Progress* prog = 0) const;
};

#endif
