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
#include <boost/type_traits/is_scalar.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
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
        std::stringstream ss;
        ss << std::setfill('0');
        ss << std::setw(2) << diff / 3600 << ':';
        ss << std::setw(2) << diff / 60 << ':';
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

class Problem {
protected:
    static const long loopFactor_ = 100;
    const Material* mat_;
    const Domain* dom_;
    long nemit_, maxscat_, maxloop_;
public:
    Problem() : mat_(0), dom_(0) {}
    Problem(const Material* mat, const Domain* dom,
            long nemit, long maxscat, long maxloop)
    : mat_(mat), dom_(dom), nemit_(nemit), maxscat_(maxscat), maxloop_(maxloop)
    {
        BOOST_ASSERT_MSG(dom_->isInit(), "Domain setup not complete");
    };
    virtual ~Problem() {}
    virtual Progress initProgress() const = 0;
    template<typename T>
    Field<T> initField() const {
        Field<T> fld;
        const Domain::SdomPtrs& sdomPtrs = dom_->sdomPtrs();
        std::for_each(sdomPtrs.begin(), sdomPtrs.end(), AddRegionF<T>(fld));
        return fld;
    }
private:
    template<typename T>
    struct AddRegionF {
        Field<T>& fld_;
        AddRegionF(Field<T>& fld) : fld_(fld) {}
        void operator()(const Subdomain* sdom) {
            typedef typename Field<T>::value_type Value;
            fld_.insert( Value(sdom, sdom->initData<T>()) );
        }
    };
protected:
    template<typename T, typename F, typename M>
    Field<T> solveField(const F& functor, M factor,
                        Rng& gen, Progress* prog) const;
};

class Solution {
protected:
    static const int precision_ = 9;
    static const int width_ = precision_ + 7;
public:
    Solution() {}
    virtual ~Solution() {}
    virtual const Problem* problem() const = 0;
    friend std::ostream& operator<<(std::ostream& os, const Solution& sol) {
        std::ios_base::fmtflags flags = std::cout.flags();
        sol.print(os, precision_, width_);
        return os << std::setiosflags(flags);
    }
private:
    virtual void print(std::ostream& os, int precision, int width) const = 0;
};

class Trajectory;

class TrajProblem : public Problem {
public:
    typedef Trajectory Sol;
private:
    boost::optional<Phonon::Prop> prop_;
    boost::optional<Eigen::Vector3d> pos_, dir_;
public:
    TrajProblem() : Problem() {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop,
                const Eigen::Vector3d& pos, const Eigen::Vector3d& dir,
                long maxscat = 100, long maxloop = 0)
    : Problem(mat, dom, 1, maxscat, maxloop), prop_(prop), pos_(pos), dir_(dir)
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Phonon::Prop& prop, const Eigen::Vector3d& pos,
                long maxscat = 100, long maxloop = 0)
    : Problem(mat, dom, 1, maxscat, maxloop), prop_(prop), pos_(pos), dir_()
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Eigen::Vector3d& pos, const Eigen::Vector3d& dir,
                long maxscat = 100, long maxloop = 0)
    : Problem(mat, dom, 1, maxscat, maxloop), prop_(), pos_(pos), dir_(dir)
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                const Eigen::Vector3d& pos,
                long maxscat = 100, long maxloop = 0)
    : Problem(mat, dom, 1, maxscat, maxloop), prop_(), pos_(pos), dir_()
    {}
    TrajProblem(const Material* mat, const Domain* dom,
                long maxscat = 100, long maxloop = 0)
    : Problem(mat, dom, 1, maxscat, maxloop), prop_(), pos_(), dir_()
    {}
    Progress initProgress() const {
        return Progress(maxscat_, std::min(10l, maxscat_));
    }
    Sol solve(Rng& gen, Progress* prog = 0) const;
};

class Trajectory : public Solution {
private:
    const TrajProblem* prob_;
    Eigen::ArrayXXd arr_;
public:
    Trajectory() : Solution(), prob_(0), arr_() {}
    Trajectory(const TrajProblem* prob, const TrkPhonon::Traj& traj)
    : Solution(), prob_(prob), arr_(traj.size(), 3)
    {
        Eigen::ArrayXXd::Index i = 0;
        for (TrkPhonon::Traj::const_iterator x = traj.begin(); x != traj.end();
             ++x) {
            arr_.row(i++) = *x;
        }
    }
    const TrajProblem* problem() const { return prob_; }
private:
    void print(std::ostream& os, int precision, int width) const {
        Eigen::IOFormat fmt(precision, width);
        os << "Trajectory" << std::endl;
        os << std::scientific << arr_.format(fmt) << std::endl;
    }
};

class Temperature;

class TempProblem : public Problem {
public:
    typedef Temperature Sol;
    TempProblem() : Problem() {}
    TempProblem(const Material* mat, const Domain* dom, long nemit,
                long maxscat, long maxloop = 0)
    : Problem(mat, dom, nemit, maxscat, maxloop)
    {}
    Progress initProgress() const {
        return Progress(nemit_, std::min(20l, nemit_));
    }
    Sol solve(Rng& gen, Progress* prog = 0) const;
private:
    struct TempAccumF;
};

class Temperature : public Solution {
private:
    const TempProblem* prob_;
    Field<double> fld_;
public:
    Temperature() : Solution(), prob_(0), fld_() {}
    Temperature(const TempProblem* prob)
    : Solution(), prob_(prob), fld_(prob->initField<double>())
    {}
    Temperature(const TempProblem* prob, const Field<double>& fld)
    : Solution(), prob_(prob), fld_(fld)
    {}
    const TempProblem* problem() const { return prob_; }
    Temperature& operator+=(const Temperature& temp) {
        BOOST_ASSERT_MSG(prob_ == temp.prob_,
                         "Cannot add solutions to different problems");
        fld_ += temp.fld_;
        return *this;
    }
private:
    void print(std::ostream& os, int precision, int width) const {
        os << "Temperature" << std::endl;
        fld_.print(os, precision, width);
    }
};

class Flux;

class FluxProblem : public Problem {
public:
    typedef Flux Sol;
private:
    Eigen::Vector3d dir_;
public:
    FluxProblem() : Problem() {}
    FluxProblem(const Material* mat, const Domain* dom,
                const Eigen::Vector3d& dir, long nemit,
                long maxscat, long maxloop = 0)
    : Problem(mat, dom, nemit, maxscat, maxloop), dir_(dir)
    {}
    Progress initProgress() const {
        return Progress(nemit_, std::min(20l, nemit_));
    }
    Sol solve(Rng& gen, Progress* prog = 0) const;
private:
    struct FluxAccumF;
};

class Flux : public Solution {
private:
    const FluxProblem* prob_;
    Field<double> fld_;
public:
    Flux() : Solution(), prob_(0), fld_() {}
    Flux(const FluxProblem* prob)
    : Solution(), prob_(prob), fld_(prob->initField<double>())
    {}
    Flux(const FluxProblem* prob, const Field<double>& fld)
    : Solution(), prob_(prob), fld_(fld)
    {}
    const FluxProblem* problem() const { return prob_; }
    Flux& operator+=(const Flux& flux) {
        BOOST_ASSERT_MSG(prob_ == flux.prob_,
                         "Cannot add solutions to different problems");
        fld_ += flux.fld_;
        return *this;
    }
private:
    void print(std::ostream& os, int precision, int width) const {
        os << "Flux" << std::endl;
        fld_.print(os, precision, width);
    }
};

class MultiSolution;

class MultiProblem : public Problem {
public:
    typedef MultiSolution Sol;
private:
    Eigen::Matrix3d inv_;
public:
    MultiProblem() : Problem() {}
    MultiProblem(const Material* mat, const Domain* dom,
                 const Eigen::Matrix3d& rot, long nemit,
                 long maxscat, long maxloop = 0)
    : Problem(mat, dom, nemit, maxscat, maxloop), inv_(rot.transpose())
    {
        BOOST_ASSERT_MSG(Eigen::Matrix3d::Identity().isApprox(inv_*rot),
                         "Direction matrix must be unitary");
    }
    Progress initProgress() const {
        return Progress(nemit_, std::min(20l, nemit_));
    }
    Sol solve(Rng& gen, Progress* prog = 0) const;
private:
    struct MultiAccumF;
};

class MultiSolution : public Solution {
private:
    const MultiProblem* prob_;
    Field<Eigen::Vector4d> fld_;
public:
    MultiSolution() : Solution(), prob_(0), fld_() {}
    MultiSolution(const MultiProblem* prob)
    : Solution(), prob_(prob), fld_(prob->initField<Eigen::Vector4d>())
    {}
    MultiSolution(const MultiProblem* prob, const Field<Eigen::Vector4d>& fld)
    : Solution(), prob_(prob), fld_(fld)
    {}
    const MultiProblem* problem() const { return prob_; }
    MultiSolution& operator+=(const MultiSolution& multi) {
        BOOST_ASSERT_MSG(prob_ == multi.prob_,
                         "Cannot add solutions to different problems");
        fld_ += multi.fld_;
        return *this;
    }
private:
    void print(std::ostream& os, int precision, int width) const {
        os << "Temperature and flux" << std::endl;
        fld_.print(os, precision, width);
    }
};

#endif
