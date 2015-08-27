//
//  problem.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#include "problem.h"
#include "field.h"
#include "domain.h"
#include "subdomain.h"
#include "boundary.h"
#include "material.h"
#include "phonon.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/optional.hpp>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

Clock::Clock()
{
    std::time(&start_);
}

std::string Clock::stopwatch()
{
    std::time_t now;
    std::time(&now);
    long diff = static_cast<long>( std::difftime(now, start_) );
    
    std::ostringstream ss;
    ss << std::setfill('0');
    ss << std::setw(2) << diff / 3600 << ':';
    ss << std::setw(2) << diff / 60 % 60 << ':';
    ss << std::setw(2) << diff % 60;
    return ss.str();
}

std::string Clock::timestamp()
{
    std::time_t now;
    std::time(&now);
    
    static const int size = 80;
    char buffer[size];
    std::strftime(buffer, size, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    
    return std::string(buffer);
}

Progress::Progress()
: tot_(0l), count_(0l), esc_(0l)
{}

Progress::Progress(long tot, long div)
: tot_(tot), count_(0l), div_(div), next_(0l), esc_(0l)
{
    BOOST_ASSERT_MSG(tot_ >= div_, "Too many divisions");
    for (long i = 1; i <= div_; ++i)
    {
        vec_.push_back(tot_ * i / div_);
    }
}

void Progress::clock(const Clock& clk)
{
    clk_ = clk;
}

long Progress::count() const
{
    return count_;
}

long Progress::esc() const
{
    return esc_;
}

long Progress::incrCount()
{
#pragma omp critical
    {
        count_++;
        
        if (next_ < div_ && count_ == vec_.at(next_))
        {
            next_++;
            std::cout << clk_.stopwatch() << ' ';
            std::cout << '[' << std::string(next_, '|');
            std::cout << std::string(div_ - next_, '-') << ']';
            std::cout << " esc: " << esc_ << std::endl;
            
            if (next_ == div_) std::cout << std::endl;
        }
    }
    return count_;
}

long Progress::incrEsc()
{
#pragma omp critical
    {
        esc_++;
    }
    return esc_;
}

template<typename Derived>
Problem<Derived>::Problem()
{}

template<typename Derived>
Problem<Derived>::Problem(const Material* mat, const Domain* dom)
: mat_(mat), dom_(dom)
{
    BOOST_ASSERT_MSG(dom_->isInit(), "Domain setup not complete");
}

template<typename Derived>
Problem<Derived>::~Problem()
{}

template<typename Derived>
const Material* Problem<Derived>::mat() const
{
    return mat_;
}

template<typename Derived>
const Domain* Problem<Derived>::dom() const
{
    return dom_;
}

template<typename Derived>
std::string Problem<Derived>::info() const
{
    std::ostringstream ss;
    ss << "  mat:     " << mat_ << std::endl;
    ss << "  dom:     " << dom_;
    return ss.str();
}

template<typename Derived>
std::ostream& operator<<(std::ostream& os, const Problem<Derived>& prob)
{
    return os << prob.info();
}

template class Problem< TrajProblem >;
template class Problem< TempProblem >;
template class Problem< FluxProblem >;
template class Problem< MultiProblem >;
template class Problem< CumTempProblem >;
template class Problem< CumFluxProblem >;

template std::ostream& operator<<(std::ostream& os,
                                  const Problem< TrajProblem >& prob);
template std::ostream& operator<<(std::ostream& os,
                                  const Problem< TempProblem >& prob);
template std::ostream& operator<<(std::ostream& os,
                                  const Problem< FluxProblem >& prob);
template std::ostream& operator<<(std::ostream& os,
                                  const Problem< MultiProblem >& prob);
template std::ostream& operator<<(std::ostream& os,
                                  const Problem< CumTempProblem >& prob);
template std::ostream& operator<<(std::ostream& os,
                                  const Problem< CumFluxProblem >& prob);

TrajProblem::TrajProblem()
: Base()
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Phonon::Prop& prop,
            const Eigen::Vector3d& pos, const Eigen::Vector3d& dir,
            long maxscat, long maxloop)
: Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop),
prop_(prop), pos_(pos), dir_(dir)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Phonon::Prop& prop, const Eigen::Vector3d& pos,
            long maxscat, long maxloop)
: Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop), prop_(prop), pos_(pos)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Eigen::Vector3d& pos, const Eigen::Vector3d& dir,
            long maxscat, long maxloop)
: Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop), pos_(pos), dir_(dir)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Eigen::Vector3d& pos,
            long maxscat, long maxloop)
: Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop), pos_(pos)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            long maxscat, long maxloop)
: Base(mat, dom), maxscat_(maxscat), maxloop_(maxloop)
{}

std::string TrajProblem::info() const
{
    static const Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "TrajProblem " << this << std::endl;
    ss << Base::info() << std::endl;
    if (prop_) ss << "  prop:    " << prop_->w() << " " <<
                                      prop_->p() << std::endl;
    if (pos_) ss << "  pos:     " << pos_->transpose().format(fmt) << std::endl;
    if (dir_) ss << "  dir:     " << dir_->transpose().format(fmt) << std::endl;
    ss << "  maxscat: " << maxscat_ << std::endl;
    ss << "  maxloop: " << maxloop_;
    return ss.str();
}

Progress TrajProblem::initProgress() const
{
    return Progress();
}

template<typename C, typename E>
long find(const C& cont, E elem) {
    //    unsigned long index;
    long index = std::find(cont.begin(), cont.end(), elem) - cont.begin();
    return (index == cont.size() ? -1l : index);
}

TrkPhonon::Matrix3Xd TrajProblem::solve(Rng& gen, Progress* prog) const
{
    TrkPhonon phn;
    const Subdomain* sdom = 0;
    const Boundary* bdry = 0;
    
    Phonon::Prop prop = (prop_ ? *prop_ : mat()->drawScatProp(gen));
    if (pos_)
    {
        Eigen::Vector3d pos = *pos_;
        Eigen::Vector3d dir = (dir_ ? *dir_ : drawIso(gen));
        
        phn = TrkPhonon(true, prop, pos, dir);
        
        sdom = dom()->locate(pos);
        bdry = 0;
        BOOST_ASSERT_MSG(sdom, "Position not inside domain");
    }
    else
    {
        UniformIntDist dist(0, dom()->emitPtrs().size() - 1);
        const Emitter* e = dom()->emitPtrs().at(dist(gen));
        
        phn = e->emit(prop, gen);
        
        sdom = e->emitSdom();
        bdry = e->emitBdry();
    }
    mat()->drawScatNext(phn, gen);
    
    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
    
    for (long i = 0; i < maxloop; i++)
    {
        std::cout << std::setw(2) << find(dom()->sdomPtrs(), sdom) << ": ";
        std::cout << std::setw(2) << find(sdom->bdryPtrs(), bdry)  << " ";
        std::cout << std::setw(5) << (bdry ? bdry->type() : "Null") << " -> ";
        
        double vel = mat()->vel(phn);
        bdry = sdom->advect(phn, vel);
        
        if (!phn.alive())
        {
            prog->incrEsc();
            break;
        }
        
        std::cout << std::setw(2) << find(sdom->bdryPtrs(), bdry)  << " ";
        std::cout << std::setw(5) << (bdry ? bdry->type() : "Null");
        std::cout << std::endl;
        
        if (bdry)
        {
            bdry = bdry->scatter(phn, gen);
            
            if (!bdry)
            {
                prog->incrEsc(); // Inter
                break;
            }
            
            sdom = bdry->sdom();
        }
        else
        {
            mat()->scatter(phn, gen);
        }
        if (!phn.alive() || phn.nscat() >= maxscat_) break;
    }
    
    if (prog->esc() > 0l) std::cout << "Escaped" << std::endl;
    
    return phn.trajectory();
}

template<typename T, int N>
typename CellVolF<T, N>::VectorNT
CellVolF<T, N>::operator()(const Subdomain* sdom, const Vector3l& index) const
{
    double scalar = sdom->cellVol(index);
    return VectorNT::Constant(std::max(1, N), scalar);
}

template struct CellVolF<double, Eigen::Dynamic>;
template struct CellVolF<double, 1>;
template struct CellVolF<double, 2>;
template struct CellVolF<double, 3>;
template struct CellVolF<double, 4>;

template<typename Derived>
FieldProblem<Derived>::FieldProblem()
{}

template<typename Derived>
FieldProblem<Derived>::FieldProblem(const Material* mat, const Domain* dom,
                                    long nemit, long maxscat, long maxloop)
: Base(mat, dom), nemit_(nemit), maxscat_(maxscat), maxloop_(maxloop)
{}

template<typename Derived>
FieldProblem<Derived>::~FieldProblem()
{}

template<typename Derived>
std::string FieldProblem<Derived>::info() const
{
    std::ostringstream ss;
    ss << Base::info() << std::endl;
    ss << "  nemit:   " << nemit_ << std::endl;
    ss << "  maxscat: " << maxscat_ << std::endl;
    ss << "  maxloop: " << maxloop_;
    return ss.str();
}

template<typename Derived>
Progress FieldProblem<Derived>::initProgress() const
{
    return Progress(nemit_, std::min(20l, nemit_));
}

template<typename Derived>
typename FieldProblem<Derived>::Solution
FieldProblem<Derived>::initSolution() const
{
    return Solution(Base::dom());
}

template<typename Derived>
typename FieldProblem<Derived>::Solution
FieldProblem<Derived>::solve(Rng& gen, Progress* prog) const
{
    Field<Type, Num> fld(Base::dom());
    
    const Emitter::Pointers emitPtrs = Base::dom()->emitPtrs();
    long nemitter = emitPtrs.size();
    
    Eigen::ArrayXd weight(nemitter);
    for (int i = 0; i < nemitter; ++i)
    {
        weight(i) = emitPtrs.at(i)->emitWeight();
    }
    double weightSum = weight.sum();
    
    Eigen::Array<long, Eigen::Dynamic, 1> emitCdf(nemitter);
    for (int i = 0; i < nemitter; ++i)
    {
        double frac = weight(i) / weightSum;
        double rounded = std::ceil(frac * nemit_ - 0.5);
        emitCdf(i) = std::max(1l, static_cast<long>(rounded));
        if (i > 0) emitCdf(i) += emitCdf(i - 1);
    }
    long nemit = emitCdf(nemitter - 1);
    
    double power = weightSum / nemit * Base::mat()->fluxSum() / 4.;
    
    FieldAccumF fun = accumFun();
    
    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
    
    long* emitCdfBegin = emitCdf.data();
    long* emitCdfEnd = emitCdf.data() + nemitter;
    
#pragma omp for schedule(static)
    for (long n = 0; n < nemit; ++n)
    {
        long emitIndex = (std::upper_bound(emitCdfBegin, emitCdfEnd, n) -
                          emitCdfBegin);
        
        const Emitter* e = emitPtrs.at(emitIndex);
        const Subdomain* sdom = e->emitSdom();
        const Boundary* bdry = e->emitBdry();
        
        Phonon::Prop prop = Base::mat()->drawFluxProp(gen);
#ifdef DEBUG
        TrkPhonon phn = e->emit(prop, gen);
#else
        Phonon phn = e->emit(prop, gen);
#endif
        Base::mat()->drawScatNext(phn, gen);
        
        for (long i = 0; i < maxloop; i++)
        {
            Phonon pre(phn);
            
            double vel = Base::mat()->vel(phn);
            bdry = sdom->advect(phn, vel);
            
            if (!phn.alive())
            {
                prog->incrEsc();
                break;
            }
            
            VectorNT amount = phn.sign() * fun(pre, phn);
            
            fld.accumulate(sdom, pre.pos(), phn.pos(), amount);
            
            if (bdry)
            {
                bdry = bdry->scatter(phn, gen);
                
                if (!bdry)
                {
                    prog->incrEsc(); // Inter
                    break;
                }
                
                sdom = bdry->sdom();
            }
            else
            {
                Base::mat()->scatter(phn, gen);
            }
            if (!phn.alive() || phn.nscat() >= maxscat_) break;
        }
        prog->incrCount();
    }
    
    Field<Type, Num> factor(power * postMult());
    
    Field<Type, Num> cellVol(Base::dom(), CellVolF<Type, Num>());
    
    return factor * fld.transform(fun) / cellVol;
}

template class FieldProblem< TempProblem >;
template class FieldProblem< FluxProblem >;
template class FieldProblem< MultiProblem >;
template class FieldProblem< CumTempProblem >;
template class FieldProblem< CumFluxProblem >;

TempAccumF::VectorNT TempAccumF::operator()(const Phonon& before,
                                            const Phonon& after) const
{
    Eigen::Matrix<double, 1, 1> vec;
    vec(0) = after.time() - before.time();
    return vec;
}

TempAccumF::VectorNT TempAccumF::operator()(const VectorNT& elem) const
{
    return elem;
}

TempProblem::TempProblem()
: Base()
{}

TempProblem::TempProblem(const Material* mat, const Domain* dom,
                         long nemit, long maxscat, long maxloop)
: Base(mat, dom, nemit, maxscat, maxloop)
{}

std::string TempProblem::info() const
{
    std::ostringstream ss;
    ss << "TempProblem " << this << std::endl;
    ss << Base::info();
    return ss.str();
}

TempProblem::VectorNT TempProblem::postMult() const
{
    Eigen::Matrix<double, 1, 1> vec;
    vec(0) = 1. / mat()->energySum();
    return vec;
}

TempProblem::FieldAccumF TempProblem::accumFun() const
{
    return TempAccumF();
}

FluxAccumF::FluxAccumF(const Eigen::Vector3d& dir)
: dir_(dir)
{}

FluxAccumF::VectorNT FluxAccumF::operator()(const Phonon& before,
                                            const Phonon& after) const
{
    Eigen::Matrix<double, 1, 1> vec;
    vec(0) = (after.pos() - before.pos()).dot(dir_);
    return vec;
}

FluxAccumF::VectorNT FluxAccumF::operator()(const VectorNT& elem) const
{
    return elem;
}

FluxProblem::FluxProblem()
: Base()
{}

FluxProblem::FluxProblem(const Material* mat, const Domain* dom,
                         const Eigen::Vector3d& dir,
                         long nemit, long maxscat, long maxloop)
: Base(mat, dom, nemit, maxscat, maxloop), dir_(dir.normalized())
{}

std::string FluxProblem::info() const
{
    static const Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "FluxProblem " << this << std::endl;
    ss << Base::info() << std::endl;
    ss << "  dir:     " << dir_.transpose().format(fmt);
    return ss.str();
}

FluxProblem::VectorNT FluxProblem::postMult() const
{
    return Eigen::Matrix<double, 1, 1>::Ones();
}

FluxProblem::FieldAccumF FluxProblem::accumFun() const
{
    return FluxAccumF(dir_);
}

MultiAccumF::MultiAccumF(const Eigen::Matrix3d& rot)
: inv_(rot.transpose())
{}

MultiAccumF::VectorNT MultiAccumF::operator()(const Phonon& before,
                                              const Phonon& after) const
{
    double dtime = after.time() - before.time();
    Eigen::Vector3d dpos = after.pos() - before.pos();
    return (Eigen::Vector4d() << dtime, dpos).finished();
}

MultiAccumF::VectorNT MultiAccumF::operator()(const VectorNT& elem) const
{
    Eigen::Vector4d newValue;
    newValue << elem(0), inv_ * elem.tail<3>();
    return newValue;
}

MultiProblem::MultiProblem()
: Base()
{}

MultiProblem::MultiProblem(const Material* mat, const Domain* dom,
                           const Eigen::Matrix3d& rot,
                           long nemit, long maxscat, long maxloop)
: Base(mat, dom, nemit, maxscat, maxloop), rot_(rot)
{
    BOOST_ASSERT_MSG(Matrix3d::Identity().isApprox(rot * rot.transpose()),
                     "Direction matrix must be orthogonal");
}

MultiProblem::MultiProblem(const Material* mat, const Domain* dom,
                           long nemit, long maxscat, long maxloop)
: Base(mat, dom, nemit, maxscat, maxloop), rot_(Matrix3d::Identity())
{}

std::string MultiProblem::info() const
{
    static const Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "MultiProblem " << this << std::endl;
    ss << Base::info() << std::endl;
    ss << "  rot:     " << rot_.format(fmt);
    return ss.str();
}

MultiProblem::VectorNT MultiProblem::postMult() const
{
    return Eigen::Vector4d(1. / mat()->energySum(), 1., 1., 1.);
}

MultiProblem::FieldAccumF MultiProblem::accumFun() const
{
    return MultiAccumF(rot_);
}

template<typename F>
CumF<F>::CumF(long size, long step, const F& fun)
: size_(size), step_(step), fun_(fun)
{}

template<typename F>
typename CumF<F>::VectorNT CumF<F>::operator()(const Phonon& before,
                                      const Phonon& after) const
{
    long index = (before.nscat() + step_ - 1) / step_;
    BOOST_ASSERT_MSG(index <= size_, "Index out of bounds");
    
    Eigen::Matrix<double, 1, 1> value = fun_(before, after);
    return value(0) * Eigen::VectorXd::Unit(size_ + 1, index);
}

template<typename F>
typename CumF<F>::VectorNT CumF<F>::operator()(const VectorNT& elem) const
{
    Eigen::VectorXd cumsum(elem);
    for (long i = 1; i <= size_; ++i)
    {
        cumsum(i) += cumsum(i-1);
    }
    return cumsum;
}

CumTempAccumF::CumTempAccumF(long size, long step, const TempAccumF& fun)
: CumF<TempAccumF>(size, step, fun)
{}

CumTempProblem::CumTempProblem()
: Base()
{}

CumTempProblem::CumTempProblem(const Material* mat, const Domain* dom,
                               long nemit, long size,
                               long maxscat, long maxloop)
: Base(mat, dom, nemit, maxscat, maxloop), size_(size)
{
    step_ = (maxscat - 1) / size;
    if ((maxscat - 1) % size != 0l) step_++;
    
}

std::string CumTempProblem::info() const
{
    std::ostringstream ss;
    ss << "CumTempProblem " << this << std::endl;
    ss << Base::info() << std::endl;
    ss << "  size:    " << size_;
    return ss.str();
}

CumTempProblem::VectorNT CumTempProblem::postMult() const
{
    double scalar = 1. / mat()->energySum();
    return Eigen::VectorXd::Constant(size_ + 1, scalar);
}

CumTempProblem::FieldAccumF CumTempProblem::accumFun() const
{
    return CumTempAccumF(size_, step_, TempAccumF());
}

CumFluxAccumF::CumFluxAccumF(long size, long step, const FluxAccumF& fun)
: CumF<FluxAccumF>(size, step, fun)
{}

CumFluxProblem::CumFluxProblem()
: Base()
{}

CumFluxProblem::CumFluxProblem(const Material* mat, const Domain* dom,
                               const Eigen::Vector3d& dir,
                               long nemit, long size,
                               long maxscat, long maxloop)
: Base(mat, dom, nemit, maxscat, maxloop), dir_(dir.normalized()), size_(size)
{
    step_ = (maxscat - 1) / size;
    if ((maxscat - 1) % size != 0l) step_++;
    
}

std::string CumFluxProblem::info() const
{
    static const Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << Base::info() << std::endl;
    ss << "  size:    " << size_ << std::endl;
    ss << "  dir:     " << dir_.transpose().format(fmt);
    return ss.str();
}

CumFluxProblem::VectorNT CumFluxProblem::postMult() const
{
    return Eigen::VectorXd::Ones(size_ + 1);
}

CumFluxProblem::FieldAccumF CumFluxProblem::accumFun() const
{
    return CumFluxAccumF(size_, step_, FluxAccumF(dir_));
}


//struct AccumWeightF {
//    double operator()(double weight, const Emitter* emit) const {
//        return weight + emit->emitWeight();
//    }
//};
//
//struct CalcEmitF {
//    double totWeight;
//    long nemit;
//    CalcEmitF(double w, long n) : totWeight(w), nemit(n) {}
//    long operator()(const Emitter* emit) const {
//        double frac = emit->emitWeight() / totWeight;
//        double rounded = std::ceil(frac * nemit - 0.5);
//        return std::max(1l, static_cast<long>(rounded));
//    }
//};

//template<typename T, typename F>
//struct CalcFieldF {
//    F factor;
//    CalcFieldF(const F& fac) : factor(fac) {}
//    void operator()(typename Field<T>::value_type& pair) const {
//        const Subdomain* sdom = pair.first;
//        Data<T>& data = pair.second;
//        Collection coll(0,0,0);
//        Eigen::Map<Collection::Vector3s> index(coll.data());
//        typedef typename Data<T>::Size Size;
//        for (Size k = 0; k < data.shape()[2]; ++k) {
//            coll[2] = k;
//            for (Size j = 0; j < data.shape()[1]; ++j) {
//                coll[1] = j;
//                for (Size i = 0; i < data.shape()[0]; ++i) {
//                    coll[0] = i;
//                    data(coll) *= factor / sdom->cellVol( index.cast<long>() );
//                }
//            }
//        }
//    }
//};

//template<typename Derived>
//struct CalcFieldF {
//    typedef typename Derived::Type Type;
//    typedef typename Derived::Factor Factor;
//    typedef typename Derived::AccumF AccumF;
//    AccumF functor;
//    Factor factor;
//    CalcFieldF(const AccumF& fun, const Factor& pow)
//    : functor(fun), factor(pow) {}
//    void operator()(typename Field<Type>::value_type& pair) const {
//        const Subdomain* sdom = pair.first;
//        Data<Type>& data = pair.second;
//        Collection coll(0,0,0);
//        Eigen::Map<Collection::Vector3s> index(coll.data());
//        typedef typename Data<Type>::Size Size;
//        for (Size k = 0; k < data.shape()[2]; ++k) {
//            coll[2] = k;
//            for (Size j = 0; j < data.shape()[1]; ++j) {
//                coll[1] = j;
//                for (Size i = 0; i < data.shape()[0]; ++i) {
//                    coll[0] = i;
//                    data(coll) = factor / sdom->cellVol( index.cast<long>() ) *
//                                 functor( data(coll) );
//                }
//            }
//        }
//    }
//};

//struct State {
//    double time;
//    Phonon phn;
//    long nscatMat;
//    State(double t, const Phonon& p, long n)
//    : time(t), phn(p), nscatMat(n) {}
//};

//template<typename Derived>
//Field<typename FieldProblem<Derived>::Type>
//FieldProblem<Derived>::solveField(const AccumF& fun, const Factor& fac,
//                                  Progress* prog, Rng& gen) const
//{
//    Field<Type> fld = initField();
//
//    const Domain::EmitPtrs& emitPtrs = dom()->emitPtrs();
//    double totWeight = std::accumulate(emitPtrs.begin(), emitPtrs.end(),
//                                       0., AccumWeightF());
//    std::vector<long> emitPdf( emitPtrs.size() );
//    std::transform(emitPtrs.begin(), emitPtrs.end(),
//                   emitPdf.begin(), CalcEmitF(totWeight, nemit_));
//    std::vector<long> emitCdf( emitPtrs.size() );
//    std::partial_sum(emitPdf.begin(), emitPdf.end(), emitCdf.begin());
//    long nemit = emitCdf.back();
//    double power = totWeight / nemit * mat()->fluxSum() / 4.;
//
//    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
//#pragma omp for schedule(static)
//    for (long n = 0; n < nemit; ++n) {
//        std::vector<long>::size_type eIndex;
//        eIndex = (std::upper_bound(emitCdf.begin(), emitCdf.end(), n) -
//                  emitCdf.begin());
//
//        const Emitter* e = emitPtrs.at(eIndex);
//        const Subdomain* sdom = e->emitSdom();
//        const Boundary* bdry = e->emitBdry();
//#ifdef DEBUG
//        TrkPhonon phn = e->emit(mat()->drawFluxProp(gen), gen);
//#else
//        Phonon phn = e->emit(mat()->drawFluxProp(gen), gen);
//#endif
//        double time = 0.;
//        double scatDist = mat()->drawScatDist(phn.prop(), gen);
//        for (long i = 0; i < maxloop; i++) {
////            State before(time, phn, nscatMat);
//            bdry = sdom->advect(phn, scatDist);
//            time += ((phn.pos() - before.phn.pos()).norm() /
//                     mat()->vel(phn.prop()));
////            State after(time, phn, nscatMat);
//
////            try {
////                sdom->accumulate<Type>(before.phn.pos(),
////                                       after.phn.pos(),
////                                       fld[sdom],
////                                       phn.sign() * fun(before, after));
////            } catch(const Grid::OutOfRange& e) {
////#ifdef DEBUG
////                std::cerr << e.what() << std::endl;
////                std::cerr << "Trajectory" << std::endl;
////                std::cerr << phn.trajectory() << std::endl;
////#endif
////            }
//            sdom->accumulate<Type>(phn, before.phn.pos(), after.phn.pos(),
//                                   fld[sdom], phn.sign()*fun(before, after));
//            if (!phn.alive()) {
//                incrEsc(prog);
//                break;
//            }
//
//            if (bdry) {
//                Boundary::Pointers bdryPtrs = bdry->scatter(phn, gen);
//                if (!phn.alive()) break;
//                for (Boundary::Pointers::const_iterator b = bdryPtrs.begin();
//                     b != bdryPtrs.end(); ++b)
//                {
//                    bdry = *b;
//                    sdom = bdry->sdom();
//                    if (bdryPtrs.size() == 1 || sdom->isInside( phn.pos() )) {
//                        break;
//                    }
//                    sdom = 0;
//                }
//                if (!sdom) {
//                    incrEsc(prog);
//                    break;
//                }
//            } else {
//                scatDist = mat()->scatter(phn, gen);
//            }
//            if (!phn.alive() || phn.nscat() >= maxscat_) break;
//        }
//        incrCount(prog);
//    }
//
//    std::for_each(fld.begin(), fld.end(), CalcFieldF<Derived>(fun, fac*power));
//    return fld;
//}
