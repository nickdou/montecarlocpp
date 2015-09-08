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

//----------------------------------------
//  Helper classes
//----------------------------------------

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
: tot_(0), count_(0), esc_(0)
{}

Progress::Progress(long tot, long div)
: tot_(tot), count_(0), div_(div), next_(0), esc_(0)
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
        }
        
        if (count_ == tot_) std::cout << std::endl;
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

//----------------------------------------
//  Problem
//----------------------------------------

Problem::Problem()
{}

Problem::Problem(const Material* mat, const Domain* dom)
: mat_(mat), dom_(dom)
{
    BOOST_ASSERT_MSG(dom_->isInit(), "Domain setup not complete");
}

Problem::~Problem()
{}

const Material* Problem::mat() const
{
    return mat_;
}

const Domain* Problem::dom() const
{
    return dom_;
}

std::string Problem::info() const
{
    std::ostringstream ss;
    ss << "  mat:     " << mat_ << std::endl;
    ss << "  dom:     " << dom_;
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Problem& prob)
{
    return os << prob.info();
}

//----------------------------------------
//  Trajectory problem
//----------------------------------------

TrajProblem::TrajProblem()
: Problem()
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Phonon::Prop& prop,
            const Vector3d& pos, const Vector3d& dir,
            long maxscat, long maxloop)
: Problem(mat, dom), maxscat_(maxscat), maxloop_(maxloop), prop_(prop),
pos_(pos), dir_(dir)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Phonon::Prop& prop, const Vector3d& pos,
            long maxscat, long maxloop)
: Problem(mat, dom), maxscat_(maxscat), maxloop_(maxloop), prop_(prop),
pos_(pos)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Vector3d& pos, const Vector3d& dir,
            long maxscat, long maxloop)
: Problem(mat, dom), maxscat_(maxscat), maxloop_(maxloop), pos_(pos), dir_(dir)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            const Vector3d& pos,
            long maxscat, long maxloop)
: Problem(mat, dom), maxscat_(maxscat), maxloop_(maxloop), pos_(pos)
{}

TrajProblem::TrajProblem(const Material* mat, const Domain* dom,
            long maxscat, long maxloop)
: Problem(mat, dom), maxscat_(maxscat), maxloop_(maxloop)
{}

std::string TrajProblem::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "TrajProblem " << static_cast<const Problem*>(this) << std::endl;
    ss << Problem::info() << std::endl;
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

template<typename T>
long find(const std::vector<T>& cont, const T& elem)
{
    long index = std::find(cont.begin(), cont.end(), elem) - cont.begin();
    return (index == cont.size() ? -1 : index);
}

ArrayXXd TrajProblem::solve(Rng& gen, Progress* prog) const
{
    const Subdomain* sdom = 0;
    const Boundary* bdry = 0;
    
    TrkPhonon phn;
    Phonon::Prop prop = (prop_ ? *prop_ : mat()->drawScatProp(gen));
    if (pos_)
    {
        Vector3d pos = *pos_;
        Vector3d dir = (dir_ ? *dir_ : drawIso(gen));
        
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
    
    if (prog->esc() > 0) std::cout << "Escaped" << std::endl;
    
    return phn.trajectory();
}


VectorXd CellVolF::operator()(const Subdomain* sdom,
                                     const Vector3l& index) const
{
    return VectorXd::Constant(1, sdom->cellVol(index));
}

//----------------------------------------
//  Field problem
//----------------------------------------

FieldProblem::FieldProblem()
{}

FieldProblem::FieldProblem(const Material* mat, const Domain* dom,
                           long nemit, long maxscat, long maxloop)
: Problem(mat, dom)
{
    const Emitter::Pointers emitPtrs = dom->emitPtrs();
    long nemitter = emitPtrs.size();
    
    VectorXd weight(nemitter);
    for (long i = 0; i < nemitter; ++i)
    {
        weight(i) = emitPtrs.at(i)->emitWeight();
    }
    double weightSum = weight.sum();
    
    emitPdf_.resize(nemitter);
    for (long i = 0; i < nemitter; ++i)
    {
        double frac = weight(i) / weightSum;
        double rounded = std::ceil(frac * nemit - 0.5);
        emitPdf_(i) = std::max(1l, static_cast<long>(rounded));
    }
    
    nemit_ = emitPdf_.sum();
    maxscat_ = maxscat;
    maxloop_ = (maxloop != 0 ? maxloop : loopFactor_ * maxscat_);
    
    power_ = weightSum / nemit_ * mat->fluxSum() / 4.;
}

FieldProblem::~FieldProblem()
{}

std::string FieldProblem::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << Problem::info() << std::endl;
    ss << "  emitpdf: " << emitPdf_.transpose().format(fmt) << std::endl;
    ss << "  nemit:   " << nemit_ << std::endl;
    ss << "  maxscat: " << maxscat_ << std::endl;
    ss << "  maxloop: " << maxloop_ << std::endl;
    ss << "  power:   " << power_;
    return ss.str();
}

Progress FieldProblem::initProgress() const
{
    return Progress(nemit_, std::min(20l, nemit_));
}

ArrayXXd FieldProblem::initSolution() const
{
    return Field(rows(), dom()).data();
}

ArrayXXd FieldProblem::solve(Rng& gen, Progress* prog) const
{
    Field fld(rows(), dom());
    
    VectorXl emitCdf(emitPdf_);
    for (long i = 1; i < emitCdf.size(); ++i)
    {
        emitCdf(i) += emitCdf(i - 1);
    }
    
    const long* emitCdfBegin = emitCdf.data();
    const long* emitCdfEnd = emitCdf.data() + emitCdf.size();
    
#pragma omp for schedule(static)
    for (long n = 0; n < nemit_; ++n)
    {
        long emitIndex = (std::upper_bound(emitCdfBegin, emitCdfEnd, n) -
                          emitCdfBegin);
        
        const Emitter* e = dom()->emitPtrs().at(emitIndex);
        const Subdomain* sdom = e->emitSdom();
        const Boundary* bdry = e->emitBdry();
        
        Phonon::Prop prop = mat()->drawFluxProp(gen);
#ifdef DEBUG
        TrkPhonon phn = e->emit(prop, gen);
#else
        Phonon phn = e->emit(prop, gen);
#endif
        mat()->drawScatNext(phn, gen);
        
        for (long i = 0; i < maxloop_; i++)
        {
            Phonon pre(phn);
            
            double vel = mat()->vel(phn);
            bdry = sdom->advect(phn, vel);
            
            if (!phn.alive())
            {
                prog->incrEsc();
                break;
            }
            
            VectorXd amount = phn.sign() * accumAmt(pre, phn);
            
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
                mat()->scatter(phn, gen);
            }
            if (!phn.alive() || phn.nscat() >= maxscat_) break;
        }
        prog->incrCount();
    }
    
    ArrayXXd sol = postProc(fld.data());
    
    Eigen::Array<double, 1, Eigen::Dynamic> vol;
    vol = Field(1, dom(), CellVolF()).data().row(0);
    
    return power_ * (sol.rowwise() / vol);
}

//----------------------------------------
//  Field problem implementations
//----------------------------------------

TempProblem::TempProblem()
: FieldProblem()
{}

TempProblem::TempProblem(const Material* mat, const Domain* dom,
                         long nemit, long maxscat, long maxloop)
: FieldProblem(mat, dom, nemit, maxscat, maxloop)
{}

std::string TempProblem::info() const
{
    std::ostringstream ss;
    ss << "TempProblem " << static_cast<const Problem*>(this) << std::endl;
    ss << FieldProblem::info();
    return ss.str();
}

long TempProblem::rows() const
{
    return 1;
}

VectorXd TempProblem::accumAmt(const Phonon& before, const Phonon& after) const
{
    return VectorXd::Constant(1, after.time() - before.time());
}

ArrayXXd TempProblem::postProc(const ArrayXXd& data) const
{
    return data / mat()->energySum();
}

FluxProblem::FluxProblem()
: FieldProblem()
{}

FluxProblem::FluxProblem(const Material* mat, const Domain* dom,
                         long nemit, long maxscat, long maxloop)
: FieldProblem(mat, dom, nemit, maxscat, maxloop)
{}

std::string FluxProblem::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "FluxProblem " << static_cast<const Problem*>(this) << std::endl;
    ss << FieldProblem::info();
    return ss.str();
}

long FluxProblem::rows() const
{
    return 3;
}

VectorXd FluxProblem::accumAmt(const Phonon& before, const Phonon& after) const
{
    return after.pos() - before.pos();
}

ArrayXXd FluxProblem::postProc(const ArrayXXd& data) const
{
    return data;
}

MultiProblem::MultiProblem()
: FieldProblem()
{}

MultiProblem::MultiProblem(const Material* mat, const Domain* dom,
                           long nemit, long maxscat, long maxloop)
: FieldProblem(mat, dom, nemit, maxscat, maxloop)
{}

std::string MultiProblem::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "MultiProblem " << static_cast<const Problem*>(this) << std::endl;
    ss << FieldProblem::info();
    return ss.str();
}

long MultiProblem::rows() const
{
    return 4;
}

VectorXd MultiProblem::accumAmt(const Phonon& before, const Phonon& after) const
{
    double dtime = after.time() - before.time();
    Vector3d dpos = after.pos() - before.pos();
    return (Eigen::Vector4d() << dtime, dpos).finished();
}

ArrayXXd MultiProblem::postProc(const ArrayXXd& data) const
{
    ArrayXXd sol(data);
    sol.row(0) /= mat()->energySum();
    return sol;
}

CumTempProblem::CumTempProblem()
: FieldProblem()
{}

CumTempProblem::CumTempProblem(const Material* mat, const Domain* dom,
                               long nemit, long size,
                               long maxscat, long maxloop)
: FieldProblem(mat, dom, nemit, maxscat, maxloop), size_(size)
{
    step_ = (maxscat - 1) / size;
    if ((maxscat - 1) % size != 0) step_++;
    
}

std::string CumTempProblem::info() const
{
    std::ostringstream ss;
    ss << "CumTempProblem " << static_cast<const Problem*>(this) << std::endl;
    ss << FieldProblem::info() << std::endl;
    ss << "  size:    " << size_;
    return ss.str();
}

long CumTempProblem::rows() const
{
    return size_ + 1;
}

VectorXd CumTempProblem::accumAmt(const Phonon& before,
                                  const Phonon& after) const
{
    VectorXd vec;
    vec.setZero(size_ + 1);
    long index = (before.nscat() + step_ - 1) / step_;
    vec(index) = after.time() - before.time();
    return vec;
}

ArrayXXd CumTempProblem::postProc(const ArrayXXd& data) const
{
    ArrayXXd sol(data);
    for (long i = 0; i < size_; ++i)
    {
        sol.row(i + 1) += sol.row(i);
    }
    return sol / mat()->energySum();
}

CumFluxProblem::CumFluxProblem()
: FieldProblem()
{}

CumFluxProblem::CumFluxProblem(const Material* mat, const Domain* dom,
                               long nemit, long size,
                               long maxscat, long maxloop)
: FieldProblem(mat, dom, nemit, maxscat, maxloop), size_(size)
{
    step_ = (maxscat - 1) / size;
    if ((maxscat - 1) % size != 0) step_++;
}

std::string CumFluxProblem::info() const
{
    Eigen::IOFormat fmt(0, 0, " ", ";", "", "", "[", "]");
    std::ostringstream ss;
    ss << "CumFluxProblem " << static_cast<const Problem*>(this) << std::endl;
    ss << FieldProblem::info() << std::endl;
    ss << "  size:    " << size_;
    return ss.str();
}

long CumFluxProblem::rows() const
{
    return 3*(size_ + 1);
}

VectorXd CumFluxProblem::accumAmt(const Phonon& before,
                                  const Phonon& after) const
{
    VectorXd vec;
    vec.setZero(3*(size_ + 1));
    long index = 3*((before.nscat() + step_ - 1) / step_);
    vec.segment<3>(index) = after.pos() - before.pos();
    return vec;
}

ArrayXXd CumFluxProblem::postProc(const ArrayXXd& data) const
{
    ArrayXXd sol(data);
    long cols = data.cols();
    for (long i = 0; i < size_; ++i)
    {
        sol.block(3*(i + 1), 0, 3, cols) += sol.block(3*i, 0, 3, cols);
    }
    return sol;
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
