//
//  problem.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#include "problem.h"
#include "domain.h"
#include "grid.h"
#include "boundary.h"
#include "phonon.h"
#include "tools.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace lambda = boost::lambda;

template<typename C, typename E>
long find(const C& cont, E elem) {
//    unsigned long index;
    long index = std::find(cont.begin(), cont.end(), elem) - cont.begin();
    return (index == cont.size() ? -1l : index);
}

TrkPhonon::Trajectory TrajProblem::solve(Rng& gen, Progress* prog) const {
    TrkPhonon phn;
    const Subdomain* sdom = 0;
    const Boundary* bdry = 0;
    
    Phonon::Prop prop = (prop_ ? *prop_ : Base::mat_->drawScatProp(gen));
    if (pos_) {
        Eigen::Vector3d pos = *pos_;
        Eigen::Vector3d dir = (dir_ ? *dir_ : drawIso(gen));
        phn = TrkPhonon(prop, pos, dir, true);
        sdom = Base::dom_->locate(*pos_);
        bdry = 0;
        BOOST_ASSERT_MSG(sdom, "Position not inside domain");
    } else {
        UniformIntDist dist(0, Base::dom_->emitPtrs().size() - 1);
        const Emitter* e = Base::dom_->emitPtrs().at(dist(gen));
        phn = e->emit(prop, gen);
        sdom = e->emitSdom();
        bdry = e->emitBdry();
    }
    
    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
    double scatDist = Base::mat_->drawScatDist(phn.prop(), gen);
    long nscatMat = 0;
    for (long i = 0; i < maxloop; i++) {
        std::cout << std::setw(2) << find(Base::dom_->sdomPtrs(), sdom) << ": ";
        std::cout << std::setw(2) << find(sdom->bdryPtrs(), bdry) << " --> ";
        bdry = sdom->advect(phn, scatDist);
        std::cout << std::setw(2) << find(sdom->bdryPtrs(), bdry) << std::endl;
        
        if (bdry) {
            bdry = bdry->scatter(phn, gen);
            sdom = bdry->sdom();
        } else {
            scatDist = Base::mat_->scatter(phn, gen);
            nscatMat++;
        }
        if (!phn.alive() || phn.nscat() >= maxscat_) break;
        if (prog) prog->incrCount();
    }
    
    return phn.trajectory();
}

struct AccumWeightF {
    double operator()(double weight, const Emitter* emit) const {
        return weight + emit->emitWeight();
    }
};

struct CalcEmitF {
    double totWeight;
    long nemit;
    CalcEmitF(double w, long n) : totWeight(w), nemit(n) {}
    long operator()(const Emitter* emit) const {
        double frac = emit->emitWeight() / totWeight;
        double rounded = std::ceil(frac * nemit - 0.5);
        return std::max(1l, static_cast<long>(rounded));
    }
};

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

template<typename Derived>
struct CalcFieldF {
    typedef typename Derived::Type Type;
    typedef typename Derived::Factor Factor;
    typedef typename Derived::AccumF AccumF;
    AccumF functor;
    Factor factor;
    CalcFieldF(const AccumF& fun, const Factor& pow)
    : functor(fun), factor(pow) {}
    void operator()(typename Field<Type>::value_type& pair) const {
        const Subdomain* sdom = pair.first;
        Data<Type>& data = pair.second;
        Collection coll(0,0,0);
        Eigen::Map<Collection::Vector3s> index(coll.data());
        typedef typename Data<Type>::Size Size;
        for (Size k = 0; k < data.shape()[2]; ++k) {
            coll[2] = k;
            for (Size j = 0; j < data.shape()[1]; ++j) {
                coll[1] = j;
                for (Size i = 0; i < data.shape()[0]; ++i) {
                    coll[0] = i;
                    data(coll) = factor / sdom->cellVol( index.cast<long>() ) *
                                 functor( data(coll) );
                }
            }
        }
    }
};

struct State {
    double time;
    Phonon phn;
    long nscatMat;
    State(double t, const Phonon& p, long n)
    : time(t), phn(p), nscatMat(n) {}
};

template<typename Derived>
Field<typename FieldProblem<Derived>::Type>
FieldProblem<Derived>::solveField(const AccumF& fun, const Factor& fac,
                                  Rng& gen, Progress* prog) const
{
    Field<Type> fld = initField();
    
    const Domain::EmitPtrs& emitPtrs = Base::dom_->emitPtrs();
    double totWeight = std::accumulate(emitPtrs.begin(), emitPtrs.end(),
                                       0., AccumWeightF());
    std::vector<long> emitPdf( emitPtrs.size() );
    std::transform(emitPtrs.begin(), emitPtrs.end(),
                   emitPdf.begin(), CalcEmitF(totWeight, nemit_));
    std::vector<long> emitCdf( emitPtrs.size() );
    std::partial_sum(emitPdf.begin(), emitPdf.end(), emitCdf.begin());
    long nemit = emitCdf.back();
    double power = totWeight / nemit * Base::mat_->fluxSum() / 4.;
    
    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
#pragma omp for schedule(static)
    for (long n = 0; n < nemit; ++n) {
        std::vector<long>::size_type eIndex;
        eIndex = (std::upper_bound(emitCdf.begin(), emitCdf.end(), n) -
                  emitCdf.begin());
        
        const Emitter* e = emitPtrs.at(eIndex);
        const Subdomain* sdom = e->emitSdom();
        const Boundary* bdry = e->emitBdry();
#ifdef DEBUG
        TrkPhonon phn = e->emit(Base::mat_->drawFluxProp(gen), gen);
#else
        Phonon phn = e->emit(Base::mat_->drawFluxProp(gen), gen);
#endif
        double time = 0.;
        long nscatMat = 0;
        double scatDist = Base::mat_->drawScatDist(phn.prop(), gen);
        for (long i = 0; i < maxloop; i++) {
            State before(time, phn, nscatMat);
            bdry = sdom->advect(phn, scatDist);
            
            time += ((phn.pos() - before.phn.pos()).norm() /
                     Base::mat_->vel(phn.prop()));
            State after(time, phn, nscatMat);
            
//            try {
//                sdom->accumulate<Type>(before.phn.pos(),
//                                       after.phn.pos(),
//                                       fld[sdom],
//                                       phn.sign() * fun(before, after));
//            } catch(const Grid::OutOfRange& e) {
//#ifdef DEBUG
//                std::cerr << e.what() << std::endl;
//                std::cerr << "Trajectory" << std::endl;
//                std::cerr << phn.trajectory() << std::endl;
//#endif
//            }
            bool status = sdom->accumulate<Type>(before.phn.pos(),
                                                 after.phn.pos(),
                                                 fld[sdom],
                                                 phn.sign()*fun(before, after));
            if (!status) {
                if (prog) {
#pragma omp critical
                    {
                        prog->incrEsc();
                    }
                }
                break;
            }
            
            if (bdry) {
                bdry = bdry->scatter(phn, gen);
                sdom = bdry->sdom();
            } else {
                scatDist = Base::mat_->scatter(phn, gen);
                nscatMat++;
            }
            
            if (!phn.alive() || phn.nscat() >= maxscat_) break;
        }
        
        if (prog) {
#pragma omp critical
            {
                prog->incrCount();
            }
        }
    }
    
    std::for_each(fld.begin(), fld.end(),
                  CalcFieldF<Derived>(fun, fac*power));
    return fld;
}

struct TempAccumF {
    double operator()(const State& before, const State& after) const {
        return after.time - before.time;
    }
    double operator()(double elem) const {
        return elem;
    }
};

Field<double> TempProblem::solve(Rng& gen, Progress* prog) const {
    TempAccumF functor;
    double factor = 1./Base::Base::mat_->energySum();
    return Base::solveField(functor, factor, gen, prog);
}

struct FluxAccumF {
    Eigen::Vector3d dir;
    FluxAccumF(const Eigen::Vector3d& d) : dir(d) {}
    double operator()(const State& before, const State& after) const {
        return (after.phn.pos() - before.phn.pos()).dot(dir);
    }
    double operator()(double elem) const {
        return elem;
    }
};

Field<double> FluxProblem::solve(Rng& gen, Progress* prog) const {
    FluxAccumF functor(dir_);
    double factor = 1.;
    return Base::solveField(functor, factor, gen, prog);
}

struct MultiAccumF {
    Eigen::Array4d operator()(const State& before, const State& after) const {
        double dtime = after.time - before.time;
        Eigen::Vector3d dpos = after.phn.pos() - before.phn.pos();
        return (Eigen::Array4d() << dtime, dpos).finished();
    }
    Eigen::Array4d operator()(const Eigen::Array4d& elem) const {
        return elem;
    }
};

Field<Eigen::Array4d> MultiProblem::solve(Rng& gen, Progress* prog) const {
    typedef Eigen::Array4d Array4d;
    MultiAccumF functor;
    Array4d factor(1./mat_->energySum(), 1., 1., 1.);
    Field<Array4d> fld = Base::solveField(functor, factor, gen, prog);
    
    if (!inv_.isApprox(Eigen::Matrix3d::Identity())) {
        for (Field<Array4d>::iterator it = fld.begin(); it != fld.end(); ++it){
            Data<Array4d>& data = it->second;
            for (Data<Array4d>::Size i = 0; i < data.num_elements(); ++i) {
                Array4d& element = *(data.data() + i);
                element.tail<3>() = inv_ * element.tail<3>().matrix();
            }
        }
    }
    return fld;
}

template<typename T, typename F>
struct CumF {
    typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayXT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;
    long step, size;
    F functor;
    CumF(long st, long sz, const F& fun) : step(st), size(sz), functor(fun) {}
    ArrayXT operator()(const State& before, const State& after) const {
        BOOST_ASSERT_MSG(before.phn.nscat() == after.phn.nscat(),
                         "Scattering occured during advection step");
//        long index = before.phn.nscat() / step;
        long index = (before.phn.nscat() + step - 1) / step;
        BOOST_ASSERT_MSG(index <= size, "Index out of bounds");
        return functor(before, after) * VectorXT::Unit(size + 1, index);
    }
    ArrayXT operator()(const ArrayXT& elem) const {
        ArrayXT cumsum(elem);
        for (long i = 1; i <= size; ++i) {
            cumsum(i) += cumsum(i-1);
        }
        return cumsum;
    }
};

struct CumTempAccumF : public CumF<double, TempAccumF> {
    CumTempAccumF(long step, long size, const TempAccumF& functor)
    : CumF<double, TempAccumF>(step, size, functor) {}
};

Field<Eigen::ArrayXd> CumTempProblem::solve(Rng& gen, Progress* prog) const {
    CumTempAccumF functor(step_, size_, TempAccumF());
    double factor = 1./Base::Base::mat_->energySum();
    return Base::solveField(functor, factor, gen, prog);
}

struct CumFluxAccumF : public CumF<double, FluxAccumF> {
    CumFluxAccumF(long step, long size, const FluxAccumF& functor)
    : CumF<double, FluxAccumF>(step, size, functor) {}
};

Field<Eigen::ArrayXd> CumFluxProblem::solve(Rng& gen, Progress* prog) const {
    CumFluxAccumF functor(step_, size_, FluxAccumF(dir_));
    double factor = 1.;
    return Base::solveField(functor, factor, gen, prog);
}


