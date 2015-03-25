//
//  problem.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#include "problem.h"
#include "domain.h"
#include "boundary.h"
#include "phonon.h"
#include "tools.h"
#include "random.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <string>

std::string dispBdry(const Subdomain::BdryPtrs& cont, const Boundary* bdry) {
    if (!bdry) return "middle";
    switch (std::find(cont.begin(), cont.end(), bdry) - cont.begin()) {
        case  0: return "back  ";
        case  1: return "left  ";
        case  2: return "bottom";
        case  3: return "front ";
        case  4: return "right ";
        case  5: return "top   ";
        case  6: return "NOT FOUND";
    }
    return "ERROR";
}

Trajectory TrajProblem::solve(Rng& gen, Progress* prog) const {
    TrkPhonon phn;
    const Subdomain* sdom = 0;
    const Boundary* bdry = 0;
    
    Phonon::Prop prop = (prop_ ? *prop_ : mat_->drawScatProp(gen));
    if (pos_) {
        Eigen::Vector3d pos = *pos_;
        Eigen::Vector3d dir = (dir_ ? *dir_ : drawIso(gen));
        phn = TrkPhonon(prop, pos, dir, true);
        sdom = dom_->locate(*pos_);
        bdry = 0;
    } else {
        UniformIntDist dist(0, dom_->emitPtrs().size() - 1);
        const Emitter* e = dom_->emitPtrs().at(dist(gen));
        phn = e->emit(prop, gen);
        sdom = e->emitSdom();
        bdry = e->emitBdry();
    }
    
    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
    double scatDist = mat_->drawScatDist(phn.prop(), gen);
    for (long i = 0; i < maxloop; i++) {
        std::cout << dispBdry(sdom->bdryPtrs(), bdry) << " --> ";
        bdry = sdom->advect(phn, scatDist);
        std::cout << dispBdry(sdom->bdryPtrs(), bdry) << std::endl;
        
        long nscatMat = 0;
        if (bdry) {
            bdry = bdry->scatter(phn, gen);
            sdom = bdry->sdom();
        } else {
            scatDist = mat_->scatter(phn, gen);
            nscatMat++;
        }
        if (!phn.alive() || phn.nscat() >= maxscat_) break;
        if (prog) prog->increment();
    }
    return Trajectory(this, phn.traj());
}

struct AccumWeightF {
    double operator()(double weight, const Emitter* emit) const {
        return weight + emit->emitWeight();
    }
};

struct CalcEmitF {
    double totWeight_;
    long nemit_;
    CalcEmitF(double totWeight, long nemit)
    : totWeight_(totWeight), nemit_(nemit) {}
    long operator()(const Emitter* emit) const {
        double frac = emit->emitWeight() / totWeight_;
        double rounded = std::ceil(frac * nemit_ - 0.5);
        return std::max(1l, static_cast<long>(rounded));
    }
};

template<typename T, typename M, typename Enable = void>
struct MultiplyF {
    void operator()(T& element, const M& multiplier) const {
        element.array() *= multiplier.array();
    }
};

template<typename T, typename M>
struct MultiplyF<T, M,
typename boost::enable_if< boost::is_scalar<M> >::type>
{
    void operator()(T& element, M multiplier) const {
        element *= multiplier;
    }
};

template<typename T, typename M>
struct CalcFieldF {
    M factor_;
    CalcFieldF(const M& factor) : factor_(factor) {}
    void operator()(typename Field<T>::value_type& pair) const {
        const Subdomain* sdom = pair.first;
        Data<T>& data = pair.second;
        Collection coll(0,0,0);
        Eigen::Map<Collection::Vector3s> index(coll.data());
        MultiplyF<T, M> multiply;
        typedef typename Data<T>::Size Size;
        for (Size k = 0; k < data.shape()[2]; ++k) {
            coll[2] = k;
            for (Size j = 0; j < data.shape()[1]; ++j) {
                coll[1] = j;
                for (Size i = 0; i < data.shape()[0]; ++i) {
                    coll[0] = i;
                    multiply(data(coll),
                             factor_ / sdom->cellVol( index.cast<long>() ));
                }
            }
        }
    }
};

struct State {
    double time;
    Eigen::Vector3d pos;
    State(double t, const Eigen::Vector3d& x) : time(t), pos(x) {}
};

template<typename T, typename F, typename M>
Field<T> Problem::solveField(const F& functor, M factor,
                             Rng& gen, Progress* prog) const
{
    Field<T> fld = initField<T>();
    
    const Domain::EmitPtrs& emitPtrs = dom_->emitPtrs();
    double totWeight = std::accumulate(emitPtrs.begin(), emitPtrs.end(),
                                       0., AccumWeightF());
    std::vector<long> emitPdf( emitPtrs.size() );
    std::transform(emitPtrs.begin(), emitPtrs.end(),
                   emitPdf.begin(), CalcEmitF(totWeight, nemit_));
    std::vector<long> emitCdf( emitPtrs.size() );
    std::partial_sum(emitPdf.begin(), emitPdf.end(), emitCdf.begin());
    long nemit = emitCdf.back();
    double power = totWeight / nemit * mat_->fluxSum() / 4.;
    
    long maxloop = (maxloop_ != 0 ? maxloop_ : loopFactor_ * maxscat_);
    #pragma omp for schedule(static)
    for (long n = 0; n < nemit; ++n) {
        std::vector<long>::size_type eIndex;
        eIndex = (std::upper_bound(emitCdf.begin(), emitCdf.end(), n) -
                  emitCdf.begin());
        
        const Emitter* e = emitPtrs.at(eIndex);
        Phonon phn = e->emit(mat_->drawFluxProp(gen), gen);
        const Subdomain* sdom = e->emitSdom();
        const Boundary* bdry = e->emitBdry();
        
        double time = 0.;
        double scatDist = mat_->drawScatDist(phn.prop(), gen);
        for (long i = 0; i < maxloop; i++) {
            State before(time, phn.pos());
//            std::cout << dispBdry(sdom->bdryPtrs(), bdry) << " --> ";
            bdry = sdom->advect(phn, scatDist);
//            std::cout << dispBdry(sdom->bdryPtrs(), bdry) << std::endl;
            
            time += (phn.pos() - before.pos).norm() / mat_->vel(phn.prop());
            State after(time, phn.pos());
            sdom->accumulate<T>(before.pos, after.pos, fld[sdom],
                                phn.sign() * functor(before, after));
            
            long nscatMat = 0;
            if (bdry) {
                bdry = bdry->scatter(phn, gen);
                sdom = bdry->sdom();
            } else {
                scatDist = mat_->scatter(phn, gen);
                nscatMat++;
            }
            
            if (!phn.alive() || phn.nscat() >= maxscat_) break;
        }
        
        if (prog) {
            #pragma omp critical
            {
                prog->increment();
            }
        }
    }
    std::for_each(fld.begin(), fld.end(), CalcFieldF<T, M>(power * factor));
    return fld;
}

struct TempProblem::TempAccumF {
    double operator()(const State& before, const State& after) const {
        return after.time - before.time;
    }
};

template Field<double>
Problem::solveField(const TempProblem::TempAccumF& functor,
                    double factor, Rng& gen, Progress* prog) const;

Temperature TempProblem::solve(Rng& gen, Progress* prog) const {
    TempAccumF functor;
    double factor = 1./mat_->energySum();
    return Temperature(this, solveField<double>(functor, factor, gen, prog));
}

struct FluxProblem::FluxAccumF {
    Eigen::Vector3d dir_;
    FluxAccumF(const Eigen::Vector3d& dir) : dir_(dir) {}
    double operator()(const State& before, const State& after) const {
        return (after.pos - before.pos).dot(dir_);
    }
};

template Field<double>
Problem::solveField(const FluxProblem::FluxAccumF& functor,
                    double factor, Rng& gen, Progress* prog) const;

Flux FluxProblem::solve(Rng& gen, Progress* prog) const {
    FluxAccumF functor(dir_);
    double factor = 1.;
    return Flux(this, solveField<double>(functor, factor, gen, prog));
}

struct MultiProblem::MultiAccumF {
    Eigen::Vector4d operator()(const State& before, const State& after) const {
        double dtime = after.time - before.time;
        Eigen::Vector3d dpos = after.pos - before.pos;
        return (Eigen::Vector4d() << dtime, dpos).finished();
    }
};

template Field<Eigen::Vector4d>
Problem::solveField(const MultiProblem::MultiAccumF& functor,
                    Eigen::Vector4d factor, Rng& gen, Progress* prog) const;

MultiSolution MultiProblem::solve(Rng& gen, Progress* prog) const {
    typedef Eigen::Vector4d Vector4d;
    MultiAccumF functor;
    Vector4d factor = Vector4d::Ones();
    factor(0) = 1./mat_->energySum();
    Field<Vector4d> fld = solveField<Vector4d>(functor, factor, gen, prog);
    if (!inv_.isApprox(Eigen::Matrix3d::Identity())) {
        for (Field<Vector4d>::iterator it = fld.begin(); it != fld.end(); ++it){
            Data<Vector4d>& data = it->second;
            for (Data<Vector4d>::Size i = 0; i < data.num_elements(); ++i) {
                Vector4d& element = *(data.data() + i);
                element.tail<3>() = inv_ * element.tail<3>().matrix();
            }
        }
    }
    return MultiSolution(this, fld);
}
