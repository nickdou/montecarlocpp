//
//  material.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#ifndef __montecarlocpp__material__
#define __montecarlocpp__material__

#include "phonon.h"
#include "tools.h"
#include "random.h"
#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <string>

class Material {
private:
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::ArrayXXd ArrayXXd;
    class Dist {
        DiscreteDist wDist;
        std::vector<DiscreteDist> pDist;
    public:
        Dist() {}
        Dist(const ArrayXXd& pdf);
        Phonon::Prop drawProp(Rng& gen) const {
            long w = wDist(gen);
            long p = pDist.at(w)(gen);
            return Phonon::Prop(w, p);
        }
    };
    long np_, nw_;
    static const int nscat_ = 2;
    double T_, k_;
    ArrayXd omega_;
    ArrayXXd tau_, vel_;
    Dist energyDist_, fluxDist_, scatDist_;
    double energySum_, fluxSum_, scatSum_;
    std::string info_;
public:
    Material() {}
    Material(const char* disp, const char* relax, double temp);
    double temp() const { return T_; }
    double cond() const { return k_; }
    double vel(const Phonon::Prop& prop) const {
        return vel_(prop.w(), prop.p());
    }
    double energySum() const { return energySum_; }
    double fluxSum() const { return fluxSum_; }
    double scatSum() const { return scatSum_; }
    Phonon::Prop drawEnergyProp(Rng& gen) const {
        return energyDist_.drawProp(gen);
    }
    Phonon::Prop drawFluxProp(Rng& gen) const {
        return fluxDist_.drawProp(gen);
    }
    Phonon::Prop drawScatProp(Rng& gen) const {
        return scatDist_.drawProp(gen);
    }
    double drawScatDist(const Phonon::Prop& prop, Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        double r = dist(gen);
        return vel(prop) * tau_(prop.w(), prop.p()) * -std::log(1. - r);
    }
    double scatter(Phonon& phn, Rng& gen) const {
        phn.prop(drawScatProp(gen));
        phn.dir<true>(drawIso(gen));
        return drawScatDist(phn.prop(), gen);
    }
    friend std::ostream& operator<<(std::ostream& os, const Material& mat) {
        return os << mat.info_;
    }
};

#endif
