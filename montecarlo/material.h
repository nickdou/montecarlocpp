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
#include "random.h"
#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <string>

using Eigen::ArrayXd;
using Eigen::ArrayXXd;

class Material
{
private:
    class Dist
    {
    private:
        DiscreteDist wDist_;
        std::vector<DiscreteDist> pDist_;
        
    public:
        Dist();
        Dist(const ArrayXXd& pdf);
        
        Phonon::Prop drawProp(Rng& gen) const;
    };
    
    
    static const int nscat_ = 2;
    long np_, nw_;
    double T_, k_;
    ArrayXd omega_;
    ArrayXXd tau_, vel_;
    Dist energyDist_, fluxDist_, scatDist_;
    double energySum_, fluxSum_, scatSum_;
    std::string disp_, relax_;
    
    std::string info() const;
    
public:
    Material();
    Material(const std::string& disp, const std::string& relax, double temp);
    
    double temp() const;
    double cond() const;
    double tau(const Phonon& phn) const;
    double vel(const Phonon& phn) const;
    
    Phonon::Prop drawEnergyProp(Rng& gen) const;
    Phonon::Prop drawFluxProp(Rng& gen) const;
    Phonon::Prop drawScatProp(Rng& gen) const;
    double energySum() const;
    double fluxSum() const;
    double scatSum() const;
    
    void drawScatNext(Phonon& phn, Rng& gen) const;
    void scatter(Phonon& phn, Rng& gen) const;
    
    friend std::ostream& operator<<(std::ostream& os, const Material& mat);
};

#endif
