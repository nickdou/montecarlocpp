//
//  material.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "material.h"
#include "phonon.h"
#include "random.h"
#include "constants.h"
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

Material::Dist::Dist()
{}

Material::Dist::Dist(const Eigen::ArrayXXd& pdf)
{
    long nw = pdf.rows();
    long np = pdf.cols();
    
    Eigen::ArrayXd rowSum = pdf.rowwise().sum();
    double* beg = rowSum.data();
    double* end = beg + nw;
    wDist_ = DiscreteDist(beg, end);
    
    pDist_.reserve(nw);
    for (long w = 0; w < nw; w++)
    {
        Eigen::ArrayXd row = pdf.row(w);
        beg = row.data();
        end = beg + np;
        pDist_.push_back( DiscreteDist(beg, end) );
    }
}

Phonon::Prop Material::Dist::drawProp(Rng& gen) const {
    long w = wDist_(gen);
    long p = pDist_.at(w)(gen);
    return Phonon::Prop(w, p);
}

template<typename Derived>
void extractArray(std::istream& is, const Eigen::DenseBase<Derived>& data)
{
    typedef typename Derived::Scalar Scalar;
    typedef typename Derived::Index Index;
    
    is >> std::ws;
    std::string line;
    Index i = 0;
    while (i < data.rows() && std::getline(is, line))
    {
        std::stringstream ss(line);
        for (Index j = 0; j < data.cols(); ++j)
        {
            Scalar element;
            ss >> element;
            const_cast<Scalar&>(data(i, j)) = element;
            BOOST_ASSERT_MSG(ss, "Array extraction failed");
        }
        i++;
    }
    BOOST_ASSERT_MSG(i == data.rows(), "Array extraction failed");
}

const int Material::nscat_;

Material::Material()
{}

Material::Material(const std::string& disp, const std::string& relax,
                   double temp)
: T_(temp)
{
    std::ostringstream ss;
    ss << "Material " << this << std::endl;
    ss << "  disp:  " << disp << std::endl;
    ss << "  relax: " << relax << std::endl;
    ss << "  temp:  " << temp;
    info_ = ss.str();
    
    // Read dispersion file
    std::ifstream dispFile( disp.c_str() );
    BOOST_ASSERT_MSG(dispFile, "Error opening dispersion file");
    
    dispFile >> nw_ >> np_ >> std::ws;
    BOOST_ASSERT_MSG(nw_ > 0 && np_ > 0, "Invalid dispersion file");
    
    Eigen::ArrayXXd dispData(nw_, 2+2*np_);
    extractArray(dispFile, dispData);
    
    omega_.resize(nw_);
    Eigen::ArrayXXd domega(nw_, np_);
    vel_.resize(nw_, np_);
    Eigen::ArrayXXd dos(nw_, np_);
    
    omega_ = dispData.col(0);
    for (long p = 0; p < np_; p++)
    {
        domega.col(p) = dispData.col(1);
        vel_.col(p) = dispData.col(2+2*p);
        dos.col(p) = dispData.col(3+2*p);
    }
    
    // Read relaxation time file
    std::ifstream relaxFile( relax.c_str() );
    BOOST_ASSERT_MSG(relaxFile, "Error opening relaxation time file");
    
    Eigen::Array<double, 4*nscat_, Eigen::Dynamic> coeffs(4*nscat_, np_);
    extractArray(relaxFile, coeffs.transpose());
    
    // Calculate tau
    tau_.resize(nw_, np_);
    tau_.setZero();
    for (long p = 0; p < np_; p++)
    {
        Eigen::ArrayXd tauinv(nw_);
        tauinv.setZero();
        for (int j = 0; j < nscat_; j++)
        {
            Eigen::Array4d c = coeffs.block<4, 1>(4*j, p);
            BOOST_ASSERT_MSG(c(0) >= 0., "Scattering times cannot be negative");
            if (c(0) <= Dbl::min()) continue;
            tauinv += (c(0) * omega_.pow(c(1)) *
                       std::pow(T_, c(2)) * std::exp(-c(3)/T_));
        }
        tau_.col(p) = tauinv.inverse();
    }
    BOOST_ASSERT_MSG(tau_.allFinite(),
                     "Scattering time model produced infinite values");
    
    // Calculate dedT
    Eigen::ArrayXXd dedT(nw_, np_);
    Eigen::ArrayXd x = HBAR/(KB*T_) * omega_;
    for (long w = 0; w < nw_; w++)
    {
        double val = KB * (std::fabs(x(w)) < Dbl::epsilon() ?
                           1. - x(w)*x(w)/12. :
                           std::pow(x(w) / (2.*std::sinh(x(w)/2.)), 2));
        dedT.row(w).setConstant(val);
    }
    
    // Calculate distributions
    Eigen::ArrayXXd energyPdf = dedT * dos * domega;
    energyDist_ = Dist(energyPdf);
    energySum_ = energyPdf.sum();
    
    Eigen::ArrayXXd fluxPdf = vel_ * energyPdf;
    fluxDist_ = Dist(fluxPdf);
    fluxSum_ = fluxPdf.sum();
    
    Eigen::ArrayXXd scatPdf = energyPdf / tau_;
    scatDist_ = Dist(scatPdf);
    scatSum_ = scatPdf.sum();
    
    // Calculate k
    k_ = (tau_ * vel_.pow(2) * energyPdf).sum()/3.;
}

double Material::temp() const
{
    return T_;
}

double Material::cond() const
{
    return k_;
}

double Material::tau(const Phonon& phn) const
{
    Phonon::Prop prop = phn.prop();
    return tau_(prop.w(), prop.p());
}

double Material::vel(const Phonon& phn) const
{
    Phonon::Prop prop = phn.prop();
    return vel_(prop.w(), prop.p());
}

double Material::energySum() const
{
    return energySum_;
}

double Material::fluxSum() const
{
    return fluxSum_;
}
double Material::scatSum() const
{
    return scatSum_;
}

Phonon::Prop Material::drawEnergyProp(Rng& gen) const
{
    return energyDist_.drawProp(gen);
}

Phonon::Prop Material::drawFluxProp(Rng& gen) const
{
    return fluxDist_.drawProp(gen);
}

Phonon::Prop Material::drawScatProp(Rng& gen) const
{
    return scatDist_.drawProp(gen);
}

void Material::drawScatNext(Phonon& phn, Rng& gen) const
{
    static UniformDist01 dist; // [0, 1)
    double r = dist(gen);
    phn.scatNext(vel(phn) * tau(phn) * -std::log(1. - r));
}

void Material::scatter(Phonon& phn, Rng& gen) const
{
    phn.prop(drawScatProp(gen));
    phn.dir(drawIso(gen), true);
    drawScatNext(phn, gen);
}

std::ostream& operator<<(std::ostream& os, const Material& mat) {
    return os << mat.info_;
}


