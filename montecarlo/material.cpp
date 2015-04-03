//
//  material.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "material.h"
#include "tools.h"
#include "random.h"
#include "constants.h"
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>

const int Material::nscat_;

Material::Material(const char* disp, const char* relax, double temp) : T_(temp)
{
    std::ostringstream ss;
    ss << "Material " << this << std::endl;
    ss << "  disp:  " << disp << std::endl;
    ss << "  relax: " << relax << std::endl;
    ss << "  temp:  " << temp;
    info_ = ss.str();
    
    // Read dispersion file
    std::ifstream dispFile(disp);
    BOOST_ASSERT_MSG(dispFile, "Error opening dispersion file");
    
    dispFile >> nw_ >> np_ >> std::ws;
    BOOST_ASSERT_MSG(nw_ > 0 && np_ > 0, "Invalid dispersion file");
    
    ArrayXXd dispData(nw_, 2+2*np_);
    extractArray(dispFile, dispData);
    
    omega_.resize(nw_);
    ArrayXXd domega(nw_, np_);
    vel_.resize(nw_, np_);
    ArrayXXd dos(nw_, np_);
    
    omega_ = dispData.col(0);
    for (long p = 0; p < np_; p++) {
        domega.col(p) = dispData.col(1);
        vel_.col(p) = dispData.col(2+2*p);
        dos.col(p) = dispData.col(3+2*p);
    }
    
    // Read relaxation time file
    std::ifstream relaxFile(relax);
    BOOST_ASSERT_MSG(relaxFile, "Error opening relaxation time file");
    
    Eigen::Array<double, 4*nscat_, Eigen::Dynamic> coeffs(4*nscat_, np_);
    extractArray(relaxFile, coeffs.transpose());
    
    // Calculate tau
    tau_.resize(nw_, np_);
    tau_.setZero();
    for (long p = 0; p < np_; p++) {
        ArrayXd tauinv(nw_);
        tauinv.setZero();
        for (int j = 0; j < nscat_; j++) {
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
    ArrayXXd dedT(nw_, np_);
    ArrayXd x = HBAR/(KB*T_) * omega_;
    for (long w = 0; w < nw_; w++) {
        double val = KB * (std::fabs(x(w)) < Dbl::epsilon() ?
                           1. - x(w)*x(w)/12. :
                           std::pow(x(w) / (2.*std::sinh(x(w)/2.)), 2));
        dedT.row(w).setConstant(val);
    }
    
    // Calculate distributions
    ArrayXXd energyPdf = dedT * dos * domega;
    energyDist_ = Dist(energyPdf);
    energySum_ = energyPdf.sum();
    
    ArrayXXd fluxPdf = vel_ * energyPdf;
    fluxDist_ = Dist(fluxPdf);
    fluxSum_ = fluxPdf.sum();
    
    ArrayXXd scatPdf = energyPdf / tau_;
    scatDist_ = Dist(scatPdf);
    scatSum_ = scatPdf.sum();
    
    // Calculate k
    k_ = (tau_ * vel_.pow(2) * energyPdf).sum()/3.;
}

Material::Dist::Dist(const ArrayXXd& pdf) {
    long nw = pdf.rows();
    long np = pdf.cols();
    
    ArrayXd rowSum = pdf.rowwise().sum();
    double* beg = rowSum.data();
    double* end = beg + nw;
    wDist = DiscreteDist(beg, end);
    
    pDist.reserve(nw);
    for (long w = 0; w < nw; w++) {
        ArrayXd row = pdf.row(w);
        beg = row.data();
        end = beg + np;
        pDist.push_back( DiscreteDist(beg, end) );
    }
}

