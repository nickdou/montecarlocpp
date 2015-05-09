//
//  main.cpp
//  montecarlo
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "problem.h"
#include "domain.h"
#include "data.h"
#include "material.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

typedef Dev::result_type Seed;

Seed getSeed() {
    static Dev urandom;
    Seed s = urandom();
#pragma omp critical
    {
        std::cout << s << ' ';
    }
#pragma omp barrier
#pragma omp single
    {
        std::cout << std::endl;
    }
    return s;
}

TrajProblem::Solution solveTraj(const TrajProblem& prob) {
    std::cout << "Seed" << std::endl;
    Rng gen(getSeed());
    return prob.solve(gen);
}

void checkAlignedDomain(const Domain* dom) {
    Material mat("input/grey_disp.txt", "input/grey_relax2.txt", 300.);
    Eigen::Vector3d pos, dir;
    TrajProblem prob;
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers = dom->centers();
    for (int j = 0; j < centers.cols(); ++j) {
        pos = centers.col(j);
        for (int i = 0; i < 6; ++i) {
            dir = (i < 3 ? -1 : 1) * Eigen::Vector3d::Unit(i % 3);
            Eigen::IOFormat fmt(0, 0, " ", "", "", "", "[", "]");
            std::cout << std::endl;
            std::cout << "pos " << pos.format(fmt) << std::endl;
            std::cout << "dir " << dir.format(fmt) << std::endl;
            prob = TrajProblem(&mat, dom, pos, dir, 2l, 2l);
            std::cout << solveTraj(prob) << std::endl;
        }
    }
}

template<typename Derived>
typename Derived::Solution solveField(const FieldProblem<Derived>& prob,
                                      const Clock& clk)
{
    typedef typename Derived::Solution Solution;
    Solution result = prob.initField();
    Progress prog = prob.initProgress();
    prog.clock(clk);
    std::cout << "Seeds" << std::endl;
#pragma omp parallel
    {
        Rng gen(getSeed());
        Solution sol = prob.solve(gen, &prog);
#pragma omp critical
        {
            result += sol;
        }
    }
    return result;
}

template<int N, typename Derived>
Statistics<typename Derived::Solution>
solveFieldN(const FieldProblem<Derived>& prob, const Clock& clk)
{
    typedef typename Derived::Solution Solution;
    std::vector<Solution> vec;
    for (int i = 0; i < N; ++i) {
        std::cout << "Solution " << i << std::endl;
        Solution sol = solveField(prob, clk);
        std::cout << sol << std::endl;
        vec.push_back(sol);
    }
    
    Statistics<Solution> stats(prob.initField());
    std::for_each(vec.begin(), vec.end(), stats.accumulator());
    return stats;
}

template<int N, typename Derived, typename F>
Statistics<typename F::Type>
solveFieldN(const FieldProblem<Derived>& prob, const Clock& clk, const F& fun)
{
    typedef typename Derived::Solution Solution;
    typedef typename F::Type Type;
    std::vector<Type> vec;
    for (int i = 0; i < N; ++i) {
        std::cout << "Solution " << i << std::endl;
        Solution sol = solveField(prob, clk);
        std::cout << sol << std::endl;
        std::cout << "Output " << i << std::endl;
        Type out = fun(sol);
        std::cout << out << std::endl <<std::endl;
        vec.push_back(out);
    }
    
    Statistics<Type> stats(prob.initElem());
    std::for_each(vec.begin(), vec.end(), stats.accumulator());
    return stats;
}

template<typename Derived>
struct AverageEndsF {
    typedef typename Derived::Type Type;
    static const int nsdom = 7;
    static const int itop = 26;
    const OctetDomain* oct;
    Eigen::VectorXd weight;
    Type zero;
    AverageEndsF(const OctetDomain& dom,
                 const Eigen::Matrix<double, 5, 1>& dim,
                 const Type& z)
    : oct(&dom), weight(nsdom), zero(z)
    {
        weight << dim[3], dim[4], dim[2],
                  dim[4], dim[2] - dim[4], dim[4], dim[3];
    }
    Type operator()(const Field<Type>& fld) const {
        Type fldSum = zero;
        for (int b = 0; b < 2; ++b) {
            int begin = (b == 0 ? 0 : itop);
            for (int s = 0; s < nsdom; ++s) {
                typedef typename Field<Type>::const_iterator Iter;
                Iter it = fld.find(oct->sdomPtrs()[begin+s]);
                BOOST_ASSERT_MSG(it != fld.end(), "Subdomain not found");
                const Data<Type>& data = it->second;
                Collection shape = data.shapeColl();
                Type sdomSum = zero;
                for (typename Data<Type>::Size i = 0; i < shape[0]; ++i) {
                    for (typename Data<Type>::Size j = 0; j < shape[1]; ++j) {
                        if (s == 0) {
                            sdomSum += data[i][j][0];
                        } else {
                            sdomSum += data[i][j][shape[2]-1];
                        }
                    }
                }
                fldSum += weight(s) * sdomSum / (shape[0] * shape[1]);
            }
        }
        return fldSum / (2*weight.sum());
    }
};

int main(int argc, const char * argv[]) {
    std::stringstream ss;
    for (int i = 1; i < argc; ++i) {
        ss << argv[i] << ' ';
    }
    
    Clock clk;
    std::string time = clk.timestamp();
    
//    std::ofstream ofmain;
//    std::string filename;
//    ss >> filename;
//    if (filename == "cout") {
//        osmain = &std::cout;
//    } else {
//        filename = "output/" + filename;
//        ofmain.open(filename.c_str(), std::ios::out | std::ios::trunc);
//        ofmain.open(filename.c_str(), std::ios::out | std::ios::app);
//        if (!ofmain.is_open()) return 1;
//        std::cout << time << std::endl;
//        std::cout << "Output file: " << filename << std::endl;
//        osmain = &ofmain;
//    }
    
    std::cout << time << std::endl;
#ifdef DEBUG
    std::cout << "DEBUG" << std::endl;
#endif
    
//    Material mat("input/grey_disp.txt", "input/grey_relax2.txt", 300.);
    Material mat("input/Si_disp.txt", "input/Si_relax2.txt", 300.);
    
    std::cout << mat << std::endl;
    
//    Eigen::Vector3d dim(2e-6, 2e-6, 2e-6);
//    Eigen::Matrix<long, 3, 1> div(0, 0, 0);
//    BulkDomain dom(dim, div, 2.);
    
//    Eigen::Vector3d dim(2e-6, 2e-8, 2e-6);
//    Eigen::Matrix<long, 3, 1> div(0, 10, 0);
//    FilmDomain dom(dim, div, 2.);
    
//    Eigen::Matrix<double, 5, 1> dim;
//    Eigen::Matrix<long, 5, 1> div;
//    dim << 2e-8, 2e-8, 2e-8, 2e-8, 2e-8;
//    div << 10, 10, 10, 10, 0;
//    TeeDomain dom(dim, div, 3.);
    
//    Eigen::Matrix<double, 4, 1> dim;
//    Eigen::Matrix<long, 4, 1> div;
//    dim << 2e-6, 2e-8, 2e-8, 2e-8;
//    div << 0, 10, 10, 10;
//    TubeDomain dom(dim, div, 2.);
    
//    Eigen::Vector3d flux = Eigen::Vector3d::UnitX();
    
    double cell, beam;
    Eigen::Matrix<double, 5, 1> dim;
    Eigen::Matrix<long, 5, 1> div;
    double deltaT;
//    dim << 2e-6, 2e-6, 8e-7, 2e-7, 4e-8;
//    div << 4, 4, 0, 0, 0;
//    deltaT = 5.68;
    ss >> cell >> dim(2) >> dim(3) >> dim(4);
    beam = cell/4. * std::sqrt(2.);
    dim(0) = beam - dim(2) - dim(4);
    dim(1) = beam - dim(3) - dim(4);
    BOOST_ASSERT_MSG((dim.array() > 0.).all(), "Dimensions must be positive");
    BOOST_ASSERT_MSG(dim(2) > dim(4), "Major axis smaller than wall thickness");
    
    ss >> div(0) >> div(2) >> div(3) >> div(4);
    div(1) = div(0);
    
    deltaT = beam * 2e6;
    
    OctetDomain dom(dim, div, deltaT);
    
    Eigen::Vector3d flux = Eigen::Vector3d::UnitZ();
    
//    checkAlignedDomain(&dom);
    
    std::cout << dom << std::endl;
    
//    long maxscat = 100;
//    long maxloop = 100 * maxscat;
    
//    long maxscat, maxloop;
//    ss >> maxscat >> maxloop;
//    TrajProblem prob(&mat, &dom, maxscat, maxloop);
//    std::cout << prob << std::endl;kkkk
//    std::cout << solveTraj(prob) << std::endl;

//    long nemit = 1000000;
    
//    long nemit, maxscat, maxloop;
//    ss >> nemit >> maxscat >> maxloop;
    
//    TempProblem prob(&mat, &dom, nemit, maxscat, maxloop);
//    FluxProblem prob(&mat, &dom, flux, nemit, maxscat, maxloop);
//    MultiProblem prob(&mat, &dom, Eigen::Matrix3d::Identity(),
//                      nemit, maxscat, maxloop);
    
//    long size = 10;
    
    long nemit, size, maxscat, maxloop;
    ss >> nemit >> size >> maxscat >> maxloop;

//    CumTempProblem prob(&mat, &dom, nemit, size, maxscat, maxloop);
    CumFluxProblem prob(&mat, &dom, flux, nemit, size, maxscat, maxloop);
    
    std::cout << prob << std::endl;
    
//    std::cout << solveField(prob, clk) << std::endl;
//    std::cout << solveFieldN< 10 >(prob, clk) << std::endl;
    AverageEndsF<CumFluxProblem> fun(dom, dim, prob.initElem());
    std::cout << solveFieldN< 10 >(prob, clk, fun) << std::endl;
    
    std::cout << std::endl;
    std::cout << "Total time" << std::endl;
    std::cout << clk.stopwatch() << std::endl;
    
    return 0;
}


