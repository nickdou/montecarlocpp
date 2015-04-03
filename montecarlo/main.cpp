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
#include <vector>
#include <fstream>
#include <iostream>

std::ofstream ofmain;
std::ostream& osmain = ofmain;
//std::ostream& osmain = std::cout;

typedef Dev::result_type Seed;

Seed getSeed() {
    static Dev urandom;
    Seed s = urandom();
#pragma omp critical
    {
        osmain << s << ' ';
    }
#pragma omp barrier
#pragma omp single
    {
        osmain << std::endl;
    }
    return s;
}

TrajProblem::Solution solveTraj(const TrajProblem& prob) {
    osmain << "Seed" << std::endl;
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
            osmain << std::endl;
            osmain << "pos " << pos.format(fmt) << std::endl;
            osmain << "dir " << dir.format(fmt) << std::endl;
            prob = TrajProblem(&mat, dom, pos, dir, 2l, 2l);
            osmain << solveTraj(prob) << std::endl;
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
    prog.ostream(osmain);
    osmain << "Seeds" << std::endl;
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
        osmain << "Solution " << i << std::endl;
        Solution sol = solveField(prob, clk);
        osmain << sol << std::endl;
        vec.push_back(sol);
    }
    
    Statistics<Solution> stats(prob.initField());
    std::for_each(vec.begin(), vec.end(), stats.accumulator());
    return stats;
}

//int main(int argc, const char * argv[]) {
int main() {
    ofmain.open("./output/debug.log", std::ios::out | std::ios::trunc);
    if (!ofmain.is_open()) return 1;
    
    Clock clk;
    osmain << clk.timestamp() << std::endl;
    
#ifdef DEBUG
    osmain << "DEBUG" << std::endl;
#endif
    
//    Material mat("input/grey_disp.txt", "input/grey_relax2.txt", 300.);
    Material mat("input/Si_disp.txt", "input/Si_relax2.txt", 300.);
    
    osmain << mat << std::endl;
    
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
    
    Eigen::Matrix<double, 5, 1> dim;
    Eigen::Matrix<long, 5, 1> div;
    dim << 2e-6, 2e-6, 8e-7, 2e-7, 4e-8;
    div << 4, 4, 0, 0, 0;
    OctetDomain dom(dim, div, 5.68);
    
    Eigen::Vector3d flux = Eigen::Vector3d::UnitZ();
    
//    checkAlignedDomain(&dom);
    
    osmain << dom << std::endl;
    
    long maxscat = 100;
    long maxloop = 100 * maxscat;
    
//    TrajProblem prob(&mat, &dom, maxscat, maxloop);
//    osmain << prob << std::endl;
//    osmain << solveTraj(prob) << std::endl;

    long nemit = 1000000;

//    TempProblem prob(&mat, &dom, nemit, maxscat, maxloop);
    FluxProblem prob(&mat, &dom, flux, nemit, maxscat, maxloop);
//    MultiProblem prob(&mat, &dom, Eigen::Matrix3d::Identity(),
//                      nemit, maxscat, maxloop);
    
//    long step = maxscat / 10l;

//    CumTempProblem prob(&mat, &dom, nemit, step, maxscat, maxloop);
//    CumFluxProblem prob(&mat, &dom, flux, nemit, step, maxscat, maxloop);
    
    osmain << prob << std::endl;
    osmain << solveField(prob, clk) << std::endl;
    osmain << solveFieldN< 3 >(prob, clk) << std::endl;
    
    osmain << "Total time" << std::endl;
    osmain << clk.stopwatch() << std::endl;
    
    return 0;
}


