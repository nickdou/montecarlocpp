//
//  main.cpp
//  montecarlo
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "problem.h"
#include "domain.h"
#include "material.h"
#include "random.h"
#include <Eigen/Core>
#include <iostream>

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

Trajectory trajSolve(const TrajProblem& prob) {
    std::cout << "Seed" << std::endl;
    Rng gen(getSeed());
    return prob.solve(gen);
}

template<typename P>
typename P::Sol fieldSolve(const P& prob, const Clock& clk) {
    typedef typename P::Sol Sol;
    Sol tot(&prob);
    Progress prog = prob.initProgress();
    prog.clock(clk);
    std::cout << "Seeds" << std::endl;
    #pragma omp parallel
    {
        Rng gen(getSeed());
        Sol sol = prob.solve(gen, &prog);
        #pragma omp critical
        {
            tot += sol;
        }
    }
    return tot;
}

//int main(int argc, const char * argv[]) {
int main() {
    Clock clk;
    std::cout << clk.timestamp() << std::endl;
    
//    Material mat("input/grey_disp.txt", "input/grey_relax2.txt", 300.);
    Material mat("input/Si_disp.txt", "input/Si_relax2.txt", 300.);
    
//    FilmDomain dom(Eigen::Vector3d(2e-6, 2e-8, 2e-6),
//                   Eigen::Matrix<long, 3, 1>(0, 20, 0),
//                   2.);
    
    Eigen::Matrix<double, 5, 1> dim;
    dim << 2e-8, 2e-8, 2e-8, 2e-8, 2e-8;
    Eigen::Matrix<long, 5, 1> div;
    div << 10, 10, 10, 10, 0;
    TeeDomain dom(dim, div, 3.);
    
    long maxscat = 100, maxloop = 10000;
    
//    TrajProblem prob(&mat, &dom, maxscat, maxloop);
//    std::cout << trajSolve(prob) << std::endl;
    
    long nemit = 1000;
    
//    TempProblem prob(&mat, &dom, nemit, maxscat, maxloop);
//    FluxProblem prob(&mat, &dom, Eigen::Vector3d::UnitX(),
//                     nemit, maxscat, maxloop);
    MultiProblem prob(&mat, &dom, Eigen::Matrix3d::Identity(),
                      nemit, maxscat, maxloop);
    
    std::cout << fieldSolve(prob, clk) << std::endl;
    
    return 0;
}

void testScatter(Rng& gen) {
    Material silicon("input/Si_disp.txt", "input/Si_relax2.txt", 300.);
    const int N = 100;
    double scatDist[N];
    double sum = 0;
    Phonon::Prop prop = silicon.drawScatProp(gen);
    std::cout << prop.w() << ' ' << prop.p() << std::endl;
    for (int i = 0; i < 100; ++i) {
        scatDist[i] = silicon.drawScatDist(prop, gen);
        std::cout << scatDist[i] << std::endl;
        sum += scatDist[i];
    }
    std::cout << "Average: " << sum / N << std::endl;
}
