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

//int main(int argc, const char * argv[]) {
int main() {
    Clock clk;
    std::cout << clk.timestamp() << std::endl;
    
//    Material grey("input/grey_disp.txt", "input/grey_relax2.txt", 300.);
    Material si("input/Si_disp.txt", "input/Si_relax2.txt", 300.);
    FilmDomain box(Eigen::Vector3d(2e-6, 2e-8, 2e-6),
                   Eigen::Matrix<long, 3, 1>(0, 20, 0),
                   2.);
    long nemit = 1000, maxscat = 100, maxloop = 10000;
    
//    TrajProblem trajProb(&si, &box, maxscat, maxloop);
//    std::cout << "Seed" << std::endl;
//    Rng gen(getSeed());
//    std::cout << trajProb.solve(gen) << std::endl;
    
//    TempProblem tempProb(&si, &box, nemit, maxscat, maxloop);
//    FluxProblem fluxProb(&si, &box, Eigen::Vector3d::UnitX(),
//                         nemit, maxscat, maxloop);
    MultiProblem multiProb(&si, &box, Eigen::Matrix3d::Identity(),
                           nemit, maxscat, maxloop);
    
    MultiSolution multi(&multiProb);
    Progress prog = multiProb.initProgress();
    prog.clock(clk);
    std::cout << "Seeds" << std::endl;
    #pragma omp parallel
    {
        Rng gen(getSeed());
        MultiSolution sol = multiProb.solve(gen, &prog);
        #pragma omp critical
        {
            multi += sol;
        }
    }
    std::cout << multi << std::endl;
    
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
