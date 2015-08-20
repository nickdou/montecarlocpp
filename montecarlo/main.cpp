//
//  main.cpp
//  montecarlo
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "problem.h"
#include "domain.h"
#include "field.h"
#include "material.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

typedef Dev::result_type Seed;

#ifdef DEBUG
Seed getSeed()
{
    static Seed s(0);
    Seed r;
#pragma omp critical
    {
        r = s++;
    }
    return r;
}

#else
Seed getSeed()
{
    static Dev urandom;
    return urandom();
}

#endif

void printSeed(Seed s)
{
#pragma omp single
    {
        std::cout << "  seeds: ";
    }
#pragma omp critical
    {
        std::cout << s << ' ';
    }
#pragma omp barrier
#pragma omp single
    {
        std::cout << std::endl;
    }
}

TrajProblem::Solution solveTraj(const TrajProblem& prob)
{
    static int n = 0;
    std::cout << "Trajectory " << n++ << std::endl;
    
    Seed s = getSeed();
    printSeed(s);
    Rng gen(s);
    TrajProblem::Solution sol = prob.solve(gen);
    
    std::cout << sol << std::endl << std::endl;
    return sol;
}

void checkDomain(const Material* mat, const Domain* dom)
{
    Eigen::Matrix<double, 3, Eigen::Dynamic> pts = dom->checkpoints();
    
    for (int j = 0; j < pts.cols(); ++j)
    {
        Eigen::Vector3d pos = pts.col(j);
        for (int i = 0; i < 6; ++i)
        {
            Eigen::Vector3d dir = Eigen::Vector3d::Unit(i % 3);
            if (i >= 3) dir *= -1;
            
            TrajProblem prob(mat, dom, pos, dir, 2l, 2l);

            std::cout << prob << std::endl;
            
            solveTraj(prob);
        }
    }
}

template<typename Derived>
typename Derived::Solution
solveField(const FieldProblem<Derived>& prob, const Clock& clk)
{
    typedef typename Derived::Solution Solution;
    
    static int n = 0;
    std::cout << "Solution " << n++ << std::endl;
    
    Solution sol = prob.initSolution();
    Progress prog = prob.initProgress();
    prog.clock(clk);
    
#pragma omp parallel
    {
        Seed s = getSeed();
        printSeed(s);
        Rng gen(s);
        Solution partial = prob.solve(gen, &prog);
        
#pragma omp critical
        {
            sol += partial;
        }
    }
    
    std::cout << sol << std::endl << std::endl;
    return sol;
}

template<typename Derived>
std::vector< typename Derived::Solution >
solveField(const FieldProblem<Derived>& prob, const Clock& clk, long nsim)
{
    typedef typename Derived::Solution Solution;
    
    std::vector<Solution> sol;
    for (int i = 0; i < nsim; ++i)
    {
        sol.push_back( solveField(prob, clk) );
    }
    return sol;
}

template<typename S>
Statistics<S> calcStats(const std::vector<S>& sol)
{
    Statistics<S> stats;
    for (typename std::vector<S>::const_iterator s = sol.begin();
         s != sol.end(); ++s)
    {
        stats.add(*s);
    }
    
    std::cout << stats << std::endl << std::endl;
    return stats;
}

//----------------------------------------
//  Main
//----------------------------------------

// grey    [T]
// silicon [T]
// custom  [T] [disp] [relax]

// bulk  [dim(0,1,2)]                      [div(0)]
// film  [dim(0,2)] [dim(1)]               [div(1)]
// hex   [dim(0)] [dim(1,2,3)]
// jct   [dim(0,1,2)] [dim(3)]
// tee   [dim(0,1,2,3)] [dim(4)]           [div(0,1,2,3)]
// tube  [dim(0)] [dim(1,2)] [dim(3)]      [div(1,2)] [div(3)]
// octet [cell] [dim(2)] [dim(3)] [dim(4)] [div(0,1)] [div(2)] [div(3)] [div(4)]

// check
// traj                   [maxscat] [maxloop]
// temp    [nemit]        [maxscat] [maxloop] [nsim]
// flux    [nemit]        [maxscat] [maxloop] [nsim]
// multi   [nemit]        [maxscat] [maxloop] [nsim]
// cumtemp [nemit] [size] [maxscat] [maxloop] [nsim]
// cumflux [nemit] [size] [maxscat] [maxloop] [nsim]

int main(int argc, const char * argv[]) {
    std::stringstream argss;
    for (int i = 1; i < argc; ++i) {
        argss << argv[i] << ' ';
    }
    
    std::cout << std::string(40, '-') << std::endl;
    
    Clock clk;
    std::cout << clk.timestamp() << std::endl;
    
#ifdef DEBUG
    std::cout << "DEBUG" << std::endl;
#endif
    
    std::cout << argss.str() << std::endl << std::endl;
    
    std::string prefix;
    argss >> prefix;
    
    int cd = chdir( prefix.c_str() );
    BOOST_ASSERT_MSG(cd == 0, "Invalid directory");
    
    std::string matStr;
    argss >> matStr;
    
    double T = 300.;
    argss >> T;
    
    const Material* mat = 0;
    if (matStr == "grey")
    {
        mat = new Material("grey_disp.txt", "grey_relax2.txt", T);
    }
    else if (matStr == "silicon")
    {
        mat = new Material("Si_disp.txt", "Si_relax2.txt", T);
    }
    else if (matStr == "custom")
    {
        std::string disp, relax;
        argss >> disp >> relax;
        
        mat = new Material(disp, relax, T);
    }
    else
    {
        BOOST_ASSERT_MSG(mat, "Invalid material");
    }
    std::cout << *mat << std::endl << std::endl;
    
    std::string domStr;
    argss >> domStr;
    
    Eigen::Matrix<double, Eigen::Dynamic, 1> dim;
    Eigen::Matrix<long, Eigen::Dynamic, 1> div;
    double deltaT = 0.;
    
    const Domain* dom = 0;
    if (domStr == "bulk")
    {
        double dim0;
        argss >> dim0;
        dim.setConstant(3, dim0);
        div.setZero(3);
        argss >> div(0);
        deltaT = 1e6 * dim0;
        
        dom = new BulkDomain(dim, div, deltaT);
    }
    else if (domStr == "film")
    {
        dim.resize(3);
        argss >> dim(0) >> dim(1);
        dim(2) = dim(0);
        div.setZero(3);
        argss >> div(1);
        deltaT = 1e6 * dim(0);
        
        dom = new FilmDomain(dim, div, deltaT);
    }
    else if (domStr == "hex")
    {
        dim.resize(4);
        argss >> dim(0) >> dim(1);
        dim(2) = dim(1);
        dim(3) = dim(1);
        long div0 = 0l;
        deltaT = 1e6 * dim(0);
        
        dom = new HexDomain(dim, div0, deltaT);
    }
    else if (domStr == "jct")
    {
        dim.resize(4);
        argss >> dim(0) >> dim(3);
        dim(1) = dim(0);
        dim(2) = dim(0);
        div.setZero(4);
        deltaT = 2e6 * dim(0);
        
        dom = new JctDomain(dim, div, deltaT);
    }
    else if (domStr == "tee")
    {
        double dim0;
        argss >> dim0;
        dim.setConstant(5, dim0);
        argss >> dim(4);
        long div0;
        argss >> div0;
        div.setConstant(5, div0);
        div(4) = 0l;
        deltaT = 3e6 * dim(0);
        
        dom = new TeeDomain(dim, div, deltaT);
    }
    else if (domStr == "tube")
    {
        dim.resize(4);
        argss >> dim(0) >> dim(1) >> dim(3);
        dim(2) = dim(1);
        div.setZero(4);
        argss >> div(1) >> div(3);
        div(2) = div(1);
        deltaT = 1e6 * dim(0);
        
        dom = new TubeDomain(dim, div, deltaT);
    }
    else if (domStr == "octet")
    {
        double cell;
        dim.resize(5);
        argss >> cell >> dim(2) >> dim(3) >> dim(4);
        double beam = cell/4. * std::sqrt(2.);
        dim(0) = beam - dim(2) - dim(4);
        dim(1) = beam - dim(3) - dim(4);
        BOOST_ASSERT_MSG((dim.array() > 0.).all(),
                         "Dimensions must be positive");
        BOOST_ASSERT_MSG(dim(2) > dim(4),
                         "Major axis smaller than wall thickness");
        div.resize(5);
        argss >> div(0) >> div(2) >> div(3) >> div(4);
        div(1) = div(0);
        deltaT = 2e6 * beam;
        
        dom = new OctetDomain(dim, div, deltaT);
    }
    else
    {
        delete mat;
        BOOST_ASSERT_MSG(dom, "Invalid domain");
    }
    std::cout << *dom << std::endl << std::endl;
    
    std::string probStr;
    argss >> probStr;
    
    long nemit = 0l;
    long size = 0l;
    long maxscat = 0l;
    long maxloop = 0l;
    long nsim = 0l;
    
    if (probStr == "check")
    {
        checkDomain(mat, dom);
    }
    else if (probStr == "traj")
    {
        argss >> maxscat >> maxloop;
        
        TrajProblem prob(mat, dom, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        solveTraj(prob);
    }
    else if (probStr == "temp")
    {
        argss >> nemit >> maxscat >> maxloop >> nsim;
        
        TempProblem prob(mat, dom, nemit, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        if (nsim == 1l) solveField(prob, clk);
        else calcStats( solveField(prob, clk, nsim) );
    }
    else if (probStr == "flux")
    {
        argss >> nemit >> maxscat >> maxloop >> nsim;
        Eigen::Vector3d dir = -(dom->gradT());
        
        FluxProblem prob(mat, dom, dir, nemit, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        if (nsim == 1l) solveField(prob, clk);
        else calcStats( solveField(prob, clk, nsim) );
    }
    else if (probStr == "multi")
    {
        argss >> nemit >> maxscat >> maxloop >> nsim;
        
        MultiProblem prob(mat, dom, nemit, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        if (nsim == 1l) solveField(prob, clk);
        else calcStats( solveField(prob, clk, nsim) );
    }
    else if (probStr == "cumtemp")
    {
        argss >> nemit >> size >> maxscat >> maxloop >> nsim;
        
        CumTempProblem prob(mat, dom, nemit, size, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        if (nsim == 1l) solveField(prob, clk);
        else calcStats( solveField(prob, clk, nsim) );
    }
    else if (probStr == "cumflux")
    {
        argss >> nemit >> size >> maxscat >> maxloop >> nsim;
        Eigen::Vector3d dir = -(dom->gradT());
        
        CumFluxProblem prob(mat, dom, dir, nemit, size, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        if (nsim == 1l) solveField(prob, clk);
        else calcStats( solveField(prob, clk, nsim) );
    }
    else
    {
        delete mat;
        delete dom;
        BOOST_ASSERT_MSG(false, "Invalid problem");
    }
    
    std::cout << "Total time" << std::endl;
    std::cout << clk.stopwatch() << std::endl << std::endl;
    
    delete mat;
    delete dom;
    return 0;
}

//template<typename Derived>
//struct AverageOctetF {
//    typedef typename Derived::Type Type;
//    static const int nsdom = 7;
//    static const int itop = 26;
//    const OctetDomain* oct;
//    Eigen::VectorXd weight;
//    Type zero;
//    AverageOctetF(const OctetDomain& dom,
//                 const Eigen::Matrix<double, 5, 1>& dim,
//                 const Type& z)
//    : oct(&dom), weight(nsdom), zero(z)
//    {
//        weight << dim[3], dim[4], dim[2],
//                  dim[4], dim[2] - dim[4], dim[4], dim[3];
//    }
//    Type operator()(const Field<Type>& fld) const {
//        Type fldSum = zero;
//        for (int b = 0; b < 2; ++b) {
//            int begin = (b == 0 ? 0 : itop);
//            for (int s = 0; s < nsdom; ++s) {
//                typedef typename Field<Type>::const_iterator Iter;
//                Iter it = fld.find(oct->sdomPtrs()[begin+s]);
//                BOOST_ASSERT_MSG(it != fld.end(), "Subdomain not found");
//                const Data<Type>& data = it->second;
//                Collection shape = data.shapeColl();
//                Type sdomSum = zero;
//                for (typename Data<Type>::Size i = 0; i < shape[0]; ++i) {
//                    for (typename Data<Type>::Size j = 0; j < shape[1]; ++j) {
//                        if (s == 0) {
//                            sdomSum += data[i][j][0];
//                        } else {
//                            sdomSum += data[i][j][shape[2]-1];
//                        }
//                    }
//                }
//                fldSum += weight(s) * sdomSum / (shape[0] * shape[1]);
//            }
//        }
//        return fldSum / (2*weight.sum());
//    }
//};


