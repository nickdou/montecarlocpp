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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using Eigen::Matrix3d;
using Eigen::ArrayXXd;

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
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

void printSolution(const ArrayXXd& sol)
{
    std::ios_base::fmtflags mask = std::cout.flags();
    
    std::cout << std::scientific << std::setprecision(9);
    
    for (long i = 0; i < sol.rows(); ++i)
    {
        for (long j = 0; j < sol.cols(); ++j)
        {
            std::cout << std::setw(16) << sol(i, j) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout.flags(mask);
}

ArrayXXd solveTraj(const TrajProblem* prob, const Clock& clk)
{
    static long n = 0;
    std::cout << "Trajectory " << n++ << std::endl;
    
    Progress prog = prob->initProgress();
    prog.clock(clk);
    
    Seed s = getSeed();
    printSeed(s);
    Rng gen(s);
    ArrayXXd sol = prob->solve(gen, &prog);
    
    printSolution(sol);
    
    return sol;
}

Matrix3Xd checkDomain(const Material* mat, const Domain* dom,
                      const Matrix3d& rot, const Clock& clk)
{
    std::vector<ArrayXXd> traj;
    std::vector<long> ind;
    ind.push_back(0);
    
    Matrix3Xd pts = dom->checkpoints();
    for (long p = 0; p < pts.cols(); ++p)
    {
        Vector3d pos = pts.col(p);
        for (int i = 0; i < 6; ++i)
        {
            Vector3d dir = rot.col(i % 3);
            if (i >= 3) dir *= -1;
            
            TrajProblem prob(mat, dom, pos, dir, 2, 2);

            std::cout << prob << std::endl << std::endl;
            
            traj.push_back( solveTraj(&prob, clk) );
            
            ind.push_back(ind.back() + traj.back().cols() + 1);
        }
    }
    
    Matrix3Xd sol(3, ind.back() - 1);
    sol.setConstant( Dbl::quiet_NaN() );
    for (long n = 0; n < traj.size(); ++n)
    {
        long j = ind.at(n);
        long cols = ind.at(n + 1) - ind.at(n) - 1;
        sol.block(0, j, 3, cols) = traj.at(n);
    }
    
    std::cout << "Combined Trajectory" << std::endl;
    printSolution(sol);
    
    return sol;
}


ArrayXXd solveField(const FieldProblem* prob, const Clock& clk)
{
    static long n = 0;
    std::cout << "Solution " << n++ << std::endl;
    
    ArrayXXd sol = prob->initSolution();
    Progress prog = prob->initProgress();
    prog.clock(clk);
    
#pragma omp parallel
    {
        Seed s = getSeed();
        printSeed(s);
        Rng gen(s);
        ArrayXXd partial = prob->solve(gen, &prog);
        
#pragma omp critical
        {
            sol += partial;
        }
    }
    
    std::cout << "Output" << std::endl;
    printSolution(sol);
    
    ArrayXXd avg = prob->dom()->average(sol);
    if (avg.cols() != sol.cols() || avg.rows() != sol.rows() ||
        (avg != sol).any())
    {
        std::cout << "Averaged" << std::endl;
        printSolution(avg);
    }
    return avg;
}

std::vector<ArrayXXd> solveField(const FieldProblem* prob, long nsim,
                                 const Clock& clk)
{
    std::vector<ArrayXXd> sol;
    for (long i = 0; i < nsim; ++i)
    {
        sol.push_back( solveField(prob, clk) );
    }
    return sol;
}

Statistics<ArrayXXd> calcStats(const std::vector<ArrayXXd>& sol)
{
    long rows = sol.front().rows();
    long cols = sol.front().cols();
    Statistics<ArrayXXd> stats( ArrayXXd::Zero(rows, cols) );
    
    for (typename std::vector<ArrayXXd>::const_iterator s = sol.begin();
         s != sol.end(); ++s)
    {
        stats.add(*s);
    }
    
    std::cout << "Mean" << std::endl;
    printSolution(stats.mean());
    std::cout << "Standard Deviation" << std::endl;
    printSolution(stats.variance().sqrt());
    
    return stats;
}

//----------------------------------------
//  Main
//----------------------------------------

// grey    [T]
// silicon [T]
// custom  [T] [disp] [relax]

// bulk  [dim(0,1,2)]                        [div(0)]
// film  [dim(0,2)] [dim(1)]                 [div(1)]
// hex   [dim(0)] [dim(1,2,3)]
// jct   [dim(0,1,2)] [dim(3)]
// tee   [dim(0,1,2,3)] [dim(4)]             [div(0,1,2,3)]
// tube  [dim(0)] [dim(1,2)] [dim(3)]        [div(1,2)] [div(3)]
// octet [dim(0)] [dim(1)] [dim(2)] [dim(3)] [div(0)] [div(1)] [div(2)] [div(3)]
//       [dT]

// check   [rot]
// traj    [pos]   [dir]  [maxscat] [maxloop]
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
    BOOST_ASSERT_MSG(!argss.fail(), "Invalid material arguments");
    
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
        deltaT = 1e6 * dim(0);
        
        dom = new HexDomain(dim, deltaT);
    }
    else if (domStr == "pyr")
    {
        dim.resize(3);
        argss >> dim(0) >> dim(1);
        dim(2) = dim(1);
        deltaT = 1e6 * dim(0);
        
        dom = new PyrDomain(dim, deltaT);
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
        div(4) = 0;
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
        dim.resize(4);
        argss >> dim(0) >> dim(1) >> dim(2) >> dim(3);
        div.resize(4);
        argss >> div(0) >> div(1) >> div(2) >> div(3);
        argss >> deltaT;
        BOOST_ASSERT_MSG((dim.array() > 0.).all(),
                         "Dimensions must be positive");
        
        dom = new OctetDomain(dim, div, deltaT);
    }
    else
    {
        delete mat;
        BOOST_ASSERT_MSG(dom, "Invalid domain");
    }
    BOOST_ASSERT_MSG(!argss.fail(), "Invalid domain arguments");
    
    std::cout << *dom << std::endl << std::endl;
    
    std::string probStr;
    argss >> probStr;
    
    long nemit = 0;
    long size = 0;
    long maxscat = 0;
    long maxloop = 0;
    long nsim = 0;
    
    const FieldProblem* prob = 0;
    if (probStr == "check")
    {
        Matrix3d rot;
        argss >> rot(0, 0) >> rot(0, 1) >> rot(0, 2);
        argss >> rot(1, 0) >> rot(1, 1) >> rot(1, 2);
        argss >> rot(2, 0) >> rot(2, 1) >> rot(2, 2);
        
        checkDomain(mat, dom, rot, clk);
    }
    else if (probStr == "traj")
    {
        Vector3d pos, dir;
        argss >> pos(0) >> pos(1) >> pos(2);
        argss >> dir(0) >> dir(1) >> dir(2);
        argss >> maxscat >> maxloop;
        
        TrajProblem prob(mat, dom, pos, dir, maxscat, maxloop);
        std::cout << prob << std::endl << std::endl;
        
        solveTraj(&prob, clk);
    }
    else if (probStr == "temp")
    {
        argss >> nemit >> maxscat >> maxloop >> nsim;
        
        prob = new TempProblem(mat, dom, nemit, maxscat, maxloop);
    }
    else if (probStr == "flux")
    {
        argss >> nemit >> maxscat >> maxloop >> nsim;
        
        prob = new FluxProblem(mat, dom, nemit, maxscat, maxloop);
    }
    else if (probStr == "multi")
    {
        argss >> nemit >> maxscat >> maxloop >> nsim;
        
        prob = new MultiProblem(mat, dom, nemit, maxscat, maxloop);
    }
    else if (probStr == "cumtemp")
    {
        argss >> nemit >> size >> maxscat >> maxloop >> nsim;
        
        prob = new CumTempProblem(mat, dom, nemit, size, maxscat, maxloop);
    }
    else if (probStr == "cumflux")
    {
        argss >> nemit >> size >> maxscat >> maxloop >> nsim;
        
        prob = new CumFluxProblem(mat, dom, nemit, size, maxscat, maxloop);
    }
    else
    {
        delete mat;
        delete dom;
        BOOST_ASSERT_MSG(false, "Invalid problem");
    }
    BOOST_ASSERT_MSG(!argss.fail(), "Invalid problem arguments");
    
    if (prob)
    {
        std::cout << *prob << std::endl << std::endl;
        
        if (nsim == 1) solveField(prob, clk);
        else calcStats( solveField(prob, nsim, clk) );
    }
    
    std::cout << "Total time: " << clk.stopwatch() << std::endl << std::endl;
    
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


