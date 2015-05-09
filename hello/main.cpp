//
//  main.cpp
//  hello
//
//  Created by Nicholas Dou on 1/22/15.
//
//

#include "../montecarlo/domain.h"
#include "../montecarlo/phonon.h"
#include "../montecarlo/random.h"
#include <vector>
#include <iomanip>
#include <iostream>

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    
    using Eigen::Vector3d;
    using Eigen::Matrix3d;
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    
    Vector3d o = Vector3d::Zero();
    Vector3l div = Vector3l(3, 4, 1);
    Matrix3d mat = div.cast<double>().asDiagonal();
//    Vector3l shape = div.cwiseMax(1);
    Vector3d gradT = Vector3d::UnitZ();
    
    typedef Tetrahedron<SpecBoundary, SpecBoundary, SpecBoundary, SpecBoundary> Sdom;
    Sdom sdom(o, mat, div, gradT);
    
    cout << std::setprecision(6);
//    cout << std::setw(9) << sdom.sdomVol() << endl;
//    
//    for (int k = 0; k < shape(2); ++k) {
//        for (int i = 0; i < shape(0); ++i) {
//            for (int j = 0; j < shape(1); ++j) {
//                cout << std::setw(9) << sdom.cellVol(Vector3l(i, j, k)) << ' ';
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
    
    Dev urandom;
    Rng gen(urandom());
    
    const int N = 1000;
    Eigen::Matrix<double, N, 3> pos;
    for (int i = 0; i < N; ++i) {
        Phonon phn = sdom.emit(Phonon::Prop(), gen);
        pos.row(i) = phn.pos().transpose();
    }
    cout << pos << endl;
    
    return 0;
}
