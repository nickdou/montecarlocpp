//
//  main.cpp
//  hello
//
//  Created by Nicholas Dou on 1/22/15.
//
//

#include "../montecarlo/domain.h"
#include "../montecarlo/boundary.h"
#include "../montecarlo/data.h"
#include <Eigen/Core>
#include <utility>
#include <vector>
#include <iostream>

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    typedef Parallelepiped<SpecBoundary, SpecBoundary, SpecBoundary> Sdom;
    std::vector<Sdom> vec;
    typedef Eigen::Array3d Type;
    Type zero = Type::Zero();
    Field<Type> fld1, fld2, fldz;
    for (int s = 0; s < 2; ++s) {
        typedef Field<Type>::value_type Value;
        vec.push_back(Sdom(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(),
                           Eigen::Matrix<long, 3, 1>(5, 0, 0)));
        fld1.insert(Value(&vec.back(), vec.back().initData(Type(2., 4., 6.))));
        fld2.insert(Value(&vec.back(), vec.back().initData(Type(1., 2., 3.))));
        fldz.insert(Value(&vec.back(), vec.back().initData(zero)));
    }
    Data<Type> data1 = fld1.begin()->second;
    Data<Type> data2 = fld2.begin()->second;
    Data<Type> dataz = fldz.begin()->second;
    
//    cout << "d1" << endl << data1 << endl;
//    cout << "d2" << endl << data2 << endl;
//    cout << "+"  << endl << data1 + data2 << endl;
//    cout << "-"  << endl << data1 - data2 << endl;
//    cout << "*"  << endl << data1 * data2 << endl;
//    cout << "/"  << endl << data1 / data2 << endl;
//    cout << "+2" << endl << data1 + 2. << endl;
//    cout << "-2" << endl << data1 - 2. << endl;
//    cout << "*2" << endl << data1 * 2. << endl;
//    cout << "/2" << endl << data1 / 2. << endl;
//    cout << "2+" << endl << 2. + data1 << endl;
//    cout << "2-" << endl << 2. - data1 << endl;
//    cout << "2*" << endl << 2. * data1 << endl;
//    cout << "2/" << endl << 2. / data1 << endl;
//    
//    cout << "f1" << endl << fld1 << endl;
//    cout << "f2" << endl << fld2 << endl;
//    cout << "+"  << endl << fld1 + fld2 << endl;
//    cout << "-"  << endl << fld1 - fld2 << endl;
//    cout << "*"  << endl << fld1 * fld2 << endl;
//    cout << "/"  << endl << fld1 / fld2 << endl;
//    cout << "+2" << endl << fld1 + 2. << endl;
//    cout << "-2" << endl << fld1 - 2. << endl;
//    cout << "*2" << endl << fld1 * 2. << endl;
//    cout << "/2" << endl << fld1 / 2. << endl;
//    cout << "2+" << endl << 2. + fld1 << endl;
//    cout << "2-" << endl << 2. - fld1 << endl;
//    cout << "2*" << endl << 2. * fld1 << endl;
//    cout << "2/" << endl << 2. / fld1 << endl;
    
//    std::vector<double> vec;
//    for (int i = 0; i < 10; ++i) vec.push_back(static_cast<double>(i));
//    Statistics<double> stats(0.);
//    std::for_each(vec.begin(), vec.end(), stats.accumulator());
    
    Statistics< Field<Type> > stats(fldz);
    stats.add(fld1);
    stats.add(fld2);
    cout << stats.mean() << endl;
    cout << stats.variance() << endl;
    
    return 0;
}
