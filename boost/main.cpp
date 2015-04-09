//
//  main.cpp
//  boost
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "../montecarlo/data.h"
#include <boost/multi_array.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <algorithm>
#include <iostream>

namespace lambda = boost::lambda;

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    
//    typedef double Type;
//    Type z = 0.;
//    Type o = 1.;
    
    typedef Eigen::Vector3d Type;
    Type z = Eigen::Vector3d::Zero();
    Type o = Eigen::Vector3d::Ones();
    
    Data<Type> arr2( Collection(2, 2, 2) );
    Data<Type> arr3( Collection(3, 3, 3) );
    
    Type t;
    t = z;
    std::generate(arr2.data(), arr2.data() + arr2.num_elements(),
                  lambda::var(t) += o);
    t = z;
    std::generate(arr3.data(), arr3.data() + arr3.num_elements(),
                  lambda::var(t) += o);
    
    const Data<Type>* arr = &arr3;
    
    cout << *arr << endl;
    
//    Data<Type>::index_gen indices;
//    Data<Type>::index_range all;
//    boost::multi_array<Type, 2> slice(boost::extents[0][0],
//                                      boost::fortran_storage_order());
//    slice.resize( boost::extents[arr.shape()[0]][arr.shape()[1]] );
//    slice = arr[indices[all][all][0]];
//    
//    Eigen::IOFormat fmt(0, 0, " ", "", "", "", "[", "]");
//    std::for_each(slice.data(), slice.data() + slice.num_elements(),
//                  cout << lambda::bind(&Eigen::Vector3d::format,
//                                       lambda::_1, fmt) << '\n');
//    cout << endl;
//    
//    Type sum = std::accumulate(slice.data(),
//                               slice.data() + slice.num_elements(),
//                               z);
    
    Type sum = z;
    const Collection shape = arr->shapeColl();
    for (Data<Type>::Size i = 0; i < shape[0]; ++i) {
        for (Data<Type>::Size j = 0; j < shape[1]; ++j) {
            sum += (*arr)[i][j][0];
        }
    }
    
    cout << sum << endl;
    return 0;
}


