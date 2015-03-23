//
//  main.cpp
//  eigen
//
//  Created by Nicholas Dou on 1/23/15.
//
//

#include <Eigen/Core>
#include <iostream>
#include <limits>

using std::cout;
using std::endl;

int main(int argc, const char * argv[]) {
    Eigen::Array4d a(0.,1.,2.,3.);
    cout << std::numeric_limits<double>::epsilon() << endl;
    return 0;
}
