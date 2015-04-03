//
//  main.cpp
//  eigen
//
//  Created by Nicholas Dou on 1/23/15.
//
//

#include <Eigen/Core>
#include <iostream>

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    cout << Eigen::Vector3d::UnitX() / 2. << endl;
    return 0;
}
