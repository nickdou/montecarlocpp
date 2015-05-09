//
//  main.cpp
//  eigen
//
//  Created by Nicholas Dou on 1/23/15.
//
//

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    
    Eigen::Vector3d vec(0, 1, 2);
    cout << vec << endl;
    return 0;
}
