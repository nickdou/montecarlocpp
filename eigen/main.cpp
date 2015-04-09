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
    
    Eigen::VectorXd vec(5);
    vec << 1, 2, 3, 4, 5;
    cout << vec.sum() << endl;
    return 0;
}
