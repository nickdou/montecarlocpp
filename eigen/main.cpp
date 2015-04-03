//
//  main.cpp
//  eigen
//
//  Created by Nicholas Dou on 1/23/15.
//
//

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::ostream* osmain;

int main(int argc, const char * argv[]) {
    std::stringstream ss;
    for (int i = 1; i < argc; ++i) {
        ss << argv[i] << ' ';
    }
    
    std::string filename;
    ss >> filename;
    if (filename == "cout") {
        osmain = &std::cout;
    } else {
        std::ofstream ofmain(filename, std::ios::out | std::ios::trunc);
        if (!ofmain.is_open()) return 1;
        osmain = &ofmain;
    }
    
    Eigen::Vector3d vec;
    ss >> vec(0) >> vec(1) >> vec(2);
    
    *osmain << ("folder/" + filename).c_str() << std::endl;
    *osmain << vec << std::endl;
    
    return 0;
}
