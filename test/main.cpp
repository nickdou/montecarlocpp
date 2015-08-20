//
//  main.cpp
//  test
//
//  Created by Nicholas Dou on 8/5/15.
//
//

#include <Eigen/Core>
#include <iostream>
#include <sstream>
#include <string>

using namespace Eigen;
using std::cout;
using std::endl;

int getNext() {
    static int i = 0;
    int next = i++;
    return next;
}

int main(int argc, const char * argv[])
{
    for (int j = 0; j < 10; ++j)
    {
        cout << getNext() << endl;
    }
}
