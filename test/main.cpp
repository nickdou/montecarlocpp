//
//  main.cpp
//  test
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "tools.h"
#include <Eigen/Core>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;

void testExtract() {
    string input("1 1 1\n1 1 1\n1 1 1");
    istringstream is(input);
    Array22d a, b, c;
    tools::extractArray(is, a);
    cout << a << endl;
    b = tools::cumSum(a);
    cout << b << endl;
    c = tools::pdf2Cdf(a);
    cout << c << endl;
}

void testSearch() {
    const int N = 10;
    double array[N];
    for (int i = 0; i < N; i++) array[i] = static_cast<double>(i);
    cout << tools::searchBin(array, 1.5, N) << endl;
    
    Vector3d a(0,1,2);
    cout << tools::searchBin(a, 1.5) << endl;
    
    vector<double> v(N);
    for (int i = 0; i < N; i++) v[i] = static_cast<double>(i);
    cout << tools::searchBin(v, 1.5) << endl;
    cout << tools::searchBin(array, 1.5, 5, N) << endl;
}



int main(int argc, const char * argv[]) {
//    testExtract();
    testSearch();
    return 0;
}
