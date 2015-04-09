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
    
    const int N = 14;
    const int arr[N] = { 0,  1,  2,  3,  4,  5,  6,
                        26, 27, 28, 29, 30, 31, 32};
    
    for(const int* it = arr; it != arr + N; ++it) {
        cout << *it << ' ';
    }
    cout << endl;
    
    return 0;
}
