//
//  main.cpp
//  hello
//
//  Created by Nicholas Dou on 1/22/15.
//
//

#include <boost/type_traits/is_same.hpp>
#include <iostream>

struct Solution {
    static const int i = 0;
};

struct Problem {};

struct ASolution;

struct AProblem : Problem {
    typedef ASolution Solution;
};

struct ASolution {
    static const int i = 1;
};

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    cout << AProblem::Solution::i << endl;
    return 0;
}
