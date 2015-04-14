//
//  main.cpp
//  hello
//
//  Created by Nicholas Dou on 1/22/15.
//
//

#include <vector>
#include <iostream>

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    
    std::vector<long> vec = {0, 1, 2, 3, 4};
    const long nbin = 2;
    
    for (std::vector<long>::const_iterator it = vec.begin(); it != vec.end();
         ++it)
    {
        long n = *it;
        cout << (n%nbin == 0 ? n/nbin : n/nbin + 1) << endl;
    }
    
    return 0;
}
