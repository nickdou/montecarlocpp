//
//  main.cpp
//  hello
//
//  Created by Nicholas Dou on 1/22/15.
//
//

#include <Eigen/Core>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/if.hpp>
#include <map>
#include <iostream>

template<typename T>
struct Vectorizable : boost::is_same<T, Eigen::Vector4d> {};

template<typename T>
struct ArrayAlloc
: boost::mpl::if_< Vectorizable<T>,
                   Eigen::aligned_allocator<T>,
                   std::allocator<T> > {};

template<typename K, typename T>
struct MapAlloc
: boost::mpl::if_< Vectorizable<T>,
                   Eigen::aligned_allocator< std::pair<const K, T> >,
                   std::allocator< std::pair<const K, T> > > {};

int main(int argc, const char * argv[]) {
    using std::cout;
    using std::endl;
    cout << boost::is_same<ArrayAlloc<Eigen::Vector4d>::type, Eigen::aligned_allocator<Eigen::Vector4d> >::value << endl;
    return 0;
}
