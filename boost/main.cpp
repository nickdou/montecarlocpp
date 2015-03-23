//
//  main.cpp
//  boost
//
//  Created by Nicholas Dou on 1/27/15.
//
//

#include "a.h"
#include <boost/type_traits.hpp>
#include <boost/fusion/view.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/optional.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random.hpp>
#include <limits>
#include <iostream>

using boost::mpl::_;

namespace fusion = boost::fusion;

struct print {
    typedef const std::ostream& type;
    template<typename T>
    type operator()(type os, const T& x) const {
        const_cast<std::ostream&>(os) << x << ' ';
        return os;
    }
};

struct add {
    typedef int result_type;
    template<typename T>
    int operator()(int i, const T& x) const {
        return i + x.val();
    }
};

/*
struct append {
    typedef const std::vector<const A*>& type;
    template<typename T>
    type operator()(type vec, const T& x) const {
        const_cast<std::vector<const A*>&>(vec).push_back(&x);
        return vec;
    }
};
*/
struct append {
    typedef std::vector<const A*> type;
    template<typename T>
    type* operator()(type* vec, const T& x) const {
        vec->push_back(&x);
        return vec;
    }
};

void testVec() {
    typedef boost::fusion::vector4<A, A, C, C> Vector;
    Vector u(1, 2, 3, 4);
    std::cout << u << std::endl;
    fusion::fold(fusion::filter_view<Vector,
                 boost::is_base_of<C,_>>(u), std::cout, print());
    std::cout << std::endl;
    typedef fusion::filter_view<Vector, boost::is_base_of<C,_>> Seq;
    typedef std::vector<const A*> State;
    State vec;
    fusion::fold(Seq(u), &vec, append());
    for(State::iterator it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << std::endl;
    }
}

void testInPlace() {
    boost::optional<A> a;
    a = boost::in_place(1);
    cout << *a << endl;
}

void testDevice() {
    boost::random_device dev;
    cout << '[' << dev.min() << ' ' << dev.max() << ']' << endl;
    cout << '[' << std::numeric_limits<unsigned int>::min() <<
    ' ' << std::numeric_limits<unsigned int>::max() << ']' << endl;
    cout << dev() << endl;
}


int main(int argc, const char * argv[]) {
    testDevice();
    return 0;
}


