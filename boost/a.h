//
//  a.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/19/15.
//
//

#ifndef __montecarlocpp__a__
#define __montecarlocpp__a__

//#include <boost/shared_ptr.hpp>
#include <iostream>

namespace {
    using namespace std;
}

class A {
    static int ind_;
    int i_, val_;
public:
    A(const int val = 0) : i_(ind_++), val_(val) {
        std::cout << 'A' << i_ << '(' << val_ << ')' << std::endl;
    }
    A(const A& a) : i_(ind_++), val_(a.val_) {
        std::cout << 'A' << i_ << '(' << a << ')' << std::endl;
    }
    A& operator=(const A& a) {
        val_ = a.val_;
        std::cout << 'A' << i_ << '=' << a << std::endl;
        return *this;
    }
    ~A() {
        std::cout << "~A" << i_ << std::endl;
    }
    int val() const {
        return val_;
    }
    friend std::ostream& operator<<(std::ostream& os, const A& a) {
        return os << 'A' << a.i_ << ':' << a.val_ ;
    }
};

class B {
    static int ind_;
    int i_, val_;
public:
    B(const int val = 0) : i_(ind_++), val_(val) {
        std::cout << 'B' << i_ << '(' << val_ << ')' << std::endl;
    }
    B(const B& b) : i_(ind_++), val_(b.val_) {
        std::cout << 'B' << i_ << '(' << b << ')' << std::endl;
    }
    B& operator=(const B& b) {
        val_ = b.val_;
        std::cout << 'B' << i_ << '=' << b << std::endl;
        return *this;
    }
    ~B() {
        std::cout << "~B" << i_ << std::endl;
    }
    int val() const {
        return val_;
    }
    friend std::ostream& operator<<(std::ostream& os, const B& b) {
        return os << 'B' << b.i_ << ':' << b.val_ ;
    }
};

class C : public A {
    static int ind_;
    int i_;
    C& operator=(const C& c);
public:
    C(const int val = 0) : A(val), i_(ind_++) {
        std::cout << 'C' << i_ << '(' << val << ')' << std::endl;
    }
    C(const C& c) : A(c.val()), i_(ind_++) {
        std::cout << 'C' << i_ << '(' << c << ')' << std::endl;
    }
    ~C() {
        std::cout << "~C" << i_ << std::endl;
    }
    friend std::ostream& operator<<(std::ostream& os, const C& c) {
        return os << 'C' << c.i_ << ':' << static_cast<const A&>(c);
    }
};

#endif
