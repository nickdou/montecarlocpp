//
//  field.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 8/7/15.
//
//

#ifndef __montecarlocpp__field__
#define __montecarlocpp__field__

#include <Eigen/Core>
#include <map>
#include <iostream>

using Eigen::Vector3d;

typedef Eigen::Matrix<long, 3, 1> Vector3l;

class Subdomain;
class Domain;

template<typename T, int N>
class Field;

template<typename T, int N>
Field<T, N> operator+(const Field<T, N>& fld1, const Field<T, N>& fld2);

template<typename T, int N>
Field<T, N> operator-(const Field<T, N>& fld1, const Field<T, N>& fld2);

template<typename T, int N>
Field<T, N> operator*(const Field<T, N>& fld1, const Field<T, N>& fld2);

template<typename T, int N>
Field<T, N> operator/(const Field<T, N>& fld1, const Field<T, N>& fld2);

template<typename T, int N>
std::ostream& operator<<(std::ostream& os, const Field<T, N>& fld);

template<typename T, int N>
class Field
{
public:
    typedef T Type;
    static const int Num = N;
    
private:
    typedef Eigen::Array<T, N, Eigen::Dynamic> ArrayNXT;
    typedef Eigen::Matrix<T, N, 1> VectorNT;
    typedef std::map< const Subdomain*, Eigen::Matrix<long, 5, 1> > Map;
    
    const Domain* dom_;
    Map map_;
    ArrayNXT data_;
    
    void initDom();
    void setRows(long rows);
    
public:
    Field();
    Field(T elem);
    Field(const VectorNT& vec);
    explicit Field(const Domain* dom);
    
    template<typename F>
    explicit Field(const Domain* dom, const F& fun);
    
    template<typename F>
    Field& transform(const F& fun);
    
    Field& accumulate(const Subdomain* sdom,
                      const Vector3d& ipos, const Vector3d& fpos,
                      const VectorNT& amount);
    
    Field& operator+=(const Field& fld);
    Field& operator-=(const Field& fld);
    Field& operator*=(const Field& fld);
    Field& operator/=(const Field& fld);
    
    VectorNT average(const Field& weight = Field( static_cast<T>(1) )) const;
    
    const ArrayNXT& data() const;
    
    friend std::ostream& operator<< <>(std::ostream& os, const Field& fld);
    
private:
    Eigen::Ref<VectorNT> col(const Eigen::Matrix<long, 5, 1>& stride,
                             const Vector3l& index);
    ArrayNXT match(const Field& fld);
};

template<typename S>
class Statistics
{
private:
    long n_;
    S z_, m_, s_;
    
public:
    Statistics();
    Statistics(const S& zero);
    
    S mean() const;
    S variance() const;
    
    void add(const S& x);
};

template<typename S>
std::ostream& operator<<(std::ostream& os, const Statistics<S>& stats);

#endif
