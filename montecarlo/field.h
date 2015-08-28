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
using Eigen::VectorXd;
using Eigen::ArrayXXd;

typedef Eigen::Matrix<long, 3, 1> Vector3l;
typedef Eigen::Matrix<long, 5, 1> Vector5l;

class Subdomain;
class Domain;

class Field
{
private:
    typedef std::map<const Subdomain*, Vector5l> Map;
    
    const Domain* dom_;
    Map map_;
    ArrayXXd data_;
    
    Eigen::Ref<VectorXd> col(const Vector5l& stride, const Vector3l& index);
    void init(long rows);
    
public:
    Field();
    Field(long rows, const Domain* dom);
    template<typename F>
    Field(long rows, const Domain* dom, const F& fun);
    
    const ArrayXXd& data() const;
    
    Field& accumulate(const Subdomain* sdom,
                      const Vector3d& ipos, const Vector3d& fpos,
                      const VectorXd& amount);
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

#endif
