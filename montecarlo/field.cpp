//
//  field.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 8/7/15.
//
//

#include "problem.h"
#include "field.h"
#include "domain.h"
#include "subdomain.h"
#include <boost/assert.hpp>
#include <iostream>
#include <cmath>

//----------------------------------------
//  Field
//----------------------------------------

Field::Field()
: dom_(0)
{}

void Field::init(long rows)
{
    BOOST_ASSERT_MSG(dom_->isInit(), "Domain not initialized");
    
    long cols = 0;
    for (Subdomain::Pointers::const_iterator s = dom_->sdomPtrs().begin();
         s != dom_->sdomPtrs().end(); ++s)
    {
        Vector3l shape = (*s)->shape();
        long shapeProd = shape.prod();
        
        if (shapeProd == 0) continue;
        
        Vector5l stride;
        stride << cols, 1, shape(0), shape(0)*shape(1), shapeProd;
        map_[*s] = stride;
        
        cols += shapeProd;
    }
    data_.setZero(rows, cols);
}

Field::Field(long rows, const Domain* dom)
: dom_(dom)
{
    init(rows);
}

template<typename F>
Field::Field(long rows, const Domain* dom, const F& fun)
: dom_(dom)
{
    init(rows);
    
    long n = 0;
    for (Subdomain::Pointers::const_iterator s = dom_->sdomPtrs().begin();
         s != dom_->sdomPtrs().end(); ++s)
    {
        Vector3l shape = (*s)->shape();
        
        for (long k = 0; k < shape(2); ++k)
        {
            for (long j = 0; j < shape(1); ++j)
            {
                for (long i = 0; i < shape(0); ++i)
                {
                    data_.col(n) = fun(*s, Vector3l(i, j, k));
                    n++;
                }
            }
        }
    }
}

template Field::Field(long, const Domain*, const CellVolF&);
template Field::Field(long, const Domain*, const OctetDomain::WeightF&);

const ArrayXXd& Field::data() const
{
    return data_;
}

Eigen::Ref<VectorXd> Field::col(const Vector5l& stride, const Vector3l& index)
{
    return data_.col(stride(0) + stride.segment<3>(1).dot(index));
}

Field& Field::accumulate(const Subdomain* sdom,
                         const Vector3d& bpos, const Vector3d& epos,
                         const VectorXd& amount)
{
    int flag = sdom->accumFlag();
    if (flag < -1)
    {
        return *this;
    }
    
    Map::const_iterator it = map_.find(sdom);
    BOOST_ASSERT_MSG(it != map_.end(), "Subdomain not found in field");
    Vector5l stride = it->second;
    
    if (flag < 0)
    {
        col(stride, Vector3l::Zero()) += amount;
        return *this;
    }
    
    Vector3d bcoord = sdom->coord(bpos);
    Vector3d ecoord = sdom->coord(epos);
    Vector3d dcoord = ecoord - bcoord;
    
    Vector3l bindex = sdom->coord2index(bcoord);
    Vector3l eindex = sdom->coord2index(ecoord);

    if (flag < 3)
    {
        long d = flag;
        long b = bindex(d);
        long e = eindex(d);
        
        Vector3l bvec = b * Vector3l::Unit(d);
        Vector3l evec = e * Vector3l::Unit(d);
        
        if (b == e)
        {
            col(stride, bvec) += amount;
            return *this;
        }
        
        VectorXd cellAmount = amount / std::abs(dcoord(d));
        int pm;
        if (b < e)
        {
            col(stride, bvec) += cellAmount * (1 + b - bcoord(d));
            col(stride, evec) += cellAmount * (ecoord(d) - e);
            pm = 1;
        }
        else
        {
            col(stride, bvec) += cellAmount * (bcoord(d) - b);
            col(stride, evec) += cellAmount * (1 + e - ecoord(d));
            pm = -1;
        }
        
        Vector3l nvec = Vector3l::Zero();
        for (long n = b + pm; n != e; n += pm)
        {
            nvec(d) = n;
            col(stride, nvec) += cellAmount;
        }
    }
    else
    {
        std::map<double, Vector3l> borders;
        std::map<double, Vector3l>::iterator ins = borders.begin();
        for (int d = 0; d < 3; d++)
        {
            if (std::abs(dcoord(d)) < Dbl::min()) continue;
            
            long b = bindex(d);
            long e = eindex(d);
            int pm;
            if (b == e)
            {
                continue;
            }
            else if (b < e)
            {
                b++;
                e++;
                pm = 1;
            }
            else
            {
                pm = -1;
            }
            
            ins = borders.begin();
            Vector3l step = pm * Vector3l::Unit(d);
            bool search = !borders.empty();
            for (long n = b; n != e; n += pm)
            {
                double param = (n - bcoord(d)) / dcoord(d);
                
                if (search)
                {
                    std::map<double, Vector3l>::iterator found;
                    found = borders.find(param);
                    if (found != borders.end())
                    {
                        found->second += step;
                        ins = found;
                        continue;
                    }
                }
                
                std::map<double, Vector3l>::value_type pair(param, step);
                ins = borders.insert(ins, pair);
            }
        }
        std::map<double, Vector3l>::value_type pairEnd(1., Vector3l::Zero());
        borders.insert(ins, pairEnd);
        
        Vector3l index = bindex;
        double param = 0.;
        for (std::map<double, Vector3l>::const_iterator it = borders.begin();
             it != borders.end(); ++it)
        {
            col(stride, index) += amount * (it->first - param);
            param = it->first;
            index += it->second;
        }
        BOOST_ASSERT_MSG(index == eindex, "Final index incorrect");
    }
    return *this;
}

//----------------------------------------
//  Statistics
//----------------------------------------

template<typename S>
Statistics<S>::Statistics()
: n_(0)
{}

template<typename S>
Statistics<S>::Statistics(const S& zero)
: n_(0), z_(zero), m_(zero), s_(zero)
{}

template<typename S>
S Statistics<S>::mean() const
{
    return m_;
}

template<typename S>
S Statistics<S>::variance() const
{
    if (n_ < 2)
    {
        return z_;
    }
    else
    {
        return s_ / (n_ - 1);
    }
}

template<typename S>
void Statistics<S>::add(const S& x)
{
    n_++;
    S d = x - m_;
    m_ += d / n_;
    s_ += d * (x - m_);
}

template class Statistics<ArrayXXd>;

