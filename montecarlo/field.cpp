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
#include <algorithm>
#include <iostream>
#include <cmath>

template<typename T, int N>
Field<T, N>::Field()
: dom_(0)
{
    data_.setZero(std::max(1, N), 1);
}

template<typename T, int N>
Field<T, N>::Field(T elem)
: dom_(0)
{
    data_.setConstant(std::max(1, N), 1, elem);
}

template<typename T, int N>
Field<T, N>::Field(const VectorNT& vec)
: dom_(0), data_(vec)
{}

template<typename T, int N>
void Field<T, N>::initDom()
{
    long ncell = 0l;
    for (Subdomain::Pointers::const_iterator s = dom_->sdomPtrs().begin();
         s != dom_->sdomPtrs().end(); ++s)
    {
        Vector3l shape = (*s)->shape();
        long shapeProd = shape.prod();
        BOOST_ASSERT_MSG(shapeProd > 0l, "Subdomain has no cells");
        
        Eigen::Matrix<long, 4, 1> stride;
        stride << ncell, 1l, shape(0), shape(0)*shape(1);
        map_[*s] = stride;
        
        ncell += shapeProd;
    }
    
    data_.setZero(std::max(1, N), ncell);
}

template<typename T, int N>
void Field<T, N>::setRows(long rows)
{
    if (N == Eigen::Dynamic && rows > 1l && data_.rows() == 1l)
    {
        data_.conservativeResize(rows, data_.cols());
        data_ = data_.row(0).replicate(rows, 1l);
    }
}

template<typename T, int N>
Field<T, N>::Field(const Domain* dom)
: dom_(dom)
{
    initDom();
}

template<typename T, int N>
template<typename F>
Field<T, N>::Field(const Domain* dom, const F& fun)
: dom_(dom)
{
    initDom();
    
    VectorNT vec = fun(dom->sdomPtrs().front(), Vector3l::Zero());
    setRows(vec.rows());
    
    long n = 0l;
    for (Subdomain::Pointers::const_iterator s = dom->sdomPtrs().begin();
         s != dom->sdomPtrs().end(); ++s)
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

template<typename T, int N>
template<typename F>
Field<T, N>& Field<T, N>::transform(const F& fun)
{
    for (long n = 0; n < data_.cols(); ++n)
    {
        data_.col(n) = fun( data_.col(n) );
    }
    return *this;
}

template<typename T, int N>
Eigen::Ref< typename Field<T, N>::VectorNT >
Field<T, N>::col(const Eigen::Matrix<long, 4, 1>& stride, const Vector3l& index)
{
    return data_.col(stride(0) + stride.tail<3>().dot(index));
}

template<typename T, int N>
Field<T, N>& Field<T, N>::accumulate(const Subdomain* sdom,
                                     const Vector3d& ipos, const Vector3d& fpos,
                                     const VectorNT& amount)
{
    setRows(amount.rows());
    
    Map::const_iterator it = map_.find(sdom);
    BOOST_ASSERT_MSG(it != map_.end(), "Subdomain not found in field");
    Eigen::Matrix<long, 4, 1> stride = it->second;
    
    int flag = sdom->accumFlag();
    if (flag < 0)
    {
        col(stride, Vector3l::Zero()) += amount;
        return *this;
    }
    
    Vector3d icoord = sdom->coord(ipos);
    Vector3d fcoord = sdom->coord(fpos);
    Vector3d dcoord = fcoord - icoord;
    
    Vector3l iindex = sdom->coord2index(icoord);
    Vector3l findex = sdom->coord2index(fcoord);

    if (flag < 3)
    {
        long d = flag;
        long i = iindex(d);
        long f = findex(d);
        
        Vector3l ivec = i * Vector3l::Unit(d);
        Vector3l fvec = f * Vector3l::Unit(d);
        
        if (i == f)
        {
            col(stride, ivec) += amount;
            return *this;
        }
        
        VectorNT cellAmount = amount / std::abs(dcoord(d));
        int pm;
        if (i < f)
        {
            col(stride, ivec) += cellAmount * (1l + i - icoord(d));
            col(stride, fvec) += cellAmount * (fcoord(d) - f);
            pm = 1;
        }
        else
        {
            col(stride, ivec) += cellAmount * (icoord(d) - i);
            col(stride, fvec) += cellAmount * (1l + f - fcoord(d));
            pm = -1;
        }
        
        Vector3l nvec = Vector3l::Zero();
        for (long n = i + pm; n != f; n += pm)
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
            
            long i = iindex(d);
            long f = findex(d);
            int pm;
            if (i == f)
            {
                continue;
            }
            else if (i < f)
            {
                i++;
                f++;
                pm = 1;
            }
            else
            {
                pm = -1;
            }
            
            ins = borders.begin();
            Vector3l step = pm * Vector3l::Unit(d);
            bool search = !borders.empty();
            for (long n = i; n != f; n += pm)
            {
                double param = (n - icoord(d)) / dcoord(d);
                
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
        
        Vector3l index = iindex;
        double param = 0.;
        for (std::map<double, Vector3l>::const_iterator b = borders.begin();
             b != borders.end(); ++b)
        {
            col(stride, index) += amount * (b->first - param);
            param = b->first;
            index += b->second;
        }
        BOOST_ASSERT_MSG(index == findex, "Final index incorrect");
    }
    return *this;
}

template<typename T, int N>
typename Field<T, N>::ArrayNXT Field<T, N>::match(const Field<T, N>& fld)
{
    ArrayNXT fldData = fld.data_;
    
    if (N == Eigen::Dynamic)
    {
        long fldRows = fldData.rows();
        long rows = data_.rows();
        
        if (fldRows > 1l && rows == 1l)
        {
            data_.conservativeResize(fldRows, data_.cols());
            data_ = data_.row(0).replicate(fldRows, 1l);
        }
        else if (rows > 1l && fldRows == 1l)
        {
            fldData.conservativeResize(rows, fldData.cols());
            fldData = fldData.row(0).replicate(rows, 1l);
        }
    }
    
    if (dom_ != fld.dom_)
    {
        if (dom_ == 0)
        {
            dom_ = fld.dom_;
            map_ = fld.map_;
            
            long fldCols = fldData.cols();
            if (fldCols > 1l)
            {
                data_.conservativeResize(data_.rows(), fldCols);
                data_ = data_.col(0).replicate(1l, fldCols);
            }
        }
        else if (fld.dom_ == 0)
        {
            long cols = data_.cols();
            if (cols > 1l)
            {
                fldData.conservativeResize(fldData.rows(), cols);
                fldData = fldData.col(0).replicate(1l, cols);
            }
        }
        else
        {
            BOOST_ASSERT_MSG(dom_ == fld.dom_,
                             "Field domains are not the same");
        }
    }
    BOOST_ASSERT_MSG(data_.size() == fldData.size(),
                     "Data sizes could not be matched");
    
    return fldData;
}

template<typename T, int N>
Field<T, N>& Field<T, N>::operator+=(const Field& fld)
{
    data_ += match(fld);
    return *this;
}

template<typename T, int N>
Field<T, N>& Field<T, N>::operator-=(const Field& fld)
{
    data_ -= match(fld);
    return *this;
}

template<typename T, int N>
Field<T, N>& Field<T, N>::operator*=(const Field& fld)
{
    data_ *= match(fld);
    return *this;
}

template<typename T, int N>
Field<T, N>& Field<T, N>::operator/=(const Field& fld)
{
    data_ /= match(fld);
    return *this;
}

template<typename T, int N>
Field<T, N> operator+(const Field<T, N>& fld1, const Field<T, N>& fld2)
{
    Field<T, N> fld(fld1);
    return fld += fld2;
}

template<typename T, int N>
Field<T, N> operator-(const Field<T, N>& fld1, const Field<T, N>& fld2)
{
    Field<T, N> fld(fld1);
    return fld -= fld2;
}

template<typename T, int N>
Field<T, N> operator*(const Field<T, N>& fld1, const Field<T, N>& fld2)
{
    Field<T, N> fld(fld1);
    return fld *= fld2;
}

template<typename T, int N>
Field<T, N> operator/(const Field<T, N>& fld1, const Field<T, N>& fld2)
{
    Field<T, N> fld(fld1);
    return fld /= fld2;
}

template<typename T, int N>
std::ostream& operator<<(std::ostream& os, const Field<T, N>& fld)
{
    return os << fld.data_;
}

template class Field<double, Eigen::Dynamic>;
template class Field<double, 1>;
template class Field<double, 2>;
template class Field<double, 3>;
template class Field<double, 4>;

template
Field<double, Eigen::Dynamic>::Field(const Domain*,
                                     const CellVolF<double, Eigen::Dynamic>&);
template Field<double, 1>::Field(const Domain*, const CellVolF<double, 1>&);
template Field<double, 2>::Field(const Domain*, const CellVolF<double, 2>&);
template Field<double, 3>::Field(const Domain*, const CellVolF<double, 3>&);
template Field<double, 4>::Field(const Domain*, const CellVolF<double, 4>&);

template Field<double, 1>& Field<double, 1>::transform(const TempAccumF& fun);
template Field<double, 1>& Field<double, 1>::transform(const FluxAccumF& fun);
template Field<double, 4>& Field<double, 4>::transform(const MultiAccumF& fun);
template Field<double, Eigen::Dynamic>&
Field<double, Eigen::Dynamic>::transform(const CumTempAccumF&);
template Field<double, Eigen::Dynamic>&
Field<double, Eigen::Dynamic>::transform(const CumFluxAccumF&);

template Field<double, Eigen::Dynamic>
operator+(const Field<double, Eigen::Dynamic>&,
          const Field<double, Eigen::Dynamic>&);
template Field<double, Eigen::Dynamic>
operator-(const Field<double, Eigen::Dynamic>&,
          const Field<double, Eigen::Dynamic>&);
template Field<double, Eigen::Dynamic>
operator*(const Field<double, Eigen::Dynamic>&,
          const Field<double, Eigen::Dynamic>&);
template Field<double, Eigen::Dynamic>
operator/(const Field<double, Eigen::Dynamic>&,
          const Field<double, Eigen::Dynamic>&);

template Field<double, 1> operator+(const Field<double, 1>&,
                                    const Field<double, 1>&);
template Field<double, 1> operator-(const Field<double, 1>&,
                                    const Field<double, 1>&);
template Field<double, 1> operator*(const Field<double, 1>&,
                                    const Field<double, 1>&);
template Field<double, 1> operator/(const Field<double, 1>&,
                                    const Field<double, 1>&);

template Field<double, 2> operator+(const Field<double, 2>&,
                                    const Field<double, 2>&);
template Field<double, 2> operator-(const Field<double, 2>&,
                                    const Field<double, 2>&);
template Field<double, 2> operator*(const Field<double, 2>&,
                                    const Field<double, 2>&);
template Field<double, 2> operator/(const Field<double, 2>&,
                                    const Field<double, 2>&);

template Field<double, 3> operator+(const Field<double, 3>&,
                                    const Field<double, 3>&);
template Field<double, 3> operator-(const Field<double, 3>&,
                                    const Field<double, 3>&);
template Field<double, 3> operator*(const Field<double, 3>&,
                                    const Field<double, 3>&);
template Field<double, 3> operator/(const Field<double, 3>&,
                                    const Field<double, 3>&);

template Field<double, 4> operator+(const Field<double, 4>&,
                                    const Field<double, 4>&);
template Field<double, 4> operator-(const Field<double, 4>&,
                                    const Field<double, 4>&);
template Field<double, 4> operator*(const Field<double, 4>&,
                                    const Field<double, 4>&);
template Field<double, 4> operator/(const Field<double, 4>&,
                                    const Field<double, 4>&);

template std::ostream& operator<<(std::ostream& os,
                                  const Field<double, Eigen::Dynamic>&);
template std::ostream& operator<<(std::ostream& os,
                                  const Field<double, 1>&);
template std::ostream& operator<<(std::ostream& os,
                                  const Field<double, 2>&);
template std::ostream& operator<<(std::ostream& os,
                                  const Field<double, 3>&);
template std::ostream& operator<<(std::ostream& os,
                                  const Field<double, 4>&);

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
    return ( n_ < 2 ? z_ : s_ / static_cast<S>(n_ - 1) );
}

template<typename S>
void Statistics<S>::add(const S& x)
{
    n_++;
    S d = x - m_;
    m_ += d / static_cast<S>(n_);
    s_ += d * (x - m_);
}

template<typename S>
std::ostream& operator<<(std::ostream& os, const Statistics<S>& stats)
{
    os << "Mean" << std::endl;
    os << stats.mean() << std::endl << std::endl;
    os << "Variance" << std::endl;
    os << stats.variance();
    return os;
}

template class Statistics< Field<double, Eigen::Dynamic> >;
template class Statistics< Field<double, 1> >;
template class Statistics< Field<double, 2> >;
template class Statistics< Field<double, 3> >;
template class Statistics< Field<double, 4> >;

template std::ostream&
operator<<(std::ostream& os,
           const Statistics< Field<double, Eigen::Dynamic> >&);
template std::ostream&
operator<<(std::ostream& os,
           const Statistics< Field<double, 1> >&);
template std::ostream&
operator<<(std::ostream& os,
           const Statistics< Field<double, 2> >&);
template std::ostream&
operator<<(std::ostream& os,
           const Statistics< Field<double, 3> >&);
template std::ostream&
operator<<(std::ostream& os,
           const Statistics< Field<double, 4> >&);
