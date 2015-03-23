//
//  grid.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/28/15.
//
//

#ifndef __montecarlocpp__grid__
#define __montecarlocpp__grid__

#include "data.h"
#include "constants.h"
#include <Eigen/LU>
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <map>
#include <iostream>
#include <cmath>

class Grid {
protected:
    typedef Eigen::Matrix<unsigned long, 3, 1> Vector3ul;
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
private:
    Vector3d o_;
    Matrix3d mat_, inv_;
    Vector3ul div_, max_;
    double vol_;
    unsigned int dim_;
    unsigned int findDir() const {
        Vector3ul::Index d;
        div_.maxCoeff(&d);
        BOOST_ASSERT_MSG(d >= 0 && d < 3, "Index out of range");
        return static_cast<unsigned int>(d);
    }
    template<bool Check>
    Vector3d getCoord(const Vector3d& pos) const {
        Vector3d norm = inv_ * (pos - o_);
        if (Check) {
            static const double margin = 3*Dbl::epsilon();
            bool isInside = (norm.array() >= -margin).all() &&
                            (norm.array() <= 1. + margin).all();
            if(!isInside) {
                using std::cout;
                using std::endl;
                cout << pos.transpose() << endl;
                cout << norm.transpose() << endl;
                cout << norm.transpose().array() / Dbl::epsilon() << endl;
                cout << (1. - norm.transpose().array()) / Dbl::epsilon() << endl;
            }
            BOOST_ASSERT_MSG(isInside, "Point outside of grid");
        }
        return div_.cast<double>().cwiseProduct(norm);
    }
    Vector3ul getIndex(const Vector3d& coord) const {
        Vector3l vec(floor(coord(0)), floor(coord(1)), floor(coord(2)));
        return vec.cwiseMax(0).cast<unsigned long>().cwiseMin(max_);
    }
    static long floor(double d) {
        return static_cast<long>(std::floor(d));
    }
public:
    Grid() : vol_(0.) {}
    Grid(const Vector3d& o, const Matrix3d& mat, const Vector3ul& div)
    : o_(o), mat_(mat), inv_(mat.inverse()),
    div_(div), max_(div.cwiseMax(1) - Vector3ul::Ones()),
    vol_(mat.determinant() / div.cwiseMax(1).prod()),
    dim_(static_cast<unsigned int>( (div.array() > 0).count() ))
    {
        BOOST_ASSERT_MSG(dim_ <= 3, "Dimensionality must be between 0 and 3");
        BOOST_ASSERT_MSG(std::abs(vol_) >= Dbl::min(),
                         "Grid matrix is singular");
    }
    bool isInit() const { return vol_ != 0.; }
    const Vector3d& origin() const { return o_; }
    const Matrix3d& matrix() const { return mat_; }
    const Vector3ul& shape() const { return div_; }
    double cellVol() const { return vol_; }
    template<typename T>
    void accumulate(const Vector3d& begin, const Vector3d& end,
                    Data<T>& data, T quant) const
    {
        if (dim_ == 0) {
            *data.origin() += quant;
            return;
        }
        
        Vector3d bCoord = getCoord<true>(begin);
        Vector3d eCoord = getCoord<true>(end);
        Vector3d dCoord = eCoord - bCoord;
        if (dCoord.norm() < Dbl::min()) return;
        Vector3ul bIndex = getIndex(bCoord);
        Vector3ul eIndex = getIndex(eCoord);
        
        if (dim_ == 1) {
            static unsigned int d = findDir();
            unsigned long b = bIndex(d);
            unsigned long e = eIndex(d);
            Collection bColl(0,0,0);
            Collection eColl(0,0,0);
            bColl[d] = b;
            eColl[d] = e;
            if (b == e) {
                data(bColl) += quant;
                return;
            }
            T cellQuant = quant / std::abs(dCoord(d));
            int pm;
            if (b < e) {
                data(bColl) += cellQuant * (1 + b - bCoord(d));
                data(eColl) += cellQuant * (eCoord(d) - e);
                pm = 1;
            } else {
                data(bColl) += cellQuant * (bCoord(d) - b);
                data(eColl) += cellQuant * (1 + e - eCoord(d));
                pm = -1;
            }
            Collection iColl(0,0,0);
            for (unsigned long i = b + pm; i != e; i += pm) {
                iColl[d] = i;
                data(iColl) += cellQuant;
            }
            return;
        }
        {
            typedef std::map<double, Vector3l> Map;
            Map borderMap;
            Map::iterator ins = borderMap.begin();
            for (unsigned int d = 0; d < 3; d++) {
                unsigned long b = bIndex(d);
                unsigned long e = eIndex(d);
                int pm;
                if (b == e) {
                    continue;
                } else if (b < e) {
                    b++;
                    e++;
                    pm = 1;
                } else {
                    pm = -1;
                }
                
                if (std::abs(dCoord(d)) < Dbl::min()) continue;
                ins = borderMap.begin();
                Vector3l step = pm * Vector3l::Unit(d);
                bool search = !borderMap.empty();
                for (unsigned long i = b; i != e; i += pm) {
                    double param = (i - bCoord(d)) / dCoord(d);
                    BOOST_ASSERT_MSG(param >= 0. && param <= 1.,
                                     "Parameter out of range");
                    if (search) {
                        Map::iterator found = borderMap.find(param);
                        if (found != borderMap.end()) {
                            found->second += step;
                            ins = found;
                            continue;
                        }
                    }
                    ins = borderMap.insert(ins, Map::value_type(param, step));
                }
            }
            borderMap.insert(ins, Map::value_type(1., Vector3l::Zero()));
            
            Vector3ul index = bIndex;
            double param = 0.;
            for (Map::const_iterator border = borderMap.begin();
                 border != borderMap.end(); ++border)
            {
                data( Collection(index) ) += quant * (border->first - param);
                param = border->first;
                index = (index.cast<long>() +
                         border->second).cast<unsigned long>();
            }
            BOOST_ASSERT_MSG(index == eIndex, "Final index incorrect");
        }
    }
};

#endif
