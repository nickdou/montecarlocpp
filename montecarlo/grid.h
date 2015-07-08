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
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <exception>
#include <cmath>


class Grid {
protected:
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
//public:
//    class OutOfRange : public std::runtime_error {
//    public:
//        explicit OutOfRange(const char* msg)
//        : std::runtime_error(msg) {}
//        explicit OutOfRange(const std::stringstream& stream)
//        : std::runtime_error(stream.str()) {}
//    };
private:
    Vector3d o_;
    Matrix3d mat_, inv_;
    Vector3l div_, shape_, max_;
    double vol_;
    int dim_, dir_;
    int findDir() const {
        Vector3l::Index d;
        div_.maxCoeff(&d);
        BOOST_ASSERT_MSG(d >= 0 && d < 3, "Index out of range");
        return static_cast<int>(d);
    }
//    template<bool Check>
//    Vector3d getCoord(const Vector3d& pos) const {
//        Vector3d norm = inv_ * (pos - o_);
//        if (Check) {
//            static const double tol = 100*Dbl::epsilon();
//            bool isInside = (norm.array() >= -tol).all() &&
//                            (norm.array() <= 1. + tol).all();
//            if(!isInside) {
//#pragma omp critical
//                {
//                    std::stringstream what;
//                    Eigen::Matrix<double, 3, 4> output;
//                    output << pos, norm, norm / Dbl::epsilon(),
//                              (Vector3d::Ones() - norm) / Dbl::epsilon();
//                    Eigen::IOFormat fmt(9, 16);
//                    what << "Point outside of grid" << std::endl;
//                    what << output.transpose().format(fmt);
//                    throw(OutOfRange(what));
//                }
//            }
//            
//        }
//        return div_.cast<double>().cwiseProduct(norm);
//    }
    bool checkInside(const Vector3d& norm) const {
        static const double tol = 100*Dbl::epsilon();
        return (norm.array() >= -tol).all() && (norm.array() <= 1. + tol).all();
    }
    Vector3l coordToIndex(const Vector3d& coord) const {
        Vector3l vec(floor(coord(0)), floor(coord(1)), floor(coord(2)));
        return vec.cwiseMax(0).cwiseMin(max_);
    }
    static long floor(double d) {
        return static_cast<long>(std::floor(d));
    }
public:
    Grid() : vol_(0.) {}
    Grid(const Vector3d& o, const Matrix3d& mat, const Vector3l& div)
    : o_(o), mat_(mat), inv_(mat.inverse()),
    div_(div), shape_(div.cwiseMax(1)),
    max_(div.cwiseMax(1) - Vector3l::Ones()),
    vol_(mat.determinant()),
    dim_(static_cast<int>( (div.array() > 0).count() )), dir_(0)
    {
        if (dim_ == 1) dir_ = findDir();
        BOOST_ASSERT_MSG(dim_ <= 3, "Dimensionality must be between 0 and 3");
        BOOST_ASSERT_MSG(std::abs(vol_) >= Dbl::min(),
                         "Grid matrix is singular");
    }
    bool isInit() const { return vol_ != 0.; }
    const Vector3d& origin() const { return o_; }
    const Matrix3d& matrix() const { return mat_; }
    const Vector3l& shape() const { return shape_; }
    double volume() const { return vol_; }
    template<typename T>
    bool accumulate(const Vector3d& begin, const Vector3d& end,
                    Data<T>& data, T quant) const
    {
        if (dim_ == 0) {
            *data.origin() += quant;
            return true;
        }
        
        Vector3d bNorm = inv_ * (begin - o_);
        Vector3d eNorm = inv_ * (end - o_);
        if (!checkInside(bNorm) || !checkInside(eNorm)) return false;
        
        Vector3d bCoord = div_.cast<double>().cwiseProduct(bNorm);
        Vector3d eCoord = div_.cast<double>().cwiseProduct(eNorm);
        Vector3d dCoord = eCoord - bCoord;
        if (dCoord.norm() < Dbl::min()) return true;
        
        Vector3l bIndex = coordToIndex(bCoord);
        Vector3l eIndex = coordToIndex(eCoord);
        
        if (dim_ == 1) {
            long b = bIndex(dir_);
            long e = eIndex(dir_);
            Collection bColl(0,0,0);
            Collection eColl(0,0,0);
            bColl[dir_] = b;
            eColl[dir_] = e;
            switch (b) {
                case 0: break;
                case 1: break;
                case 2: break;
                case 3: break;
                default: break;
            }
            if (b == e) {
                data(bColl) += quant;
                return true;
            }
            T cellQuant = quant / std::abs(dCoord(dir_));
            int pm;
            if (b < e) {
                data(bColl) += cellQuant * (1 + b - bCoord(dir_));
                data(eColl) += cellQuant * (eCoord(dir_) - e);
                pm = 1;
            } else {
                data(bColl) += cellQuant * (bCoord(dir_) - b);
                data(eColl) += cellQuant * (1 + e - eCoord(dir_));
                pm = -1;
            }
            Collection iColl(0,0,0);
            for (long i = b + pm; i != e; i += pm) {
                iColl[dir_] = i;
                data(iColl) += cellQuant;
            }
            return true;
        }
        {
            typedef std::map<double, Vector3l> Map;
            Map borderMap;
            Map::iterator ins = borderMap.begin();
            for (int d = 0; d < 3; d++) {
                long b = bIndex(d);
                long e = eIndex(d);
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
                for (long i = b; i != e; i += pm) {
                    double param = (i - bCoord(d)) / dCoord(d);
//                    if (param < 0. || param > 1.) {
//#pragma omp critical
//                        {
//                            std::stringstream what;
//                            what << "Parameter out of range" << std::endl;
//                            what << std::setprecision(9) << std::setw(16);
//                            what << param;
//                            throw(OutOfRange(what));
//                        }
//                    }
                    if (param < 0. || param > 1.) return false;
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
            
            Vector3l index = bIndex;
            double param = 0.;
            for (Map::const_iterator border = borderMap.begin();
                 border != borderMap.end(); ++border)
            {
                data( Collection(index) ) += quant * (border->first - param);
                param = border->first;
                index += border->second;
            }
            BOOST_ASSERT_MSG(index == eIndex, "Final index incorrect");
            return true;
        }
    }
};

#endif
