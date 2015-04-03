//
//  tools.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/28/15.
//
//

#ifndef montecarlocpp_tools_h
#define montecarlocpp_tools_h

#include "random.h"
#include "constants.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <boost/assert.hpp>
#include <algorithm>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>

template<typename Scalar>
bool isApprox(const Scalar& a, const Scalar& b) {
    typedef std::numeric_limits<Scalar> Lim;
    Scalar tol = 100*Lim::epsilon();
    Scalar max = std::max(std::abs(a), std::abs(b));
    Scalar rhs = std::max(Lim::min(), tol * max);
    return std::abs(a - b) <= rhs;
}

template<typename Derived>
void extractArray(std::istream& is, const Eigen::DenseBase<Derived>& data) {
    typedef typename Derived::Scalar Scalar;
    typedef typename Derived::Index Index;
    is >> std::ws;
    std::string line;
    Index i = 0;
    while (i < data.rows() && std::getline(is, line)) {
        std::stringstream ss(line);
        for (Index j = 0; j < data.cols(); ++j) {
            Scalar element;
            ss >> element;
            const_cast<Scalar&>(data(i, j)) = element;
            BOOST_ASSERT_MSG(ss, "Array extraction failed");
        }
        i++;
    }
    BOOST_ASSERT_MSG(i == data.rows(), "Array extraction failed");
}

inline Eigen::Vector3d drawIso(Rng& gen) {
    static const UniformDist distOne(-1., 1.); // [-1, 1)
    double cosTheta = distOne(gen);
    double sinTheta = std::sqrt(1. - cosTheta*cosTheta);
    double phi = PI * distOne(gen);
    return Eigen::Vector3d(sinTheta * std::cos(phi),
                           sinTheta * std::sin(phi),
                           cosTheta);
}

template<bool bidir>
Eigen::Vector3d drawAniso(Rng& gen) {
    static const UniformDist distOne(-1., 1.);  // [-1, 1)
    double r = distOne(gen);
    int sign = (bidir ? (r < 0. ? -1 : 1) : 1);
    double sinSqTheta = std::abs(r); // [0, 1]
    double sinTheta = std::sqrt(sinSqTheta);
    double cosTheta = sign*std::sqrt(1. - sinSqTheta);
    double phi = PI * distOne(gen);
    return Eigen::Vector3d(sinTheta * std::cos(phi),
                           sinTheta * std::sin(phi),
                           cosTheta);
}

inline Eigen::Matrix3d reflMatrix(const Eigen::Vector3d& n) {
    return Eigen::Matrix3d::Identity() - 2.*(n*n.transpose());
}

inline Eigen::Matrix3d rotMatrix(const Eigen::Vector3d& n) {
    typedef Eigen::Quaternion<double> Quatd;
    return Quatd::FromTwoVectors(Eigen::Vector3d::UnitZ(), n).matrix();
}

inline Eigen::Matrix3d catMatrix(const Eigen::Vector3d& i,
                                 const Eigen::Vector3d& j,
                                 const Eigen::Vector3d& k)
{
    return (Eigen::Matrix3d() << i, j, k).finished();
}

inline Eigen::Matrix3d gridMatrix() {
    return Eigen::Matrix3d::Identity();
}

inline Eigen::Matrix3d gridMatrix(const Eigen::Vector3d& i) {
    typedef Eigen::Quaternion<double> Quatd;
    Quatd q = Quatd::FromTwoVectors(Eigen::Vector3d::UnitX(), i);
    return i.norm() * q.matrix();
}

inline Eigen::Matrix3d gridMatrix(const Eigen::Vector3d& i,
                                  const Eigen::Vector3d& j)
{
    Eigen::Vector3d k = std::min(i.norm(), j.norm()) * i.cross(j).normalized();
    return catMatrix(i, j, k);
}

inline Eigen::Matrix3d gridMatrix(const Eigen::Vector3d& i,
                                  const Eigen::Vector3d& j,
                                  const Eigen::Vector3d& k)
{
    return catMatrix(i, j, k);
}

#endif
