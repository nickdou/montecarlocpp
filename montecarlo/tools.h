//
//  tools.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/28/15.
//
//

#ifndef montecarlocpp_tools_h
#define montecarlocpp_tools_h

//#include "random.h"
//#include "constants.h"
//#include <Eigen/Geometry>
//#include <Eigen/Core>
//#include <boost/assert.hpp>
//#include <algorithm>
//#include <sstream>
//#include <string>
//#include <cmath>
//#include <limits>
//
//inline Eigen::Matrix3d reflMatrix(const Eigen::Vector3d& n) {
//    return Eigen::Matrix3d::Identity() - 2.*(n*n.transpose());
//}
//
//inline Eigen::Matrix3d rotMatrix(const Eigen::Vector3d& n) {
//    typedef Eigen::Quaternion<double> Quatd;
//    return Quatd::FromTwoVectors(Eigen::Vector3d::UnitZ(), n).matrix();
//}
//
//inline Eigen::Matrix3d catMatrix(const Eigen::Vector3d& i,
//                                 const Eigen::Vector3d& j,
//                                 const Eigen::Vector3d& k)
//{
//    return (Eigen::Matrix3d() << i, j, k).finished();
//}
//
//inline Eigen::Matrix3d gridMatrix() {
//    return Eigen::Matrix3d::Identity();
//}
//
//inline Eigen::Matrix3d gridMatrix(const Eigen::Vector3d& i) {
//    typedef Eigen::Quaternion<double> Quatd;
//    Quatd q = Quatd::FromTwoVectors(Eigen::Vector3d::UnitX(), i);
//    return i.norm() * q.matrix();
//}
//
//inline Eigen::Matrix3d gridMatrix(const Eigen::Vector3d& i,
//                                  const Eigen::Vector3d& j)
//{
//    Eigen::Vector3d k = std::min(i.norm(), j.norm()) * i.cross(j).normalized();
//    return catMatrix(i, j, k);
//}
//
//inline Eigen::Matrix3d gridMatrix(const Eigen::Vector3d& i,
//                                  const Eigen::Vector3d& j,
//                                  const Eigen::Vector3d& k)
//{
//    return catMatrix(i, j, k);
//}

#endif
