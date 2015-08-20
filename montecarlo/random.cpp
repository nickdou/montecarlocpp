//
//  random.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 8/6/15.
//
//

#include "random.h"
#include "constants.h"
#include <Eigen/Core>
#include <cmath>

Eigen::Vector3d drawIso(Rng& gen)
{
    static const UniformDist distOne(-1., 1.); // [-1, 1)
    
    double cosTheta = distOne(gen);
    double sinTheta = std::sqrt(1. - cosTheta*cosTheta);
    double phi = PI * distOne(gen);
    
    return Eigen::Vector3d(sinTheta * std::cos(phi),
                           sinTheta * std::sin(phi),
                           cosTheta);
}

Eigen::Vector3d drawAniso(Rng& gen, bool bidir)
{
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