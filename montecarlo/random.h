//
//  random.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 3/1/15.
//
//

#ifndef montecarlocpp_random_h
#define montecarlocpp_random_h

#include <Eigen/Core>

#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>

typedef boost::random::random_device Dev;
typedef boost::random::mt19937 Rng;
typedef boost::random::uniform_01<double> UniformDist01;
typedef boost::random::uniform_real_distribution<double> UniformDist;
typedef boost::random::uniform_int_distribution<long> UniformIntDist;
typedef boost::random::discrete_distribution<long, double> DiscreteDist;

Eigen::Vector3d drawIso(Rng& gen);

Eigen::Vector3d drawAniso(Rng& gen, bool bidir);

#endif
