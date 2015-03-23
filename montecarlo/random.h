//
//  random.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 3/1/15.
//
//

#ifndef montecarlocpp_random_h
#define montecarlocpp_random_h

#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_device.hpp>

typedef boost::random::random_device Dev;
typedef boost::random::mt19937 Rng;
typedef boost::random::uniform_01<double> UniformDist01;
typedef boost::random::uniform_real_distribution<double> UniformDist;
typedef boost::random::uniform_int_distribution<unsigned long> UniformIntDist;
typedef boost::random::discrete_distribution<unsigned long, double> DiscreteDist;

#endif
