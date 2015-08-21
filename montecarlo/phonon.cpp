//
//  phonon.cpp
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/29/15.
//
//

#include "phonon.h"
#include "constants.h"
#include <boost/assert.hpp>

Phonon::Prop::Prop()
{}

Phonon::Prop::Prop(long omega, long pol)
: w_(omega), p_(pol)
{}

long Phonon::Prop::w() const
{
    return w_;
}

long Phonon::Prop::p() const
{
    return p_;
}

Phonon::Phonon()
{}

Phonon::Phonon(bool sign, const Prop& prop,
               const Vector3d& pos, const Vector3d& dir)
: alive_(true), sign_(sign), prop_(prop), pos_(pos), dir_(dir.normalized()),
line_(pos, dir.normalized()), time_(0.), scatNext_(0.), nscat_(0l)
{}

Phonon::~Phonon()
{}

bool Phonon::alive() const
{
    return alive_;
}

void Phonon::kill()
{
    alive_ = false;
}

int Phonon::sign() const
{
    return sign_? 1 : -1;
}

const Phonon::Prop& Phonon::prop() const
{
    return prop_;
}

void Phonon::prop(const Prop& newProp)
{
    prop_ = newProp;
}

const Vector3d& Phonon::pos() const
{
    return pos_;
}

const Vector3d& Phonon::dir() const
{
    return dir_;
}

const Eigen::ParametrizedLine<double, 3>& Phonon::line() const
{
    return line_;
}

void Phonon::pos(const Vector3d& newPos)
{
    pos_ = newPos;
    line_ = Eigen::ParametrizedLine<double, 3>(pos_, dir_);
}

void Phonon::dir(const Vector3d& newDir, bool scatter)
{
    dir_ = newDir.normalized();
    line_ = Eigen::ParametrizedLine<double, 3>(pos_, dir_);
    if (scatter) nscat_++;
}

void Phonon::move(double distance, double vel)
{
    BOOST_ASSERT_MSG(distance <= scatNext_, "Movement distance too large");
    scatNext_ -= distance;
    
    BOOST_ASSERT_MSG(vel > Dbl::min(), "Velocity must be nonzero");
    time_ += distance / vel;
    
    pos(line_.pointAt(distance));
}

double Phonon::time() const
{
    return time_;
}

double Phonon::scatNext() const
{
    return scatNext_;
}

long Phonon::nscat() const
{
    return nscat_;
}

void Phonon::scatNext(double distance)
{
    BOOST_ASSERT_MSG(scatNext_ == 0., "Cannot reset scattering distance");
    scatNext_ = distance;
}

TrkPhonon::TrkPhonon()
: Phonon()
{}

TrkPhonon::TrkPhonon(bool sign, const Prop& prop,
                     const Vector3d& pos, const Vector3d& dir)
: Phonon(sign, prop, pos, dir), traj_()
{
    traj_.push_back(pos);
}

TrkPhonon::TrkPhonon(const Phonon& phn)
: Phonon(phn), traj_()
{
    traj_.push_back(phn.pos());
}

TrkPhonon::~TrkPhonon()
{}

const Vector3d& TrkPhonon::pos() const
{
    return Phonon::pos();
}

void TrkPhonon::pos(const Vector3d& newPos)
{
    Phonon::pos(newPos);
    traj_.push_back(newPos);
}

TrkPhonon::Array3Xd TrkPhonon::trajectory() const
{
    Array3Xd arr(3, traj_.size());
    Array3Xd::Index i = 0;
    for (std::vector<Vector3d>::const_iterator x = traj_.begin();
         x != traj_.end(); ++x)
    {
        arr.col(i++) = *x;
    }
    return arr;
}



