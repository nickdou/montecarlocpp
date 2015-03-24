//
//  phonon.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 1/29/15.
//
//

#ifndef __montecarlocpp__phonon__
#define __montecarlocpp__phonon__

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <vector>

class Phonon {
protected:
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::ParametrizedLine<double, 3> Line3d;
public:
    class Prop {
        long w_, p_;
    public:
        Prop() {}
        Prop(long omega, long pol) : w_(omega), p_(pol) {}
        long w() const { return w_; }
        long p() const { return p_; }
    };
private:
    Prop prop_;
    bool alive_, sign_;
    Vector3d x_, d_;
    Line3d l_;
    long nscat_;
public:
    Phonon() {}
    Phonon(const Prop& prop, const Vector3d& pos, const Vector3d& dir, bool s)
    : prop_(prop), alive_(true), sign_(s),
    x_(pos), d_(dir.normalized()), l_(pos, dir.normalized()), nscat_(0)
    {}
    virtual ~Phonon() {}
    const Prop& prop() const { return prop_; }
    void prop(const Prop& p) { prop_ = p; }
    bool alive() const { return alive_; }
    void kill() { alive_ = false; }
    int sign() const { return sign_? 1 : -1; }
    const Vector3d& pos() const { return x_; }
    const Vector3d& dir() const { return d_; }
    const Line3d& line() const { return l_; }
    long nscat() const { return nscat_; }
    virtual void pos(const Vector3d& newPos) {
        x_ = newPos;
        l_ = Line3d(x_, d_);
    }
    template<bool Scatter>
    void dir(const Vector3d& newDir) {
        d_ = newDir.normalized();
        l_ = Line3d(x_, d_);
        if (Scatter) nscat_++;
    }
    void move(double dist) {
        pos(l_.pointAt(dist));
    }
};

class TrkPhonon : public Phonon {
public:
    typedef std::vector<Vector3d> Traj;
private:
    Traj traj_;
public:
    TrkPhonon() : Phonon() {}
    TrkPhonon(const Prop& p, const Vector3d& pos, const Vector3d& dir, bool s) :
    Phonon(p, pos, dir, s), traj_() {
        traj_.push_back(pos);
    }
    TrkPhonon(const Phonon& phn) : Phonon(phn), traj_() {
        traj_.push_back(phn.pos());
    }
    ~TrkPhonon() {}
    const Traj& traj() { return traj_; }
    const Vector3d& pos() const { return Phonon::pos(); }
    void pos(const Vector3d& newPos) {
        Phonon::pos(newPos);
        traj_.push_back(newPos);
    }
};

#endif
