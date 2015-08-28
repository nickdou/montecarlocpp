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

using Eigen::Vector3d;

typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;

class Phonon
{
    
public:
    class Prop
    {
    private:
        long w_, p_;
        
    public:
        Prop();
        Prop(long omega, long pol);
        long w() const;
        long p() const;
    };
    
private:
    bool alive_, sign_;
    Prop prop_;
    Vector3d pos_, dir_;
    Eigen::ParametrizedLine<double, 3> line_;
    double time_, scatNext_;
    long nscat_;
    
public:
    Phonon();
    Phonon(bool sign, const Prop& prop,
           const Vector3d& pos, const Vector3d& dir);
    virtual ~Phonon();
    
    bool alive() const;
    void kill();
    
    int sign() const;
    
    const Prop& prop() const;
    void prop(const Prop& p);
    
    virtual const Vector3d& pos() const;
    const Vector3d& dir() const;
    const Eigen::ParametrizedLine<double, 3>& line() const;
    
    virtual void pos(const Vector3d& newPos);
    void dir(const Vector3d& newDir, bool scatter);
    void move(double distance, double vel);
    
    double time() const;
    double scatNext() const;
    long nscat() const;
    
    void scatNext(double distance);
};

class TrkPhonon : public Phonon
{
private:
    std::vector<Vector3d> traj_;
    
public:
    TrkPhonon();
    TrkPhonon(bool sign, const Prop& prop,
              const Vector3d& pos, const Vector3d& dir);
    TrkPhonon(const Phonon& phn);
    ~TrkPhonon();
    
    const Vector3d& pos() const;
    void pos(const Vector3d& newPos);
    
    Matrix3Xd trajectory() const;
};

#endif
