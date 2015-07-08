//
//  domain.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/5/15.
//
//

#ifndef __montecarlocpp__domain__
#define __montecarlocpp__domain__

#include "grid.h"
#include "data.h"
#include "boundary.h"
#include "phonon.h"
#include "tools.h"
#include "constants.h"
#include <Eigen/Core>
#include <boost/fusion/algorithm/iteration.hpp>
#include <boost/fusion/sequence/intrinsic.hpp>
#include <boost/fusion/container/vector/vector40.hpp>
#include <boost/fusion/container/vector/vector10.hpp>
#include <boost/assert.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

namespace fusion = boost::fusion;
namespace result_of = boost::fusion::result_of;

class Subdomain {
protected:
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
public:
    typedef std::vector<const Boundary*> BdryPtrs;
    typedef std::vector<const Emitter*> EmitPtrs;
private:
    BdryPtrs bdryPtrs_;
    EmitPtrs emitPtrs_;
protected:
    void addBdry(Boundary* bdry) {
        bdry->sdom(this);
        bdryPtrs_.push_back(bdry);
    }
    template<typename S>
    void addBdry(EmitBoundary<S>* bdry) {
        addBdry(static_cast<Boundary*>(bdry));
        if (bdry->emitWeight() != 0.) {
            emitPtrs_.push_back(static_cast<Emitter*>(bdry));
        }
    }
    friend class AddBdryF;
    class AddBdryF {
    private:
        Subdomain* sdom_;
    public:
        AddBdryF(Subdomain* sdom) : sdom_(sdom) {}
        template<typename B>
        void operator()(B& bdry) const {
            sdom_->addBdry(&bdry);
        }
    };
    Grid grid_;
public:
    Subdomain() {}
    Subdomain(const Grid& grid) : grid_(grid) {}
    Subdomain(const Subdomain& sdom) : grid_(sdom.grid_) {}
    Subdomain& operator=(const Subdomain& sdom) {
        grid_ = sdom.grid_;
        return *this;
    }
    virtual ~Subdomain() {}
    bool isInit() const {
        if (!grid_.isInit() || bdryPtrs_.size() == 0) {
            return false;
        }
        for (BdryPtrs::const_iterator b = bdryPtrs_.begin();
             b != bdryPtrs_.end(); ++b)
        {
            if (!(*b)->isInit()) return false;
        }
        return true;
    }
    bool isInside(const Vector3d& pos) const {
        for (BdryPtrs::const_iterator b = bdryPtrs_.begin();
             b != bdryPtrs_.end(); ++b)
        {
            if ((*b)->distance(pos) < 0.) return false;
        }
        return true;
    }
    const BdryPtrs& bdryPtrs() const { return bdryPtrs_; }
    const EmitPtrs& emitPtrs() const { return emitPtrs_; }
    template<typename T>
    Data<T> initData(const T& value) const {
        return Data<T>( Collection(grid_.shape()), value );
    }
    template<typename T>
    bool accumulate(const Vector3d& begin, const Vector3d& end,
                    Data<T>& data, T quant) const
    {
        return grid_.accumulate(begin, end, data, quant);
    }
    virtual double sdomVol() const = 0;
    virtual double cellVol(const Vector3l& index) const = 0;
    virtual const Boundary* advect(Phonon& phn, double& scatDist) const {
        double minDist = scatDist;
        const Boundary* newBdry = 0;
        for (BdryPtrs::const_iterator b = bdryPtrs_.begin();
             b != bdryPtrs_.end(); ++b)
        {
            if ((*b)->normal().dot(phn.dir()) >= 0.) continue;
            double dist = (*b)->distance(phn);
            BOOST_ASSERT_MSG(dist > 0. ||
                             (*b)->projection(phn.pos()).isApprox(phn.pos()),
                             "Phonon escaped from domain");
            if (dist < minDist) {
                minDist = dist;
                newBdry = *b;
            }
        }
        phn.move(minDist);
        scatDist -= minDist;
        return newBdry;
    }
};

class EmitSubdomain : public Subdomain, public Emitter {
private:
    Vector3d gradT_;
    Matrix3d rot_;
public:
    EmitSubdomain() : Subdomain() {}
    EmitSubdomain(const Grid& grid, const Vector3d& gradT)
    : Subdomain(grid), gradT_(gradT), rot_(rotMatrix(gradT.normalized())) {}
    EmitSubdomain(const EmitSubdomain& sdom)
    : Subdomain(sdom), gradT_(sdom.gradT_), rot_(sdom.rot_) {}
    virtual ~EmitSubdomain() {}
    const Subdomain* emitSdom() const {
        return this;
    }
    const Boundary* emitBdry() const { return 0; }
    double emitWeight() const {
        return 2 * sdomVol() * gradT_.norm();
    }
private:
    Vector3d drawDir(Rng& gen) const {
        return rot_ * drawAniso<true>(gen);
    }
    bool emitSign(const Vector3d&, const Vector3d& dir) const {
        return dir.dot(gradT_) < 0.;
    }
};

template<typename Back, typename Left, typename Bottom,
         typename Front = Back, typename Right = Left, typename Top = Bottom>
class Parallelepiped : public EmitSubdomain {
public:
    typedef fusion::vector6<Back, Left, Bottom, Front, Right, Top> BdryCont;
private:
    BdryCont bdryCont_;
    void init() {
        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(),
                         "Volume too small, check vector order");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
public:
    Parallelepiped() : EmitSubdomain() {}
    Parallelepiped(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
                   const Vector3d& gradT = Vector3d::Zero(),
                   const double TBack   = 0., const double TLeft  = 0.,
                   const double TBottom = 0., const double TFront = 0.,
                   const double TRight  = 0., const double TTop   = 0.)
    : EmitSubdomain(Grid(o, mat, div), gradT), bdryCont_(
      Back(o,            Parallelogram(mat.col(1), mat.col(2)), TBack  ),
      Left(o,            Parallelogram(mat.col(2), mat.col(0)), TLeft  ),
    Bottom(o,            Parallelogram(mat.col(0), mat.col(1)), TBottom),
     Front(o+mat.col(0), Parallelogram(mat.col(2), mat.col(1)), TFront ),
     Right(o+mat.col(1), Parallelogram(mat.col(0), mat.col(2)), TRight ),
       Top(o+mat.col(2), Parallelogram(mat.col(1), mat.col(0)), TTop   ))
    {
        init();
    }
    Parallelepiped(const Parallelepiped& para)
    : EmitSubdomain(para), bdryCont_(para.bdryCont_)
    {
        init();
    }
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry() {
        return fusion::at_c<I>(bdryCont_);
    }
    double sdomVol() const {
        return grid_.volume();
    }
    double cellVol(const Vector3l&) const {
        return grid_.volume() / grid_.shape().prod();
    }
private:
    Vector3d drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        Vector3d coord(dist(gen), dist(gen), dist(gen));
        return grid_.origin() + grid_.matrix()*coord;
    }
};

template<typename Back, typename Left, typename Bottom,
         typename Diag, typename Top = Bottom>
class TriangularPrism : public EmitSubdomain {
public:
    typedef fusion::vector5<Back, Left, Bottom, Diag, Top> BdryCont;
private:
    BdryCont bdryCont_;
    void init() {
        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(),
                         "Volume too small, check vector order");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
public:
    TriangularPrism() : EmitSubdomain() {}
    TriangularPrism(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
                    const Vector3d& gradT = Vector3d::Zero(),
                    const double TBack   = 0., const double TLeft = 0.,
                    const double TBottom = 0., const double TDiag = 0.,
                    const double TTop    = 0.)
    : EmitSubdomain(Grid(o, mat, div), gradT), bdryCont_(
  Back(o,            Parallelogram(mat.col(1), mat.col(2)),              TBack),
  Left(o,            Parallelogram(mat.col(2), mat.col(0)),              TLeft),
Bottom(o,                 Triangle(mat.col(0), mat.col(1)),            TBottom),
  Diag(o+mat.col(0), Parallelogram(mat.col(2), mat.col(1) - mat.col(0)), TDiag),
   Top(o+mat.col(2),      Triangle(mat.col(1), mat.col(0)),              TTop ))
    {
        init();
    }
    TriangularPrism(const TriangularPrism& prism)
    : EmitSubdomain(prism), bdryCont_(prism.bdryCont_)
    {
        init();
    }
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry() {
        return fusion::at_c<I>(bdryCont_);
    }
    double sdomVol() const {
        return grid_.volume() / 2.;
    }
    double cellVol(const Vector3l& index) const {
        Vector3l shape = grid_.shape();
        
        Eigen::Vector2d index2 = index.head<2>().cast<double>();
        Eigen::Vector2d shape2 = shape.head<2>().cast<double>();
        double f0 = 1. - index2.cwiseQuotient(shape2).sum();
        if (f0 <= 0.) return 0.;
        
        Eigen::Matrix<long, 2, 1> corner(1, 1);
        double f1 = f0 - corner.cast<double>().cwiseQuotient(shape2).sum();
        if (f1 >= 0.) return grid_.volume() / shape.prod();
        
        typedef Eigen::Matrix<long, 2, 2> Points;
        static const Points pts = Points::Identity();
        double frac = std::pow(f0, 2);
        for (int i = 0; i < Points::ColsAtCompileTime; i++) {
            corner = pts.col(i);
            int sign = std::pow(-1, corner.sum());
            double f = f0 - corner.cast<double>().cwiseQuotient(shape2).sum();
            if (f > 0.) frac += sign * std::pow(f, 2);
        }
        return grid_.volume() * frac / (2.*shape(2));
    }
private:
    Vector3d drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        Vector3d coord(dist(gen), dist(gen), dist(gen));
        if (coord(0) + coord(1) > 1.) {
            coord(0) = 1. - coord(0);
            coord(1) = 1. - coord(1);
        }
        return grid_.origin() + grid_.matrix()*coord;
    }
};

template<typename Back, typename Left, typename Bottom, typename Diag>
class Tetrahedron : public EmitSubdomain {
public:
    typedef fusion::vector4<Back, Left, Bottom, Diag> BdryCont;
private:
    BdryCont bdryCont_;
    void init() {
        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(),
                         "Volume too small, check vector order");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
public:
    Tetrahedron() : EmitSubdomain() {}
    Tetrahedron(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
                    const Vector3d& gradT = Vector3d::Zero(),
                    const double TBack   = 0., const double TLeft = 0.,
                    const double TBottom = 0., const double TDiag = 0.)
    : EmitSubdomain(Grid(o, mat, div), gradT), bdryCont_(
  Back(o,            Triangle(mat.col(1), mat.col(2)), TBack),
  Left(o,            Triangle(mat.col(2), mat.col(0)), TLeft),
Bottom(o,            Triangle(mat.col(0), mat.col(1)), TBottom),
  Diag(o+mat.col(0), Triangle(mat.col(2) - mat.col(0), mat.col(1) - mat.col(0)),
                                                       TDiag))
    {
        init();
    }
    Tetrahedron(const Tetrahedron& tet)
    : EmitSubdomain(tet), bdryCont_(tet.bdryCont_)
    {
        init();
    }
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry() {
        return fusion::at_c<I>(bdryCont_);
    }
    double sdomVol() const {
        return grid_.volume() / 6.;
    }
    double cellVol(const Vector3l& index) const {
        Vector3l shape = grid_.shape();
        
        Vector3d index3 = index.cast<double>();
        Vector3d shape3 = shape.cast<double>();
        double f0 = 1. - index3.cwiseQuotient(shape3).sum();
        if (f0 <= 0.) return 0.;
        
        Vector3l corner(1, 1, 1);
        double f1 = f0 - corner.cast<double>().cwiseQuotient(shape3).sum();
        if (f1 >= 0.) return grid_.volume() / shape.prod();
        
        typedef Eigen::Matrix<long, 3, 6> Points;
        typedef Eigen::Matrix<long, 3, 3> Matrix3l;
        static const Points pts = (Points() << Matrix3l::Identity(),
                                   Matrix3l::Ones() - Matrix3l::Identity())
                                  .finished();
        double frac = std::pow(f0, 3);
        for (int i = 0; i < Points::ColsAtCompileTime; i++) {
            corner = pts.col(i);
            int sign = std::pow(-1, corner.sum());
            double f = f0 - corner.cast<double>().cwiseQuotient(shape3).sum();
            if (f > 0.) frac += sign * std::pow(f, 3);
        }
        return grid_.volume() * frac / 6.;
    }
private:
    Vector3d drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        Vector3d coord(dist(gen), dist(gen), dist(gen));
        if (coord(0) + coord(1) > 1.) {
            coord(0) = 1. - coord(0);
            coord(1) = 1. - coord(1);
        }
        if (coord(1) + coord(2) > 1.) {
            double tmp = coord(2);
            coord(2) = 1. - coord(0) - coord(1);
            coord(1) = 1. - tmp;
        } else if (coord.sum() > 1.) {
            double tmp = coord(2);
            coord(2) = coord.sum() - 1.;
            coord(0) = 1. - coord(1) - tmp;
        }
        return grid_.origin() + grid_.matrix()*coord;
    }
};

class Domain {
protected:
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
    typedef Eigen::DiagonalMatrix<double, 3> Diagonal;
    typedef SpecBoundary Spec;
    typedef DiffBoundary Diff;
    typedef InterBoundary Inter;
    typedef PeriBoundary<Parallelogram> Peri4;
    typedef PeriBoundary<Triangle> Peri3;
public:
    typedef std::vector<const Subdomain*> SdomPtrs;
    typedef std::vector<const Emitter*> EmitPtrs;
private:
    SdomPtrs sdomPtrs_;
    EmitPtrs emitPtrs_;
    std::string info_;
protected:
    void addSdom(const Subdomain* sdom) {
        sdomPtrs_.push_back(sdom);
        emitPtrs_.insert(emitPtrs_.end(), sdom->emitPtrs().begin(),
                         sdom->emitPtrs().end());
    }
    void addSdom(const EmitSubdomain* sdom) {
        if (sdom->emitWeight() != 0.) {
            emitPtrs_.push_back(static_cast<const Emitter*>(sdom));
        }
        addSdom(static_cast<const Subdomain*>(sdom));
    }
    friend class AddSdomF;
    class AddSdomF {
    private:
        Domain* dom_;
    public:
        AddSdomF(Domain* dom) : dom_(dom) {}
        template<typename S>
        void operator()(const S& sdom) const {
            dom_->addSdom(&sdom);
        }
    };
    void info(const std::string& str) {
        info_ = str;
    }
private:
    Domain(const Domain&);
    Domain& operator=(const Domain&);
public:
    Domain() {}
    virtual ~Domain() {}
    virtual bool isInit() const {
        if (sdomPtrs_.size() == 0) return false;
        for (SdomPtrs::const_iterator s = sdomPtrs_.begin();
             s != sdomPtrs_.end(); ++s)
        {
            if (!(*s)->isInit()) return false;
        }
        return true;
    }
    virtual bool isInside(const Vector3d& pos) const {
        return locate(pos);
    }
    const Subdomain* locate(const Vector3d& pos) const {
        for (SdomPtrs::const_iterator s = sdomPtrs_.begin();
             s != sdomPtrs_.end(); ++s)
        {
            if ((*s)->isInside(pos)) return *s;
        }
        return 0;
    }
    const SdomPtrs& sdomPtrs() const { return sdomPtrs_; }
    const EmitPtrs& emitPtrs() const { return emitPtrs_; }
    virtual Eigen::Matrix<double, 3, Eigen::Dynamic> centers() const = 0;
    friend std::ostream& operator<<(std::ostream& os, const Domain& dom) {
        return os << dom.info_;
    }
};

class BulkDomain : public Domain {
public:
    typedef Parallelepiped<Peri4, Spec, Spec> Sdom;
private:
    Sdom sdom_;
    Vector3d center_;
public:
    BulkDomain() : Domain() {}
    BulkDomain(const Vector3d& corner, const Vector3l& div, double deltaT)
    : sdom_(Vector3d::Zero(), corner.asDiagonal(), div,
            Vector3d(-deltaT/corner(0), 0., 0.)),
//            Vector3d::Zero(), deltaT/2, 0., 0., -deltaT/2., 0., 0.),
    center_(0.5*corner)
    {
        std::ostringstream ss;
        ss << "BulkDomain " << static_cast<Domain*>(this) << std::endl;
        ss << "  dim: " << corner.transpose() << std::endl;
        ss << "  div: " << div.transpose() << std::endl;
        ss << "  dT:  " << deltaT;
        info( ss.str() );
        
        makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(corner(0), 0., 0.));
        addSdom(&sdom_);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers() const {
        return center_;
    }
};

class FilmDomain : public Domain {
public:
    typedef Parallelepiped<Peri4, Diff, Spec> Sdom;
private:
    Sdom sdom_;
    Vector3d center_;
public:
    FilmDomain() : Domain() {}
    FilmDomain(const Vector3d& corner, const Vector3l& div, double deltaT)
    : sdom_(Vector3d::Zero(), corner.asDiagonal(), div,
            Vector3d(-deltaT/corner(0), 0., 0.)),
    center_(0.5*corner)
    {
        std::ostringstream ss;
        ss << "FilmDomain " << static_cast<Domain*>(this) << std::endl;
        ss << "  dim: " << corner.transpose() << std::endl;
        ss << "  div: " << div.transpose() << std::endl;
        ss << "  dT:  " << deltaT;
        info( ss.str() );
        
        makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(corner(0), 0., 0.));
        addSdom(&sdom_);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers() const {
        return center_;
    }
};

class TeeDomain : public Domain {
public:
    typedef fusion::vector4<
            Parallelepiped<Peri4, Diff,  Spec, Inter, Diff,  Spec>,
            Parallelepiped<Inter, Spec,  Spec, Inter, Inter, Spec>,
            Parallelepiped<Diff,  Inter, Spec, Diff,  Spec,  Spec>,
            Parallelepiped<Inter, Diff,  Spec, Peri4, Diff,  Spec> > SdomCont;
private:
    SdomCont sdomCont_;
    Eigen::Matrix<double, 4, 3> centers_;
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I> {};
public:
    TeeDomain() : Domain() {}
    TeeDomain(const Eigen::Matrix<double, 5, 1>& dim,
              const Eigen::Matrix<long, 5, 1>& div,
              double deltaT);
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom() {
        return fusion::at_c<I>(sdomCont_);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers() const {
        return centers_.transpose();
    }
};

class TubeDomain : public Domain {
public:
    typedef fusion::vector3<
    Parallelepiped<Peri4, Diff,  Spec,  Peri4, Diff,  Inter>,
    Parallelepiped<Peri4, Inter, Inter, Peri4, Diff,  Diff >,
    Parallelepiped<Peri4, Spec,  Diff,  Peri4, Inter, Diff > > SdomCont;
private:
    SdomCont sdomCont_;
    Eigen::Matrix<double, 3, 3> centers_;
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I> {};
public:
    TubeDomain() : Domain() {}
    TubeDomain(const Eigen::Matrix<double, 4, 1>& dim,
               const Eigen::Matrix<long, 4, 1>& div,
               double deltaT);
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom() {
        return fusion::at_c<I>(sdomCont_);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers() const {
        return centers_.transpose();
    }
};

class OctetDomain : public Domain {
public:
    typedef fusion::vector33<
    Parallelepiped<Diff,  Spec,  Peri4, Diff,  Inter, Diff >, // 00
    Parallelepiped<Diff,  Inter, Peri4, Inter, Diff,  Diff >, // 01
    Parallelepiped<Inter, Diff,  Peri4, Inter, Diff,  Inter>, // 02
    Parallelepiped<Inter, Diff,  Peri4, Inter, Diff,  Inter>, // 03
    Parallelepiped<Inter, Diff,  Peri4, Inter, Diff,  Inter>, // 04
    Parallelepiped<Inter, Inter, Peri4, Diff,  Diff,  Inter>, // 05
    Parallelepiped<Diff,  Spec,  Peri4, Diff,  Inter, Inter>, // 06
    
    Parallelepiped<Spec,  Diff,  Inter, Inter, Inter, Diff >, // 07
    Parallelepiped<Inter, Diff,  Inter, Inter, Inter, Inter>, // 08
    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Inter>, // 09
    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Inter>, // 10
    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Inter>, // 11
    Parallelepiped<Spec,  Inter, Diff,  Inter, Spec,  Diff >, // 12
    Parallelepiped<Inter, Inter, Diff,  Diff,  Spec,  Inter>, // 13
    
    Parallelepiped<Diff,  Diff,  Inter, Inter, Inter, Inter>, // 14
    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Inter>, // 15
    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Inter>, // 16
    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Inter>, // 17
    Parallelepiped<Diff,  Inter, Inter, Diff,  Spec,  Inter>, // 18
    
    Parallelepiped<Spec,  Diff,  Diff , Inter, Inter, Inter>, // 19
    Parallelepiped<Inter, Diff,  Inter, Inter, Inter, Inter>, // 20
    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Inter>, // 21
    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Inter>, // 22
    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Inter>, // 23
    Parallelepiped<Spec,  Inter, Diff,  Inter, Spec,  Diff >, // 24
    Parallelepiped<Inter, Inter, Inter, Diff,  Spec,  Diff >, // 25
    
    Parallelepiped<Diff,  Spec,  Diff,  Diff,  Inter, Peri4>, // 26
    Parallelepiped<Diff,  Inter, Diff,  Inter, Diff,  Peri4>, // 27
    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Peri4>, // 28
    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Peri4>, // 29
    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Peri4>, // 30
    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Peri4>, // 31
    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Peri4>  // 32
    > SdomCont;
private:
    SdomCont sdomCont_;
    Eigen::Matrix<double, Eigen::Dynamic, 3> centers_;
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I> {};
public:
    OctetDomain() : Domain() {}
    OctetDomain(const Eigen::Matrix<double, 5, 1>& dim,
                const Eigen::Matrix<long, 5, 1>& div,
                double deltaT);
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom() {
        return fusion::at_c<I>(sdomCont_);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> centers() const {
        return centers_.transpose();
    }
};

#endif
