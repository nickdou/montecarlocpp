//
//  domain.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/5/15.
//
//

#ifndef __montecarlocpp__domain__
#define __montecarlocpp__domain__

#include "subdomain.h"
#include "boundary.h"
#include <Eigen/Core>
#include <boost/fusion/sequence/intrinsic.hpp>
#include <boost/fusion/container/vector/vector50.hpp>
#include <boost/fusion/container/vector/vector10.hpp>
#include <iostream>
#include <string>

namespace fusion = boost::fusion;
namespace result_of = boost::fusion::result_of;

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::ArrayXXd;

typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<long, 3, 1> Vector3l;
typedef Eigen::Matrix<long, 4, 1> Vector4l;
typedef Eigen::Matrix<long, 5, 1> Vector5l;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
typedef Eigen::DiagonalMatrix<double, 3> Diagonal3d;

class Domain
{
protected:
    typedef SpecBoundary                  Spec;
    typedef DiffBoundary                  Diff;
    typedef InterBoundary                 Inter;
    typedef PeriBoundary< Triangle >      PeriT;
    typedef PeriBoundary< Parallelogram > PeriP;
    typedef PeriBoundary< Polygon<4> >    Peri4;
    typedef PeriBoundary< Polygon<5> >    Peri5;
    typedef PeriBoundary< Polygon<6> >    Peri6;
    
private:
    Subdomain::Pointers sdomPtrs_;
    Emitter::Pointers emitPtrs_;
    
    virtual std::string info() const = 0;
    
public:
    Domain();
    Domain(const Domain& dom);
    Domain& operator=(const Domain& dom);
    virtual ~Domain();
    
    bool isInit() const;
    bool isInside(const Vector3d& pos) const;
    const Subdomain* locate(const Vector3d& pos) const;
    
    const Subdomain::Pointers& sdomPtrs() const;
    const Emitter::Pointers& emitPtrs() const;
    
    virtual Matrix3Xd checkpoints() const = 0;
    virtual ArrayXXd average(const ArrayXXd& data) const;
    
    friend std::ostream& operator<<(std::ostream& os, const Domain& dom);
    
protected:
    void addSdom(const Subdomain* sdom);
    void addSdom(const EmitSubdomain* sdom);
    
    class AddSdomF;
    friend class AddSdomF;
};

class Domain::AddSdomF
{
private:
    Domain* dom_;
    
public:
    AddSdomF(Domain* dom);
    
    void operator()(const Subdomain& bdry) const;
    void operator()(const EmitSubdomain& bdry) const;
};

class BulkDomain : public Domain
{
public:
    typedef Parallelepiped<PeriP, Spec, Spec> Sdom;
    
private:
    Sdom sdom_;
    Vector3d dim_;
    Vector3l div_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    BulkDomain();
    BulkDomain(const Vector3d& dim, const Vector3l& div, double dT);
    BulkDomain(const BulkDomain& dom);
    BulkDomain& operator=(const BulkDomain& dom);
    
    Matrix3Xd checkpoints() const;
};

class FilmDomain : public Domain
{
public:
    typedef Parallelepiped<PeriP, Diff, Spec> Sdom;
    
private:
    Sdom sdom_;
    Vector3d dim_;
    Vector3l div_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    FilmDomain();
    FilmDomain(const Vector3d& dim, const Vector3l& div, double dT);
    FilmDomain(const FilmDomain& dom);
    FilmDomain& operator=(const FilmDomain& dom);
    
    Matrix3Xd checkpoints() const;
};

class HexDomain : public Domain
{
public:
    typedef Prism< PeriBoundary< Polygon<6> >, PeriBoundary< Polygon<6> >,
        fusion::vector6<Spec, Spec, Spec, Spec, Spec, Spec> > Sdom;
    
private:
    Sdom sdom_;
    Vector4d dim_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    HexDomain();
    HexDomain(const Vector4d& dim, double dT);
    HexDomain(const HexDomain& dom);
    HexDomain& operator=(const HexDomain& dom);
    
    Matrix3Xd checkpoints() const;
};

class PyrDomain : public Domain
{
public:
    typedef Pyramid< Spec, fusion::vector4<Spec, Spec, Spec, Spec> > Sdom;
    
private:
    Sdom sdom_;
    Vector3d dim_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    PyrDomain();
    PyrDomain(const Vector3d& dim, double dT);
    PyrDomain(const PyrDomain& dom);
    PyrDomain& operator=(const PyrDomain& dom);
    
    Matrix3Xd checkpoints() const;
};

class JctDomain : public Domain
{
public:
    typedef fusion::vector3<
    Parallelepiped<PeriP, Spec,  Diff, PeriP, Inter, Diff>,
    Parallelepiped<PeriP, Inter, Diff, Inter,  Spec, Diff>,
    Parallelepiped<Inter, Inter, Diff, PeriP,  Spec, Diff> > SdomCont;
    
private:
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I>
    {};
    
    SdomCont sdomCont_;
    Vector4d dim_;
    Vector4l div_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    JctDomain();
    JctDomain(const Vector4d& dim,
              const Vector4l& div,
              double dT);
    JctDomain(const JctDomain& dom);
    JctDomain& operator=(const JctDomain& dom);
    
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom()
    {
        return fusion::at_c<I>(sdomCont_);
    }
    
    Matrix3Xd checkpoints() const;
};

class TeeDomain : public Domain
{
public:
    typedef fusion::vector4<
    Parallelepiped<PeriP, Diff,  Spec, Inter, Diff,  Spec>,
    Parallelepiped<Inter, Spec,  Spec, Inter, Inter, Spec>,
    Parallelepiped<Diff,  Inter, Spec, Diff,  Spec,  Spec>,
    Parallelepiped<Inter, Diff,  Spec, PeriP, Diff,  Spec> > SdomCont;
    
private:
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I>
    {};
    
    SdomCont sdomCont_;
    Vector5d dim_;
    Vector5l div_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    TeeDomain();
    TeeDomain(const Vector5d& dim,
              const Vector5l& div,
              double dT);
    TeeDomain(const TeeDomain& dom);
    TeeDomain& operator=(const TeeDomain& dom);
    
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom()
    {
        return fusion::at_c<I>(sdomCont_);
    }
    
    Matrix3Xd checkpoints() const;
};

class TubeDomain : public Domain
{
public:
    typedef fusion::vector3<
    Parallelepiped<PeriP, Diff,  Spec,  PeriP, Diff,  Inter>,
    Parallelepiped<PeriP, Inter, Inter, PeriP, Diff,  Diff >,
    Parallelepiped<PeriP, Spec,  Diff,  PeriP, Inter, Diff > > SdomCont;
    
private:
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I>
    {};
    
    SdomCont sdomCont_;
    Vector4d dim_;
    Vector4l div_;
    double dT_;
    
    std::string info() const;
    void init();
    
public:
    TubeDomain();
    TubeDomain(const Vector4d& dim,
               const Vector4l& div,
               double dT);
    TubeDomain(const TubeDomain& dom);
    TubeDomain& operator=(const TubeDomain& dom);
    
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom()
    {
        return fusion::at_c<I>(sdomCont_);
    }
    
    Matrix3Xd checkpoints() const;
};

class OctetDomain : public Domain
{
public:
    typedef fusion::vector42<
    Prism< Inter, Peri4, fusion::vector4< Inter, Diff,  Spec,  Diff  > >, // 00
    Prism< Diff,  Inter, fusion::vector4< Inter, Diff,  Spec,  Inter > >, // 01
    Prism< Diff,  Diff,  fusion::vector4< Inter, Inter, Spec,  Spec  > >, // 02
    Prism< Inter, Peri4, fusion::vector4< Diff,  Inter, Inter, Diff  > >, // 03
    Prism< Inter, Inter, fusion::vector4< Diff,  Inter, Inter, Inter > >, // 04
    Parallelepiped< Inter, Spec,  Diff,  Diff,  Inter, Inter >,           // 05
    
    Prism< Spec,  Inter, fusion::vector4< PeriP, Diff,  Inter, Diff  > >, // 06
    Prism< Inter, Diff,  fusion::vector4< PeriP, Diff,  Inter, Inter > >, // 07
    Prism< Diff,  Diff,  fusion::vector4< PeriP, Inter, Inter, Inter > >, // 08
    Prism< Diff,  Diff,  fusion::vector4< Inter, Inter, Inter, Inter > >, // 09
    Prism< Inter, Diff,  fusion::vector4< Inter, Inter, Diff,  Inter > >, // 10
    Prism< Spec,  Inter, fusion::vector4< Diff,  Inter, Diff,  Inter > >, // 11
    TriangularPrism< Diff,  Inter, Inter, Spec,  Inter >,                 // 12
    Pyramid< Inter, fusion::vector4< Diff,  Inter, Spec,  Inter > >,      // 13
    TriangularPrism< Inter, Inter, Diff,  Spec,  Inter >,                 // 14
    Prism< Diff,  Diff,  fusion::vector4< Inter, Inter, Spec,  Spec  > >, // 15
    
    Parallelepiped< Diff,  Spec,  Inter, Diff,  Inter, Inter >,           // 16
    Parallelepiped< Inter, Inter, Inter, Diff,  Diff,  Inter >,           // 17
    Parallelepiped< Inter, Diff,  Inter, Inter, Diff,  Inter >,           // 18
    Parallelepiped< Diff,  Inter, Inter, Inter, Diff,  Inter >,           // 19
    Parallelepiped< Diff,  Spec,  Inter, Diff,  Inter, Inter >,           // 20

    Parallelepiped< Diff,  Spec,  Inter, Diff,  Inter, Inter >,           // 21
    Parallelepiped< Diff,  Inter, Inter, Inter, Diff,  Inter >,           // 22
    Parallelepiped< Inter, Diff,  Inter, Inter, Diff,  Inter >,           // 23
    Parallelepiped< Inter, Inter, Inter, Diff,  Diff,  Inter >,           // 24
    Parallelepiped< Diff,  Spec,  Inter, Diff,  Inter, Inter >,           // 25
    
    Prism< Diff,  Diff,  fusion::vector4< Inter, Inter, Spec,  Spec  > >, // 26
    TriangularPrism< Inter, Inter, Diff,  Spec,  Inter >,                 // 27
    Pyramid< Inter, fusion::vector4< Diff,  Inter, Spec,  Inter > >,      // 28
    TriangularPrism< Diff,  Inter, Inter, Spec,  Inter >,                 // 29
    Prism< Spec,  Inter, fusion::vector4< Diff,  Inter, Diff,  Inter > >, // 30
    Prism< Inter, Diff,  fusion::vector4< Inter, Inter, Diff,  Inter > >, // 31
    Prism< Diff,  Diff,  fusion::vector4< Inter, Inter, Inter, Inter > >, // 32
    Prism< Diff,  Diff,  fusion::vector4< PeriP, Inter, Inter, Inter > >, // 33
    Prism< Inter, Diff,  fusion::vector4< PeriP, Diff,  Inter, Inter > >, // 34
    Prism< Spec,  Inter, fusion::vector4< PeriP, Diff,  Inter, Diff  > >, // 35

    Parallelepiped< Inter, Spec,  Diff,  Diff,  Inter, Inter >,           // 36
    Prism< Inter, Inter, fusion::vector4< Diff,  Inter, Inter, Inter > >, // 37
    Prism< Inter, Peri4, fusion::vector4< Diff,  Inter, Inter, Diff  > >, // 38
    Prism< Diff,  Diff,  fusion::vector4< Inter, Inter, Spec,  Spec  > >, // 39
    Prism< Diff,  Inter, fusion::vector4< Inter, Diff,  Spec,  Inter > >, // 40
    Prism< Inter, Peri4, fusion::vector4< Inter, Diff,  Spec,  Diff  > >  // 41
    > SdomCont;
    
private:
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I>
    {};
    
    class WeightF;
    
    SdomCont sdomCont_;
    Vector4d dim_;
    Vector4l div_;
    double dT_;
    Matrix3Xd pts_;
    
    std::string info() const;
    void init();
    
public:
    OctetDomain();
    OctetDomain(const Vector4d& dim,
                const Vector4l& div,
                double dT);
    OctetDomain(const OctetDomain& dom);
    OctetDomain& operator=(const OctetDomain& dom);
    
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom()
    {
        return fusion::at_c<I>(sdomCont_);
    }
    
    Matrix3Xd checkpoints() const;
    ArrayXXd average(const ArrayXXd& data) const;
};

class OctetDomain::WeightF
{
private:
    Subdomain::Pointers sdomAvg_;
    
public:
    WeightF(const OctetDomain& dom);
    
    VectorXd operator()(const Subdomain* sdom, const Vector3l& index) const;
};


//class Subdomain {
//protected:
//    typedef Eigen::Matrix<long, 3, 1> Vector3l;
//    typedef Eigen::Vector3d Vector3d;
//    typedef Eigen::Matrix3d Matrix3d;
//public:
//    typedef std::vector<const Boundary*> BdryPtrs;
//    typedef std::vector<const Emitter*> EmitPtrs;
//private:
//    BdryPtrs bdryPtrs_;
//    EmitPtrs emitPtrs_;
//protected:
//    void addBdry(Boundary* bdry) {
//        bdry->sdom(this);
//        bdryPtrs_.push_back(bdry);
//    }
//    template<typename S>
//    void addBdry(EmitBoundary<S>* bdry) {
//        addBdry(static_cast<Boundary*>(bdry));
//        if (bdry->emitWeight() != 0.) {
//            emitPtrs_.push_back(static_cast<Emitter*>(bdry));
//        }
//    }
//    friend class AddBdryF;
//    class AddBdryF {
//    private:
//        Subdomain* sdom_;
//    public:
//        AddBdryF(Subdomain* sdom) : sdom_(sdom) {}
//        template<typename B>
//        void operator()(B& bdry) const {
//            sdom_->addBdry(&bdry);
//        }
//    };
//    Grid grid_;
//public:
//    Subdomain() {}
//    Subdomain(const Grid& grid) : grid_(grid) {}
//    Subdomain(const Subdomain& sdom) : grid_(sdom.grid_) {}
//    Subdomain& operator=(const Subdomain& sdom) {
//        grid_ = sdom.grid_;
//        return *this;
//    }
//    virtual ~Subdomain() {}
//    bool isInit() const {
//        if (!grid_.isInit() || bdryPtrs_.size() == 0) {
//            return false;
//        }
//        for (BdryPtrs::const_iterator b = bdryPtrs_.begin();
//             b != bdryPtrs_.end(); ++b)
//        {
//            if (!(*b)->isInit()) return false;
//        }
//        return true;
//    }
//    bool isInside(const Vector3d& pos) const {
//        for (BdryPtrs::const_iterator b = bdryPtrs_.begin();
//             b != bdryPtrs_.end(); ++b)
//        {
//            if ((*b)->distance(pos) < 0.) return false;
//        }
//        return true;
//    }
//    const BdryPtrs& bdryPtrs() const { return bdryPtrs_; }
//    const EmitPtrs& emitPtrs() const { return emitPtrs_; }
//    template<typename T>
//    Data<T> initData(const T& value) const {
//        return Data<T>( Collection(grid_.shape()), value );
//    }
//    template<typename T>
//    void accumulate(Phonon& phn, const Vector3d& begin, const Vector3d& end,
//                    Data<T>& data, T quant) const
//    {
//        bool status = grid_.accumulate(begin, end, data, quant);
//        if (!status) phn.kill();
//    }
//    virtual double sdomVol() const = 0;
//    virtual double cellVol(const Vector3l& index) const = 0;
////    virtual const Boundary* advect(Phonon& phn, double& scatDist) const {
//    const Boundary* advect(Phonon& phn, double& scatDist) const {
//        double minDist = scatDist;
//        const Boundary* newBdry = 0;
//        for (BdryPtrs::const_iterator b = bdryPtrs_.begin();
//             b != bdryPtrs_.end(); ++b)
//        {
//            if ((*b)->normal().dot(phn.dir()) >= 0.) continue;
//            double dist = (*b)->distance(phn);
//            if (dist < 0. && !(*b)->projection(phn.pos()).isApprox(phn.pos())) {
//                phn.kill();
//                return *b;
//            }
//            if (dist < minDist) {
//                minDist = dist;
//                newBdry = *b;
//            }
//        }
//        phn.move(minDist);
//        scatDist -= minDist;
//        return newBdry;
//    }
//};
//
//class EmitSubdomain : public Subdomain, public Emitter {
//private:
//    Vector3d gradT_;
//    Matrix3d rot_;
//public:
//    EmitSubdomain() : Subdomain() {}
//    EmitSubdomain(const Grid& grid, const Vector3d& gradT)
//    : Subdomain(grid), gradT_(gradT), rot_(rotMatrix(gradT.normalized())) {}
//    EmitSubdomain(const EmitSubdomain& sdom)
//    : Subdomain(sdom), gradT_(sdom.gradT_), rot_(sdom.rot_) {}
//    virtual ~EmitSubdomain() {}
//    const Subdomain* emitSdom() const {
//        return this;
//    }
//    const Boundary* emitBdry() const { return 0; }
//    double emitWeight() const {
//        return 2. * sdomVol() * gradT_.norm();
//    }
//private:
//    Vector3d drawDir(Rng& gen) const {
//        return rot_ * drawAniso<true>(gen);
//    }
//    bool emitSign(const Vector3d&, const Vector3d& dir) const {
//        return dir.dot(gradT_) < 0.;
//    }
//};
//
//template<typename Back, typename Left, typename Bottom,
//         typename Front = Back, typename Right = Left, typename Top = Bottom>
//class Parallelepiped : public EmitSubdomain {
//public:
//    typedef fusion::vector6<Back, Left, Bottom, Front, Right, Top> BdryCont;
//private:
//    BdryCont bdryCont_;
//    void init() {
//        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(),
//                         "Volume too small, check vector order");
//        fusion::for_each(bdryCont_, AddBdryF(this));
//    }
//public:
//    Parallelepiped() : EmitSubdomain() {}
//    Parallelepiped(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
//                   const Vector3d& gradT = Vector3d::Zero(),
//                   const double TBack   = 0., const double TLeft  = 0.,
//                   const double TBottom = 0., const double TFront = 0.,
//                   const double TRight  = 0., const double TTop   = 0.)
//    : EmitSubdomain(Grid(o, mat, div), gradT), bdryCont_(
//      Back(o,            Parallelogram(mat.col(1), mat.col(2)), TBack  ),
//      Left(o,            Parallelogram(mat.col(2), mat.col(0)), TLeft  ),
//    Bottom(o,            Parallelogram(mat.col(0), mat.col(1)), TBottom),
//     Front(o+mat.col(0), Parallelogram(mat.col(2), mat.col(1)), TFront ),
//     Right(o+mat.col(1), Parallelogram(mat.col(0), mat.col(2)), TRight ),
//       Top(o+mat.col(2), Parallelogram(mat.col(1), mat.col(0)), TTop   ))
//    {
//        init();
//    }
//    Parallelepiped(const Parallelepiped& para)
//    : EmitSubdomain(para), bdryCont_(para.bdryCont_)
//    {
//        init();
//    }
//    template<int I>
//    typename result_of::at_c<BdryCont, I>::type bdry() {
//        return fusion::at_c<I>(bdryCont_);
//    }
//    double sdomVol() const {
//        return grid_.volume();
//    }
//    double cellVol(const Vector3l&) const {
//        return grid_.volume() / grid_.shape().prod();
//    }
//private:
//    Vector3d drawPos(Rng& gen) const {
//        static UniformDist01 dist; // [0, 1)
//        Vector3d coord(dist(gen), dist(gen), dist(gen));
//        return grid_.origin() + grid_.matrix()*coord;
//    }
//};
//
//template<typename Back, typename Left, typename Bottom,
//         typename Diag, typename Top = Bottom>
//class TriangularPrism : public EmitSubdomain {
//public:
//    typedef fusion::vector5<Back, Left, Bottom, Diag, Top> BdryCont;
//private:
//    BdryCont bdryCont_;
//    void init() {
//        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(),
//                         "Volume too small, check vector order");
//        fusion::for_each(bdryCont_, AddBdryF(this));
//    }
//public:
//    TriangularPrism() : EmitSubdomain() {}
//    TriangularPrism(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
//                    const Vector3d& gradT = Vector3d::Zero(),
//                    const double TBack   = 0., const double TLeft = 0.,
//                    const double TBottom = 0., const double TDiag = 0.,
//                    const double TTop    = 0.)
//    : EmitSubdomain(Grid(o, mat, div), gradT), bdryCont_(
//  Back(o,            Parallelogram(mat.col(1), mat.col(2)),              TBack),
//  Left(o,            Parallelogram(mat.col(2), mat.col(0)),              TLeft),
//Bottom(o,                 Triangle(mat.col(0), mat.col(1)),            TBottom),
//  Diag(o+mat.col(0), Parallelogram(mat.col(2), mat.col(1) - mat.col(0)), TDiag),
//   Top(o+mat.col(2),      Triangle(mat.col(1), mat.col(0)),              TTop ))
//    {
//        init();
//    }
//    TriangularPrism(const TriangularPrism& prism)
//    : EmitSubdomain(prism), bdryCont_(prism.bdryCont_)
//    {
//        init();
//    }
//    template<int I>
//    typename result_of::at_c<BdryCont, I>::type bdry() {
//        return fusion::at_c<I>(bdryCont_);
//    }
//    double sdomVol() const {
//        return grid_.volume() / 2.;
//    }
//    double cellVol(const Vector3l& index) const {
//        Vector3l shape = grid_.shape();
//        
//        Eigen::Vector2d index2 = index.head<2>().cast<double>();
//        Eigen::Vector2d shape2 = shape.head<2>().cast<double>();
//        double f0 = 1. - index2.cwiseQuotient(shape2).sum();
//        if (f0 <= 0.) return 0.;
//        
//        Eigen::Matrix<long, 2, 1> corner(1, 1);
//        double f1 = f0 - corner.cast<double>().cwiseQuotient(shape2).sum();
//        if (f1 >= 0.) return grid_.volume() / shape.prod();
//        
//        typedef Eigen::Matrix<long, 2, 2> Points;
//        static const Points pts = Points::Identity();
//        double frac = std::pow(f0, 2);
//        for (int i = 0; i < Points::ColsAtCompileTime; i++) {
//            corner = pts.col(i);
//            int sign = std::pow(-1, corner.sum());
//            double f = f0 - corner.cast<double>().cwiseQuotient(shape2).sum();
//            if (f > 0.) frac += sign * std::pow(f, 2);
//        }
//        return grid_.volume() * frac / (2.*shape(2));
//    }
//private:
//    Vector3d drawPos(Rng& gen) const {
//        static UniformDist01 dist; // [0, 1)
//        Vector3d coord(dist(gen), dist(gen), dist(gen));
//        if (coord(0) + coord(1) > 1.) {
//            coord(0) = 1. - coord(0);
//            coord(1) = 1. - coord(1);
//        }
//        return grid_.origin() + grid_.matrix()*coord;
//    }
//};
//
//template<typename Bottom, typename Top, typename Sides>
//class Prism : public EmitSubdomain {
//private:
//    typedef typename result_of::push_front<Sides, Top>::type TopSides;
//    typedef typename result_of::push_front<TopSides, Bottom>::type BotTopSides;
//public:
//    typedef typename result_of::as_vector<BotTopSides>::type BdryCont;
//private:
//    static const int N = result_of::size<Sides>::type::value;
//    typedef Eigen::Matrix<long, 3, 1> Vector3l;
//    typedef Eigen::Vector3d Vector3d;
//    typedef Eigen::Matrix3d Matrix3d;
//    typedef Eigen::Matrix<double, 3, N> Matrix3Nd;
//    typedef Eigen::Matrix<double, 1, N> RowVectorNd;
//    typedef Eigen::Matrix<double, 1, N-2> RowVectorLd;
//    BdryCont bdryCont_;
//    Matrix3Nd mat_;
//    long div_;
//    RowVectorLd vol_;
//    DiscreteDist volDist_;
//    Grid initGrid(const Vector3d& o, const Matrix3Nd& mat, long div) const {
//        Matrix3d matGrid;
//        matGrid << mat.col(1), mat.col(N-1), mat.col(0);
//        return Grid(o, matGrid, Vector3l(0, 0, div));
//    }
//    BdryCont initBdry(const Vector3d& o, const Matrix3Nd& mat,
//                      const double TBottom, const double TTop,
//                      const RowVectorNd& TSides) const
//    {
//        Eigen::Matrix<double, 3, N-1> matBottom, matTop;
//        matBottom = mat.rightCols(N-1);
//        matTop = mat.rightCols(N-1).rowwise().reverse();
//        
//        Bottom b = Bottom(o, Polygon<N>(matBottom), TBottom);
//        Top t = Top(o + mat.col(0), Polygon<N>(matTop), TTop);
//        Sides s = fusion::transform(mpl::range_c<int, 0, N>(),
//                                    MakeSidesF(o, mat, TSides));
//        return fusion::push_front( fusion::push_front(s, t), b);
//    }
//    struct MakeSidesF {
//        int ind_;
//        Vector3d o_;
//        Matrix3Nd mat_;
//        RowVectorNd T_;
//        MakeSidesF(const Vector3d& o, const Matrix3Nd& mat,
//                   const RowVectorNd& T)
//        : o_(o), mat_(mat), T_(T)
//        {}
//        template<typename I>
//        typename result_of::value_at<Sides, I>::type operator()(const I&) const{
//            typedef typename result_of::value_at<Sides, I>::type S;
//            const int ind = I::value;
//            Vector3d p = Vector3d::Zero();
//            if (ind != 0) p += mat_.col(ind);
//            Vector3d i = mat_.col(0);
//            Vector3d j = -p;
//            if (ind != N-1) j += mat_.col(ind+1);
//            return S(o_ + p, Parallelogram(i, j), T_(ind));
//        }
//    };
//    RowVectorLd initVol(const Matrix3Nd& mat) const {
//        RowVectorLd v;
//        for (int i = 0; i < N-2; ++i) {
//            v(i) = mat.col(i).cross( mat.col(i+1) ).dot( mat.col(0) );
//        }
//        return v;
//    }
//    void init() {
//        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(), "Volume too small");
//        fusion::for_each(bdryCont_, AddBdryF(this));
//    }
//public:
//    Prism() : EmitSubdomain() {}
//    Prism(const Vector3d& o, const Matrix3Nd& mat, long div,
//          const Vector3d& gradT = Vector3d::Zero(),
//          const double TBottom = 0., const double TTop = 0.,
//          const RowVectorNd& TSides = RowVectorNd::Zero())
//    : EmitSubdomain(initGrid(o, mat, div), gradT),
//    bdryCont_(initBdry(o, mat, TBottom, TTop, TSides)),
//    mat_(mat), div_(div), vol_(initVol(mat))
//    {
//        volDist_ = DiscreteDist(vol_.data(), vol_.data() + N-2);
//        init();
//    }
//    Prism(const Prism& p)
//    : EmitSubdomain(p), bdryCont_(p.bdryCont_),
//    mat_(p.mat_), div_(p.div_), vol_(p.vol_), volDist_(p.volDist_)
//    {
//        init();
//    }
//    template<int I>
//    typename result_of::at_c<BdryCont, I>::type bdry() {
//        return fusion::at_c<I>(bdryCont_);
//    }
//    double sdomVol() const {
//        return vol_.sum();
//    }
//    double cellVol(const Vector3l& index) const {
//        return vol_.sum() / div_;
//    }
//private:
//    Vector3d drawPos(Rng& gen) const {
//        long ind = volDist_(gen);
//        Matrix3d local;
//        local << mat_.col(ind+1), mat_.col(ind+2), mat_.col(0);
//        static UniformDist01 dist; // [0, 1)
//        Vector3d coord(dist(gen), dist(gen), dist(gen));
//        if (coord(0) + coord(1) > 1.) {
//            coord(0) = 1. - coord(0);
//            coord(1) = 1. - coord(1);
//        }
//        return grid_.origin() + local * coord;
//    }
//};
//
//template<typename Back, typename Left, typename Bottom, typename Diag>
//class Tetrahedron : public EmitSubdomain {
//public:
//    typedef fusion::vector4<Back, Left, Bottom, Diag> BdryCont;
//private:
//    BdryCont bdryCont_;
//    void init() {
//        BOOST_ASSERT_MSG(sdomVol() >= Dbl::min(),
//                         "Volume too small, check vector order");
//        fusion::for_each(bdryCont_, AddBdryF(this));
//    }
//public:
//    Tetrahedron() : EmitSubdomain() {}
//    Tetrahedron(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
//                    const Vector3d& gradT = Vector3d::Zero(),
//                    const double TBack   = 0., const double TLeft = 0.,
//                    const double TBottom = 0., const double TDiag = 0.)
//    : EmitSubdomain(Grid(o, mat, div), gradT), bdryCont_(
//  Back(o,            Triangle(mat.col(1), mat.col(2)), TBack),
//  Left(o,            Triangle(mat.col(2), mat.col(0)), TLeft),
//Bottom(o,            Triangle(mat.col(0), mat.col(1)), TBottom),
//  Diag(o+mat.col(0), Triangle(mat.col(2) - mat.col(0), mat.col(1) - mat.col(0)),
//                                                       TDiag))
//    {
//        init();
//    }
//    Tetrahedron(const Tetrahedron& tet)
//    : EmitSubdomain(tet), bdryCont_(tet.bdryCont_)
//    {
//        init();
//    }
//    template<int I>
//    typename result_of::at_c<BdryCont, I>::type bdry() {
//        return fusion::at_c<I>(bdryCont_);
//    }
//    double sdomVol() const {
//        return grid_.volume() / 6.;
//    }
//    double cellVol(const Vector3l& index) const {
//        Vector3l shape = grid_.shape();
//        
//        Vector3d index3 = index.cast<double>();
//        Vector3d shape3 = shape.cast<double>();
//        double f0 = 1. - index3.cwiseQuotient(shape3).sum();
//        if (f0 <= 0.) return 0.;
//        
//        Vector3l corner(1, 1, 1);
//        double f1 = f0 - corner.cast<double>().cwiseQuotient(shape3).sum();
//        if (f1 >= 0.) return grid_.volume() / shape.prod();
//        
//        typedef Eigen::Matrix<long, 3, 6> Points;
//        typedef Eigen::Matrix<long, 3, 3> Matrix3l;
//        static const Points pts = (Points() << Matrix3l::Identity(),
//                                   Matrix3l::Ones() - Matrix3l::Identity())
//                                  .finished();
//        double frac = std::pow(f0, 3);
//        for (int i = 0; i < Points::ColsAtCompileTime; i++) {
//            corner = pts.col(i);
//            int sign = std::pow(-1, corner.sum());
//            double f = f0 - corner.cast<double>().cwiseQuotient(shape3).sum();
//            if (f > 0.) frac += sign * std::pow(f, 3);
//        }
//        return grid_.volume() * frac / 6.;
//    }
//private:
//    Vector3d drawPos(Rng& gen) const {
//        static UniformDist01 dist; // [0, 1)
//        Vector3d coord(dist(gen), dist(gen), dist(gen));
//        if (coord(0) + coord(1) > 1.) {
//            coord(0) = 1. - coord(0);
//            coord(1) = 1. - coord(1);
//        }
//        if (coord(1) + coord(2) > 1.) {
//            double tmp = coord(2);
//            coord(2) = 1. - coord(0) - coord(1);
//            coord(1) = 1. - tmp;
//        } else if (coord.sum() > 1.) {
//            double tmp = coord(2);
//            coord(2) = coord.sum() - 1.;
//            coord(0) = 1. - coord(1) - tmp;
//        }
//        return grid_.origin() + grid_.matrix()*coord;
//    }
//};
//
//class OctetDomain : public Domain
//{
//public:
//    typedef fusion::vector33<
//    Parallelepiped<Diff,  Spec,  PeriP, Diff,  Inter, Diff >, // 00
//    Parallelepiped<Diff,  Inter, PeriP, Inter, Diff,  Diff >, // 01
//    Parallelepiped<Inter, Diff,  PeriP, Inter, Diff,  Inter>, // 02
//    Parallelepiped<Inter, Diff,  PeriP, Inter, Diff,  Inter>, // 03
//    Parallelepiped<Inter, Diff,  PeriP, Inter, Diff,  Inter>, // 04
//    Parallelepiped<Inter, Inter, PeriP, Diff,  Diff,  Inter>, // 05
//    Parallelepiped<Diff,  Spec,  PeriP, Diff,  Inter, Inter>, // 06
//    
//    Parallelepiped<Spec,  Diff,  Inter, Inter, Inter, Diff >, // 07
//    Parallelepiped<Inter, Diff,  Inter, Inter, Inter, Inter>, // 08
//    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Inter>, // 09
//    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Inter>, // 10
//    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Inter>, // 11
//    Parallelepiped<Spec,  Inter, Diff,  Inter, Spec,  Diff >, // 12
//    Parallelepiped<Inter, Inter, Diff,  Diff,  Spec,  Inter>, // 13
//    
//    Parallelepiped<Diff,  Diff,  Inter, Inter, Inter, Inter>, // 14
//    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Inter>, // 15
//    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Inter>, // 16
//    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Inter>, // 17
//    Parallelepiped<Diff,  Inter, Inter, Diff,  Spec,  Inter>, // 18
//    
//    Parallelepiped<Spec,  Diff,  Diff , Inter, Inter, Inter>, // 19
//    Parallelepiped<Inter, Diff,  Inter, Inter, Inter, Inter>, // 20
//    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  Inter>, // 21
//    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  Inter>, // 22
//    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, Inter>, // 23
//    Parallelepiped<Spec,  Inter, Diff,  Inter, Spec,  Diff >, // 24
//    Parallelepiped<Inter, Inter, Inter, Diff,  Spec,  Diff >, // 25
//    
//    Parallelepiped<Diff,  Spec,  Diff,  Diff,  Inter, PeriP>, // 26
//    Parallelepiped<Diff,  Inter, Diff,  Inter, Diff,  PeriP>, // 27
//    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  PeriP>, // 28
//    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  PeriP>, // 29
//    Parallelepiped<Inter, Diff,  Inter, Inter, Diff,  PeriP>, // 30
//    Parallelepiped<Inter, Inter, Inter, Diff,  Diff,  PeriP>, // 31
//    Parallelepiped<Diff,  Spec,  Inter, Diff,  Inter, PeriP>  // 32
//    > SdomCont;
//    
//private:
//    template<int I>
//    struct Sdom : result_of::value_at_c<SdomCont, I>
//    {};
//    
//    SdomCont sdomCont_;
//    
//public:
//    OctetDomain();
//    OctetDomain(const Eigen::Matrix<double, 5, 1>& dim,
//                const Eigen::Matrix<long, 5, 1>& div,
//                double dT);
//    
//    template<int I>
//    typename result_of::at_c<SdomCont, I>::type sdom()
//    {
//        return fusion::at_c<I>(sdomCont_);
//    }
//};

#endif
