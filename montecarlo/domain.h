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
#include <boost/fusion/container/vector.hpp>
#include <boost/assert.hpp>
#include <deque>
#include <vector>

namespace fusion = boost::fusion;
namespace result_of = boost::fusion::result_of;

class Domain;

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
    Data<T> initData() const {
        return Data<T>( Collection(grid_.shape().cwiseMax(1)) );
    }
    template<typename T>
    void accumulate(const Vector3d& begin, const Vector3d& end,
                    Data<T>& data, T quant) const
    {
        grid_.accumulate(begin, end, data, quant);
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
    const Subdomain* emitSdom() const { return this; }
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
        return grid_.matrix().determinant();
    }
    double cellVol(const Vector3l&) const {
        return grid_.cellVol();
    }
private:
    Vector3d drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        Vector3d coord(dist(gen), dist(gen), dist(gen));
        return grid_.origin() + grid_.matrix()*coord;
    }
};

class Domain {
protected:
    typedef Eigen::Matrix<long, 3, 1> Vector3l;
    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
public:
    typedef std::vector<const Subdomain*> SdomPtrs;
    typedef std::deque<const Emitter*> EmitPtrs;
private:
    SdomPtrs sdomPtrs_;
    EmitPtrs emitPtrs_;
protected:
    void addSdom(const Subdomain* sdom) {
        sdomPtrs_.push_back(sdom);
        emitPtrs_.insert(emitPtrs_.end(), sdom->emitPtrs().begin(),
                         sdom->emitPtrs().end());
    }
    void addSdom(const EmitSubdomain* sdom) {
        addSdom(static_cast<const Subdomain*>(sdom));
        if (sdom->emitWeight() != 0.) {
            emitPtrs_.push_front(static_cast<const Emitter*>(sdom));
        }
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
};

class BulkDomain : public Domain {
public:
    typedef Parallelepiped< PeriBoundary<Parallelogram>,
                            SpecBoundary, SpecBoundary > Sdom;
private:
    Sdom sdom_;
public:
    BulkDomain() : Domain() {}
    BulkDomain(const Vector3d& corner, const Vector3l& div, double deltaT)
    : sdom_(Vector3d::Zero(), corner.asDiagonal(), div,
            Vector3d(-deltaT/corner(0), 0., 0.))
//            Vector3d::Zero(), deltaT/2, 0., 0., -deltaT/2., 0., 0.)
    {
        makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(corner(0), 0., 0.));
        addSdom(&sdom_);
    }
};

class FilmDomain : public Domain {
public:
    typedef Parallelepiped< PeriBoundary<Parallelogram>,
                            DiffBoundary, SpecBoundary > Sdom;
private:
    Sdom sdom_;
public:
    FilmDomain() : Domain() {}
    FilmDomain(const Vector3d& corner, const Vector3l& div, double deltaT)
    : sdom_(Vector3d::Zero(), corner.asDiagonal(), div,
            Vector3d(-deltaT/corner(0), 0., 0.))
//            Vector3d::Zero(), deltaT/2, 0., 0., -deltaT/2., 0., 0.)
    {
        makePair(sdom_.bdry<0>(), sdom_.bdry<3>(), Vector3d(corner(0), 0., 0.));
        addSdom(&sdom_);
    }
};

class TeeDomain : public Domain {
private:
    typedef Eigen::DiagonalMatrix<double, 3> Diagonal;
    typedef SpecBoundary Spec;
    typedef DiffBoundary Diff;
    typedef InterBoundary Inter;
    typedef PeriBoundary<Parallelogram> Peri4;
public:
    // Back, Left, Bottom, Front, Right, Top
    typedef fusion::vector4<
            Parallelepiped<Peri4, Diff,  Spec, Inter, Diff,  Spec>,
            Parallelepiped<Inter, Spec,  Spec, Inter, Inter, Spec>,
            Parallelepiped<Diff,  Inter, Spec, Diff,  Spec,  Spec>,
            Parallelepiped<Inter, Diff,  Spec, Peri4, Diff,  Spec> > SdomCont;
private:
    SdomCont sdomCont_;
    template<int I>
    struct Sdom : result_of::value_at_c<SdomCont, I> {};
public:
    TeeDomain() : Domain() {}
    TeeDomain(const Eigen::Matrix<double, 5, 1>& dim,
              const Eigen::Matrix<long, 5, 1>& div,
              double deltaT)
    : sdomCont_(Sdom<0>::type(Vector3d::Zero(),
                              Diagonal(dim(0), dim(2), dim(4)),
                              Vector3l(div(0), div(2), div(4)),
                              Vector3d(-deltaT/(2*dim(0) + dim(1)), 0., 0.)),
                Sdom<1>::type(Vector3d(dim(0), 0., 0.),
                              Diagonal(dim(1), dim(2), dim(4)),
                              Vector3l(div(1), div(2), div(4)),
                              Vector3d(-deltaT/(2*dim(0) + dim(1)), 0., 0.)),
                Sdom<2>::type(Vector3d(dim(0), dim(2), 0.),
                              Diagonal(dim(1), dim(3), dim(4)),
                              Vector3l(div(1), div(3), div(4)),
                              Vector3d(-deltaT/(2*dim(0) + dim(1)), 0., 0.)),
                Sdom<3>::type(Vector3d(dim(0) + dim(1), 0., 0.),
                              Diagonal(dim(0), dim(2), dim(4)),
                              Vector3l(div(0), div(2), div(4)),
                              Vector3d(-deltaT/(2*dim(0) + dim(1)), 0., 0.)))
    {
        makePair(sdom<0>().bdry<0>(), sdom<3>().bdry<3>(),
                 Vector3d(2*dim(0) + dim(1), 0., 0.));
        makePair(sdom<0>().bdry<3>(), sdom<1>().bdry<0>());
        makePair(sdom<1>().bdry<4>(), sdom<2>().bdry<1>());
        makePair(sdom<1>().bdry<3>(), sdom<3>().bdry<0>());
        fusion::for_each(sdomCont_, AddSdomF(this));
    }
    template<int I>
    typename result_of::at_c<SdomCont, I>::type sdom() {
        return fusion::at_c<I>(sdomCont_);
    }
};

#endif
