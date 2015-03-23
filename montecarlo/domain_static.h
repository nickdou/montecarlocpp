//
//  domain_static.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/27/15.
//
//

#ifndef __montecarlocpp__domain_static__
#define __montecarlocpp__domain_static__

#include "boundary.h"
#include "phonon.h"
#include "constants.h"
#include "types.h"
#include <Eigen/Core>
#include <boost/multi_array.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/view.hpp>
#include <boost/fusion/sequence/intrinsic.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector/vector10.hpp>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/assert.hpp>
#include <deque>
#include <vector>
#include <cmath>

namespace mpl = boost::mpl;
namespace fusion = boost::fusion;
namespace result_of = boost::fusion::result_of;

using mpl::_;

using Eigen::Array3i;
using Eigen::Array3d;
using Eigen::Vector3d;
using Eigen::Matrix3d;

typedef boost::multi_array<double, 3> MArrayXXXd;

namespace {
    typedef boost::add_pointer<_> PtrL;
//    typedef boost::add_pointer<boost::add_const<_>> ConstPtrL;
    struct GetPtrF {
        template<typename T>
        T* operator()(const T& t) const {
            return const_cast<T*>(&t);
        }
    };
//    struct GetConstPtrF {
//        template<typename T>
//        const T* operator()(const T& t) const {
//            return &t;
//        }
//    };
    struct IsInitF {
        template<typename T>
        bool operator()(const T& t) const {
            return t->isInit();
        }
    };
}

class DomainI;

class SubdomainI {
public:
    virtual bool isInit() const = 0;
    virtual const Boundary* advect(Phonon& phn, double& scat) const = 0;
};

template<typename V>
class Subdomain : public SubdomainI {
public:
    typedef typename V::type BdrySeq;
    typedef typename mpl::filter_view< BdrySeq, Emitter::L >::type EmitSeq;
    typedef typename mpl::transform_view< BdrySeq, PtrL >::type BdryPtrSeq;
    typedef typename mpl::transform_view< EmitSeq, PtrL >::type EmitPtrSeq;
    typedef typename result_of::as_vector< BdrySeq >::type BdryCont;
    typedef typename result_of::as_vector< BdryPtrSeq >::type BdryPtrCont;
    typedef typename result_of::as_vector< EmitPtrSeq >::type EmitPtrCont;
    template<int I>
    struct Bdry : mpl::at_c<BdrySeq, I> {};
private:
    BdryPtrCont bdryPtrCont_;
    EmitPtrCont emitPtrCont_;
    const DomainI* dom_;
    double vol_;
    struct IsInsideF;
    struct AdvectF;
protected:
    void initPtrCont(BdryCont& cont) {
        bdryPtrCont_ = fusion::transform(cont, GetPtrF());
        fusion::filter_view< BdryCont, Emitter::L > emitView(cont);
        emitPtrCont_ = fusion::transform(emitView, GetPtrF());
    }
    Subdomain(const DomainI* dom, const double vol)
    : dom_(dom), vol_(vol), bdryPtrCont_(), emitPtrCont_() {}
public:
    virtual ~Subdomain() {}
    const DomainI* dom() const { return dom_; }
    double vol() const { return vol_; }
    const BdryPtrCont& bdryPtrCont() const { return bdryPtrCont_; }
    const EmitPtrCont& emitPtrCont() const { return emitPtrCont_; }
    template<int I>
    typename Bdry<I>::type* bdry() const {
        return fusion::at_c<I>(bdryPtrCont_);
    }
    virtual bool isInit() const {
        return fusion::all(bdryPtrCont_, IsInitF());
    }
    virtual bool isInside(const EVector& pos) const {
        return fusion::all(bdryPtrCont_, IsInsideF(pos));
    }
    const Boundary* advect(Phonon& phn, double& scat) const {
        typedef fusion::vector<const Boundary*, double> State;
        const Boundary* bdry = 0;
        State initial(bdry, scat);
        State final(fusion::fold(bdryPtrCont_, initial, AdvectF(phn)));
        double dist = fusion::at_c<1>(final);
        phn.move(dist);
        scat -= dist;
        return fusion::at_c<0>(final);
    }
private:
    struct IsInsideF {
        const EVector& pos_;
        IsInsideF(const EVector& pos) : pos_(pos) {}
        template<typename B>
        bool operator()(const B& bdry) const {
            return bdry->signedDistance(pos_) < 0;
        }
    };
    struct AdvectF {
        typedef fusion::vector<const Boundary*, double> State;
        const Phonon& phn_;
        AdvectF(const Phonon& phn) : phn_(phn) {}
        template<typename B>
        State operator()(const State& state, const B& bdry) const {
            if (phn_.dir().dot(bdry->normal()) >= 0) return state;
            double dist = bdry->distance(phn_);
            BOOST_ASSERT_MSG(dist > 0 ||
                             phn_.pos().isApprox(phn_.line().pointAt(dist)),
                             "Phonon escaped from domain");
            if (dist >= fusion::at_c<1>(state)) return state;
            return State(bdry, dist);
        }
    };
};

template<typename V>
class EmitSubdomain : public Subdomain<V>, public Emitter {
private:
    typedef Subdomain<V> Base;
    EVector gradT_;
    EMatrix dir_;
protected:
    EmitSubdomain(const DomainI* dom, const double vol, const EVector& gradT)
    : Base(dom, vol), gradT_(gradT),
    dir_(tools::rotMatrix(gradT.normalized())) {}
public:
    virtual ~EmitSubdomain() {}
    const SubdomainI* emitSdom() const { return this; }
    const Boundary* emitBdry() const { return 0; }
    double emitWeight() const {
        return Base::vol() * gradT_.norm();
    }
    Phonon emit(const Phonon::Prop& prop, Rng& gen) const {
        BOOST_ASSERT_MSG(emitWeight() != 0, "No phonons to emit");
        EVector phnDir = dir_ * tools::drawAniso<true>(gen);
        return Phonon(prop,
                      drawPos(gen),
                      phnDir,
                      phnDir.dot(gradT_) < 0);
    }
};

template<typename Back, typename Left, typename Bottom,
typename Front = Back, typename Right = Left, typename Top = Bottom>
class Parallelepiped
: public EmitSubdomain< mpl::vector6<Back, Left, Bottom, Front, Right, Top> > {
public:
    typedef mpl::vector6<Back, Left, Bottom, Front, Right, Top> Vec;
    typedef EmitSubdomain<Vec> Base;
private:
    typename Subdomain<Vec>::BdryCont bdryCont_;
    EVector o_, i_, j_, k_;
public:
    Parallelepiped(const DomainI* dom, const EVector& o,
                   const EVector& i, const EVector& j, const EVector& k,
                   const EVector& gradT = EVector::Zero(),
                   const double TBack   = 0, const double TLeft  = 0,
                   const double TBottom = 0, const double TFront = 0,
                   const double TRight  = 0, const double TTop   = 0)
    : Base(dom, i.cross(j).dot(k), gradT), o_(o), i_(i), j_(j), k_(k),
    bdryCont_(Back  (this, o    , Parallelogram(j, k), TBack  ),
              Left  (this, o    , Parallelogram(k, i), TLeft  ),
              Bottom(this, o    , Parallelogram(i, j), TBottom),
              Front (this, o + i, Parallelogram(k, j), TFront ),
              Right (this, o + j, Parallelogram(i, k), TRight ),
              Top   (this, o + k, Parallelogram(j, i), TTop   )) {
        
        BOOST_ASSERT_MSG(Base::vol() > 0, "Vector order incorrect");
        BOOST_ASSERT_MSG(Base::vol() >= constants::min, "Volume too small");
        Base::initPtrCont(bdryCont_);
    }
private:
    EVector drawPos(Rng& gen) const {
        static UniformDist01 dist; // [0, 1)
        double r1 = dist(gen), r2 = dist(gen), r3 = dist(gen);
        return o_ + r1*i_ + r2*j_ + r3*k_;
    }
};

class DomainI {
public:
    virtual bool isInit() const = 0;
};

template<typename V>
class Domain : public DomainI {
private:
    struct JoinEmitSeqL {
        template<typename U, typename S>
        struct apply {
            typedef typename mpl::joint_view<U, typename S::EmitSeq>::type type;
        };
    };
    struct JoinEmitPtrF {
        template<typename U, typename S>
        fusion::joint_view<const U, const typename S::EmitPtrCont>
        operator()(const U& u, const S& sdom) const {
            return fusion::joint_view<const U, const typename S::EmitPtrCont>
            (u, sdom.emitPtrCont());
        }
    };
public:
    typedef typename V::type SdomSeq;
    typedef typename mpl::filter_view< SdomSeq, Emitter::L >::type EmitSeq0;
    typedef typename mpl::fold< SdomSeq, EmitSeq0, JoinEmitSeqL >::type EmitSeq;
    typedef typename mpl::transform_view< SdomSeq, PtrL >::type SdomPtrSeq;
    typedef typename mpl::transform_view< EmitSeq, PtrL >::type EmitPtrSeq;
    typedef typename result_of::as_vector< SdomSeq >::type SdomCont;
    typedef typename result_of::as_vector< SdomPtrSeq >::type SdomPtrCont;
    typedef typename result_of::as_vector< EmitPtrSeq >::type EmitPtrCont;
    template<int I>
    struct Sdom : mpl::at_c<V, I> {};
    template<int I, int J>
    struct Bdry : Sdom<I>::type::template Bdry<J> {};
private:
    SdomPtrCont sdomPtrCont_;
    EmitPtrCont emitPtrCont_;
    struct LocateF;
protected:
    void initPtrCont(SdomCont& cont) {
        sdomPtrCont_ = fusion::transform(cont, GetPtrF());
        fusion::filter_view< SdomCont, Emitter::L > emitView(cont);
        emitPtrCont_ = fusion::fold(cont,
                                    fusion::transform(emitView, GetPtrF()),
                                    JoinEmitPtrF());
    }
    Domain() : sdomPtrCont_(), emitPtrCont_() {}
public:
    virtual ~Domain() {}
    const SdomPtrCont& sdomPtrCont() const { return sdomPtrCont_; }
    const EmitPtrCont& emitPtrCont() const { return emitPtrCont_; }
    template<int I>
    typename Sdom<I>::type* sdom() const {
        return fusion::at_c<I>(sdomPtrCont_);
    }
    template<int I, int J>
    typename Bdry<I,J>::type* bdry() const {
        return sdom<I>()->template bdry<J>();
    }
    virtual bool isInit() const {
        return fusion::all(sdomPtrCont_, IsInitF());
    }
    virtual bool isInside(const EVector& pos) const {
        return locate(pos);
    }
    const SubdomainI* locate(const EVector& pos) const {
        SubdomainI* ptr = 0;
        return fusion::fold(sdomPtrCont_, ptr, LocateF(pos));
    }
private:
    struct LocateF {
        const EVector& pos_;
        LocateF(const EVector& pos) : pos_(pos) {}
        template<typename S>
        const SubdomainI*
        operator()(const SubdomainI* ptr, const S& sdom) const {
            if (ptr) return ptr;
            if (sdom->isInside(pos_)) return sdom;
            return 0;
        }
    };
};

class BulkDomain
: public Domain< mpl::vector1< Parallelepiped< SpecBoundary, SpecBoundary,
PeriBoundary<Parallelogram> > > > {
public:
    typedef mpl::vector1< Parallelepiped< SpecBoundary, SpecBoundary,
    PeriBoundary<Parallelogram> > > Vec;
    typedef Domain<Vec> Base;
private:
    SdomCont sdomCont_;
public:
    BulkDomain(const double x, const double y, const double z,
               const double deltaT)
    : Base(),
    sdomCont_( Sdom<0>::type(this, EVector::Zero(),
                             EVector(x,0,0), EVector(0,y,0), EVector(0,0,z),
                             EVector(0,0,deltaT/z)) ) {
        makePair(*bdry<0,2>(), *bdry<0,5>(), EVector(0,0,z));
        initPtrCont(sdomCont_);
    }
};

#endif
