//
//  subdomain.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 8/7/15.
//
//

#ifndef __montecarlocpp__subdomain__
#define __montecarlocpp__subdomain__

#include "boundary.h"
#include "phonon.h"
#include "random.h"
#include <Eigen/Core>
#include <boost/fusion/algorithm/transformation.hpp>
#include <boost/fusion/algorithm/iteration.hpp>
#include <boost/fusion/sequence/intrinsic.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/container/vector/vector10.hpp>
#include <boost/mpl/range_c.hpp>
#include <vector>

namespace mpl = boost::mpl;
namespace fusion = boost::fusion;
namespace result_of = boost::fusion::result_of;

using Eigen::Vector3d;
using Eigen::Matrix3d;

typedef Eigen::Matrix<long, 3, 1> Vector3l;

class Subdomain
{
public:
    typedef std::vector<const Subdomain*> Pointers;
    
private:
    Boundary::Pointers bdryPtrs_;
    Emitter::Pointers emitPtrs_;
    
    double vol_;
    Vector3d o_;
    Matrix3d mat_, inv_;
    Vector3l div_, shape_, max_;
    int accum_;
    double eps_;
    
public:
    Subdomain();
    Subdomain(double vol, const Vector3d& o, const Matrix3d& mat,
              const Vector3l& div);
    Subdomain(const Subdomain& sdom);
    Subdomain& operator=(const Subdomain& sdom);
    virtual ~Subdomain();
    
    bool isInit() const;
    bool isInside(const Vector3d& pos) const;
    const Boundary::Pointers& bdryPtrs() const;
    const Emitter::Pointers& emitPtrs() const;
    
    const Vector3d& origin() const;
    const Matrix3d& matrix() const;
    const Vector3l& shape() const;
    
    int accumFlag() const;
    Vector3d coord(const Vector3d& pos) const;
    Vector3l coord2index(const Vector3d& coord) const;
    
    const Boundary* advect(Phonon& phn, double vel) const;
    
    double vol() const;
    virtual double cellVol(const Vector3l& index) const = 0;
    
protected:
    void addBdry(Boundary* bdry);
    void addBdry(EmitBoundary* bdry);
    
    class AddBdryF;
    friend class AddBdryF;
};

class Subdomain::AddBdryF
{
private:
    Subdomain* sdom_;
    
public:
    AddBdryF(Subdomain* sdom);
    
    void operator()(Boundary& bdry) const;
    void operator()(EmitBoundary& bdry) const;
};

class EmitSubdomain : public Subdomain, public Emitter
{
private:
    Vector3d gradT_;
    Matrix3d rot_;
    
public:
    EmitSubdomain();
    EmitSubdomain(double vol, const Vector3d& o, const Matrix3d& mat,
                  const Vector3l& div, const Vector3d& gradT);
    virtual ~EmitSubdomain();
    
    const Subdomain* emitSdom() const;
    const Boundary* emitBdry() const;
    double emitWeight() const;
    
private:
    Vector3d drawDir(Rng& gen) const;
    bool emitSign(const Vector3d&, const Vector3d& dir) const;
};

//----------------------------------------
//  Subdomain Templates
//----------------------------------------

namespace ParallelepipedImpl
{
    double cellVol(const Vector3l& index, const Vector3l& shape, double vol);
    Vector3d drawPos(const Vector3d& o, const Matrix3d& mat, Rng& gen);
}

template<typename Bac,       typename Lef,       typename Bot,
         typename Fro = Bac, typename Rig = Lef, typename Top = Bot>
class Parallelepiped : public EmitSubdomain
{
public:
    typedef fusion::vector6<Bac, Lef, Bot, Fro, Rig, Top> BdryCont;
    
private:
    typedef Parallelogram Par;
    
    BdryCont bdryCont_;
    
public:
    Parallelepiped()
    : EmitSubdomain()
    {}
    
    Parallelepiped(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
                   const Vector3d& gradT = Vector3d::Zero(),
                   const Eigen::Matrix<double, 6, 1>& T =
                   Eigen::Matrix<double, 6, 1>::Zero())
    : EmitSubdomain(mat.determinant(), o, mat, div, gradT),
    bdryCont_(Bac(o,              Par(mat.col(1), mat.col(2)), T(0)),
              Lef(o,              Par(mat.col(2), mat.col(0)), T(1)),
              Bot(o,              Par(mat.col(0), mat.col(1)), T(2)),
              Fro(o + mat.col(0), Par(mat.col(2), mat.col(1)), T(3)),
              Rig(o + mat.col(1), Par(mat.col(0), mat.col(2)), T(4)),
              Top(o + mat.col(2), Par(mat.col(1), mat.col(0)), T(5)))
    {
        init();
    }
    
    Parallelepiped(const Parallelepiped& p)
    : EmitSubdomain(p), bdryCont_(p.bdryCont_)
    {
        init();
    }
    
    Parallelepiped& operator=(const Parallelepiped& p)
    {
        EmitSubdomain::operator=(p);
        bdryCont_ = p.bdryCont_;
        
        init();
        return *this;
    }
    
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry()
    {
        return fusion::at_c<I>(bdryCont_);
    }
    
    double cellVol(const Vector3l& index) const
    {
        return ParallelepipedImpl::cellVol(index, shape(), vol());
    }
    
private:
    void init()
    {
        BOOST_ASSERT_MSG(vol() >= Dbl::min(),
                         "Volume too small, check vector order");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
    
    Vector3d drawPos(Rng& gen) const
    {
        return ParallelepipedImpl::drawPos(origin(), matrix(), gen);
    }
};

namespace TriangularPrismImpl
{
    double cellVol(const Vector3l& index, const Vector3l& shape, double vol);
    Vector3d drawPos(const Vector3d& o, const Matrix3d& mat, Rng& gen);
}

template<typename Bac, typename Lef, typename Bot,
         typename Dia,               typename Top = Bot>
class TriangularPrism : public EmitSubdomain
{
public:
    typedef fusion::vector5<Bac, Lef, Bot, Dia, Top> BdryCont;
    
private:
    typedef Parallelogram Par;
    typedef Triangle Tri;
    
    BdryCont bdryCont_;

public:
    TriangularPrism()
    : EmitSubdomain()
    {}
    
    TriangularPrism(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
                    const Vector3d& gradT = Vector3d::Zero(),
                    const Eigen::Matrix<double, 5, 1>& T =
                    Eigen::Matrix<double, 5, 1>::Zero())
    : EmitSubdomain(mat.determinant() / 2., o, mat, div, gradT),
    bdryCont_(Bac(o,              Par(mat.col(1),  mat.col(2)), T(0)),
              Lef(o,              Par(mat.col(2),  mat.col(0)), T(1)),
              Bot(o,              Tri(mat.col(0),  mat.col(1)), T(2)),
              Dia(o + mat.col(0), Par(mat.col(2),
                                      mat.col(1) - mat.col(0)), T(3)),
              Top(o + mat.col(2), Tri(mat.col(1),  mat.col(0)), T(4)))
    {
        init();
    }
    
    TriangularPrism(const TriangularPrism& t)
    : EmitSubdomain(t), bdryCont_(t.bdryCont_)
    {
        init();
    }
    
    TriangularPrism& operator=(const TriangularPrism& t)
    {
        EmitSubdomain::operator=(t);
        bdryCont_ = t.bdryCont_;
        
        init();
        return *this;
    }
    
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry()
    {
        return fusion::at_c<I>(bdryCont_);
    }
    
    double cellVol(const Vector3l& index) const
    {
        return TriangularPrismImpl::cellVol(index, shape(), vol());
    }
    
private:
    void init()
    {
        BOOST_ASSERT_MSG(vol() >= Dbl::min(),
                         "Volume too small, check vector order");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
    
    Vector3d drawPos(Rng& gen) const
    {
        return TriangularPrismImpl::drawPos(origin(), matrix(), gen);
    }
};

namespace TetrahedronImpl
{
    double cellVol(const Vector3l& index, const Vector3l& shape, double vol);
    Vector3d drawPos(const Vector3d& o, const Matrix3d& mat, Rng& gen);
}

template<typename Bac, typename Lef, typename Bot, typename Dia>
class Tetrahedron : public EmitSubdomain
{
public:
    typedef fusion::vector4<Bac, Lef, Bot, Dia> BdryCont;
    
private:
    typedef Triangle Tri;
    
    BdryCont bdryCont_;
    
public:
    Tetrahedron()
    : EmitSubdomain()
    {}
    
    Tetrahedron(const Vector3d& o, const Matrix3d& mat, const Vector3l& div,
                const Vector3d& gradT = Vector3d::Zero(),
                const Eigen::Matrix<double, 4, 1>& T =
                Eigen::Matrix<double, 4, 1>::Zero())
    : EmitSubdomain(mat.determinant() / 6., o, mat, div, gradT),
    bdryCont_(Bac(o,              Tri(mat.col(1),  mat.col(2)), T(0)),
              Lef(o,              Tri(mat.col(2),  mat.col(0)), T(1)),
              Bot(o,              Tri(mat.col(0),  mat.col(1)), T(2)),
              Dia(o + mat.col(0), Tri(mat.col(2) - mat.col(0),
                                      mat.col(1) - mat.col(0)), T(3)))
    {
        init();
    }
    
    Tetrahedron(const Tetrahedron& t)
    : EmitSubdomain(t), bdryCont_(t.bdryCont_)
    {
        init();
    }
    
    Tetrahedron& operator=(const Tetrahedron& t)
    {
        EmitSubdomain::operator=(t);
        bdryCont_ = t.bdryCont_;
        
        init();
        return *this;
    }
    
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry()
    {
        return fusion::at_c<I>(bdryCont_);
    }
    
    double cellVol(const Vector3l& index) const
    {
        return TetrahedronImpl::cellVol(index, shape(), vol());
    }
private:
    void init()
    {
        BOOST_ASSERT_MSG(vol() >= Dbl::min(),
                         "Volume too small, check vector order");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
    
    Vector3d drawPos(Rng& gen) const
    {
        return TetrahedronImpl::drawPos(origin(), matrix(), gen);
    }
};

namespace PrismImpl
{
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
    
    Matrix3d matBase(const Matrix3Xd& mat);
    Eigen::VectorXd volume(const Matrix3Xd& mat);
    
    double cellVol(const Vector3l& index, const Vector3l& shape, double vol);
    Vector3d drawPos(const Vector3d& o, const Matrix3Xd& mat,
                     const DiscreteDist& volDist, Rng& gen);
}

template<typename Bot, typename Top, typename Sid>
class Prism : public EmitSubdomain
{
private:
    static const int N = result_of::size<Sid>::type::value;
    
    typedef typename result_of::push_front<Sid, Top>::type TopSid;
    typedef typename result_of::push_front<TopSid, Bot>::type BotTopSid;
    
public:
    typedef typename result_of::as_vector<BotTopSid>::type BdryCont;
    
private:
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
    
    BdryCont bdryCont_;
    Matrix3Xd mat_;
    DiscreteDist volDist_;

public:
    Prism()
    : EmitSubdomain()
    {}
    
    Prism(const Vector3d& o, const Matrix3Xd& mat,
          const Vector3d& gradT = Vector3d::Zero(),
          const Eigen::VectorXd& T = Eigen::VectorXd::Zero(N + 2))
    : EmitSubdomain(PrismImpl::volume(mat).sum(), o,
                    PrismImpl::matBase(mat), Vector3l::Zero(), gradT),
    mat_(mat)
    {
        BOOST_ASSERT_MSG(mat.cols() == N, "Incorrect number of matrix columns");
        
        Eigen::VectorXd vol = PrismImpl::volume(mat);
        volDist_ = DiscreteDist(vol.data(), vol.data() + N - 2);
        
        Matrix3Xd matBot = mat.rightCols(N - 1);
        Matrix3Xd matTop = mat.rightCols(N - 1).rowwise().reverse();
        
        Bot b = Bot(o,              Polygon<N>(matBot), T(0));
        Top t = Top(o + mat.col(0), Polygon<N>(matTop), T(1));
        
        InitSidesF initSides(o, mat, T.tail(N));
        Sid s = fusion::transform(mpl::range_c<int, 0, N>(), initSides);
        
        bdryCont_ = fusion::push_front( fusion::push_front(s, t), b);
        init();
    }
    
    Prism(const Prism& p)
    : EmitSubdomain(p), bdryCont_(p.bdryCont_), mat_(p.mat_),
    volDist_(p.volDist_)
    {
        init();
    }
    
    Prism& operator=(const Prism& p)
    {
        EmitSubdomain::operator=(p);
        bdryCont_ = p.bdryCont_;
        mat_ = p.mat_;
        volDist_ = p.volDist_;
        
        init();
        return *this;
    }
    
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry()
    {
        return fusion::at_c<I>(bdryCont_);
    }
    
    double cellVol(const Vector3l& index) const
    {
        return PrismImpl::cellVol(index, shape(), vol());
    }
    
private:
    class InitSidesF
    {
    private:
        Vector3d o_;
        Matrix3Xd mat_;
        Eigen::VectorXd T_;
        
    public:
        InitSidesF(const Vector3d& o, const Matrix3Xd& mat,
                   const Eigen::VectorXd& T)
        : o_(o), mat_(mat), T_(T)
        {}
        
        template<typename>
        struct result;
        
        template<typename I>
        struct result< InitSidesF(I) >
        {
            typedef typename result_of::value_at<Sid, I>::type type;
        };
        
        template<typename I>
        typename result_of::value_at<Sid, I>::type operator()(const I&) const
        {
            typedef typename result_of::value_at<Sid, I>::type S;
            const int ind = I::value;
            
            Vector3d p = Vector3d::Zero();
            if (ind != 0) p += mat_.col(ind);
            
            Vector3d i = mat_.col(0);
            
            Vector3d j = -p;
            if (ind != N - 1) j += mat_.col(ind + 1);
            
            return S(o_ + p, Parallelogram(i, j), T_(ind));
        }
    };
    
    void init()
    {
        BOOST_ASSERT_MSG(vol() >= Dbl::min(), "Volume too small");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
    
    Vector3d drawPos(Rng& gen) const
    {
        return PrismImpl::drawPos(origin(), matrix(), volDist_, gen);
    }
};

namespace PyramidImpl
{
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
    
    Matrix3d matBase(const Matrix3Xd& mat);
    Eigen::VectorXd volume(const Matrix3Xd& mat);
    
    double cellVol(const Vector3l& index, const Vector3l& shape, double vol);
    Vector3d drawPos(const Vector3d& o, const Matrix3Xd& mat,
                     const DiscreteDist& volDist, Rng& gen);
}

template<typename Bot, typename Sid>
class Pyramid : public EmitSubdomain
{
private:
    static const int N = result_of::size<Sid>::type::value;
    
    typedef typename result_of::push_front<Sid, Bot>::type BotSid;
    
public:
    typedef typename result_of::as_vector<BotSid>::type BdryCont;
    
private:
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3Xd;
    
    BdryCont bdryCont_;
    Matrix3Xd mat_;
    DiscreteDist volDist_;
    
public:
    Pyramid()
    : EmitSubdomain()
    {}
    
    Pyramid(const Vector3d& o, const Matrix3Xd& mat,
            const Vector3d& gradT = Vector3d::Zero(),
            const Eigen::VectorXd& T = Eigen::VectorXd::Zero(N + 1))
    : EmitSubdomain(PyramidImpl::volume(mat).sum(), o,
                    PyramidImpl::matBase(mat), Vector3l::Zero(), gradT),
    mat_(mat)
    {
        BOOST_ASSERT_MSG(mat.cols() == N, "Incorrect number of matrix columns");
        
        Eigen::VectorXd vol = PyramidImpl::volume(mat);
        volDist_ = DiscreteDist(vol.data(), vol.data() + N - 2);
        
        Matrix3Xd matBot = mat.rightCols(N - 1);
        
        Bot b = Bot(o, Polygon<N>(matBot), T(0));
        
        InitSidesF initSides(o, mat, T.tail(N));
        Sid s = fusion::transform(mpl::range_c<int, 0, N>(), initSides);
        
        bdryCont_ = fusion::push_front(s, b);
        init();
    }
    
    Pyramid(const Pyramid& p)
    : EmitSubdomain(p), bdryCont_(p.bdryCont_), mat_(p.mat_),
    volDist_(p.volDist_)
    {
        init();
    }
    
    Pyramid& operator=(const Pyramid& p)
    {
        EmitSubdomain::operator=(p);
        bdryCont_ = p.bdryCont_;
        mat_ = p.mat_;
        volDist_ = p.volDist_;
        
        init();
        return *this;
    }
    
    template<int I>
    typename result_of::at_c<BdryCont, I>::type bdry()
    {
        return fusion::at_c<I>(bdryCont_);
    }
    
    double cellVol(const Vector3l& index) const
    {
        return PyramidImpl::cellVol(index, shape(), vol());
    }
    
private:
    class InitSidesF
    {
    private:
        Vector3d o_;
        Matrix3Xd mat_;
        Eigen::VectorXd T_;
        
    public:
        InitSidesF(const Vector3d& o, const Matrix3Xd& mat,
                   const Eigen::VectorXd& T)
        : o_(o), mat_(mat), T_(T)
        {}
        
        template<typename>
        struct result;
        
        template<typename I>
        struct result< InitSidesF(I) >
        {
            typedef typename result_of::value_at<Sid, I>::type type;
        };
        
        template<typename I>
        typename result_of::value_at<Sid, I>::type operator()(const I&) const
        {
            typedef typename result_of::value_at<Sid, I>::type S;
            const int ind = I::value;
            
            Vector3d p = Vector3d::Zero();
            if (ind != 0) p += mat_.col(ind);
            
            Vector3d i = mat_.col(0);
            if (ind != 0) i -= mat_.col(ind);
            
            Vector3d j = -p;
            if (ind != N - 1) j += mat_.col(ind + 1);
            
            return S(o_ + p, Triangle(i, j), T_(ind));
        }
    };
    
    void init()
    {
        BOOST_ASSERT_MSG(vol() >= Dbl::min(), "Volume too small");
        fusion::for_each(bdryCont_, AddBdryF(this));
    }
    
    Vector3d drawPos(Rng& gen) const
    {
        return PyramidImpl::drawPos(origin(), matrix(), volDist_, gen);
    }
};


#endif
