//
//  x.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 2/26/15.
//
//

#ifndef __montecarlocpp__x__
#define __montecarlocpp__x__

#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/view.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <iostream>

namespace mpl = boost::mpl;
namespace fusion = boost::fusion;
namespace result_of = boost::fusion::result_of;

using mpl::_;


class A {};
template<int N>
class B : public A {};

//template<bool I, bool J, bool K>
//struct BoolVec
//: public mpl::vector3<mpl::bool_<I>, mpl::bool_<J>, mpl::bool_<K>> {};
//
//template<typename V>
//struct CountTrue
//: public mpl::count_if<V, _> {};

template<bool I, bool J, bool K>
class C : public B<mpl::count_if<mpl::vector3_c<bool, I, J, K>, _>::value> {
    
};

BOOST_MPL_ASSERT_RELATION((mpl::count_if<mpl::vector3_c<bool,true,true,true>, _>::value), ==, 3);
BOOST_MPL_ASSERT((boost::is_base_of<B<3>, C<true,true,true>>));

struct Interface {
    typedef boost::is_base_of<Interface, _> L;
    virtual void f() = 0;
};

class XBase {
private:
    int i_;
public:
    XBase(int i = 0) : i_(i) {}
    int i() const { return i_; }
    void i(int val) { i_ = val; }
};

class XDerived : public XBase {
public:
    XDerived(int i = 0) : XBase(i) {}
};

class XInterface : public XBase, public Interface {
public:
    XInterface(int i = 0) : XBase(i) {}
    void f() { i(0); }
};

class XIDerived : public XInterface {
public:
    XIDerived(int i = 0) : XInterface(i) {}
};

namespace {
    typedef boost::add_pointer<_> PtrL;
    typedef boost::add_pointer<boost::add_const<_>> ConstPtrL;

    struct GetPtrF {
        template<typename T>
        T* operator()(const T& t) const {
            return const_cast<T*>(&t);
        }
    };

    struct GetConstPtrF {
        template<typename T>
        const T* operator()(const T& t) const {
            return &t;
        }
    };
        
    struct JoinISeqL {
        template<typename V, typename T>
        struct apply {
            typedef typename mpl::joint_view<V, typename T::ISeq>::type type;
        };
    };
    
    struct JoinIPtrF {
        template<typename V, typename T>
        fusion::joint_view<const V, const typename T::IPtrCont>
        operator()(const V& v, const T& t) const {
            return fusion::joint_view<const V, const typename T::IPtrCont>
            (v, t.iPtrCont());
        }
    };
};

template<typename V>
class YBase {
public:
    typedef typename V::type XSeq;
    typedef typename mpl::filter_view< XSeq, Interface::L >::type ISeq;
    typedef typename mpl::transform_view< XSeq, PtrL >::type XPtrSeq;
    typedef typename mpl::transform_view< ISeq, ConstPtrL >::type IPtrSeq;
    
    typedef typename result_of::as_vector< XSeq >::type XCont;
    typedef typename result_of::as_vector< XPtrSeq >::type XPtrCont;
    typedef typename result_of::as_vector< IPtrSeq >::type IPtrCont;
    template<int I>
    struct XElem : mpl::at_c<V, I> {};
private:
    XPtrCont xPtrCont_;
    IPtrCont iPtrCont_;
protected:
    void initPtrCont(const XCont& cont) {
        xPtrCont_ = fusion::transform(cont, GetPtrF());
        fusion::filter_view< const XCont, Interface::L > view(cont);
        iPtrCont_ = fusion::transform(view, GetConstPtrF());
    }
public:
    const XPtrCont& xPtrCont() const { return xPtrCont_; }
    const IPtrCont& iPtrCont() const { return iPtrCont_; }
    template<int I>
    typename result_of::at_c<const XPtrCont, I>::type at() const {
        return fusion::at_c<I>(xPtrCont_);
    }
};

template<typename V>
class YInterface : public YBase<V>, public Interface {
public:
    void f() {}
};

template<typename X0, typename X1, typename X2,
         typename X3 = X0, typename X4 = X1, typename X5 = X2>
class YIDerived : public YInterface< mpl::vector6<X0, X1, X2, X3, X4, X5> > {
private:
    typedef mpl::vector6<X0, X1, X2, X3, X4, X5> Vec;
    typename YBase<Vec>::XCont xCont_;
public:
    YIDerived() {}
    YIDerived(int i) : YInterface<Vec>(),
    xCont_(X0(i), X1(i+1), X2(i+2), X3(i+3), X4(i+4), X5(i+5)) {
        YBase<Vec>::initPtrCont(xCont_);
        YBase<Vec>::template at<2>()->f();
    }
};

template<typename V>
class ZBase {
public:
    typedef typename V::type YSeq;
    typedef typename mpl::filter_view< YSeq, Interface::L >::type ISeqThis;
    typedef typename mpl::fold< YSeq, ISeqThis, JoinISeqL >::type ISeq;
    BOOST_MPL_ASSERT((mpl::equal<ISeq, mpl::vector<YIDerived<XDerived, XDerived, XIDerived>, XIDerived, XIDerived>>));
    
    typedef typename mpl::transform_view< YSeq, ConstPtrL >::type YPtrSeq;
    typedef typename mpl::transform_view< ISeq, ConstPtrL >::type IPtrSeq;
    BOOST_MPL_ASSERT((mpl::equal<IPtrSeq, mpl::vector<const YIDerived<XDerived, XDerived, XIDerived>*, const XIDerived*, const XIDerived*>>));
    
    typedef typename result_of::as_vector< YSeq >::type YCont;
    typedef typename result_of::as_vector< YPtrSeq >::type YPtrCont;
    typedef typename result_of::as_vector< IPtrSeq >::type IPtrCont;
    BOOST_MPL_ASSERT((mpl::equal<IPtrCont, fusion::vector<const YIDerived<XDerived, XDerived, XIDerived>*, const XIDerived*, const XIDerived*>>));
    
    template<int I>
    struct YElem : mpl::at_c<V, I> {};
//    template<int I, int J>
//    struct XElem : YElem<I>::type::template XElem<J> {};
private:
    YPtrCont yPtrCont_;
    IPtrCont iPtrCont_;
protected:
    void initPtrCont(const YCont& cont) {
        yPtrCont_ = fusion::transform(cont, GetConstPtrF());
        fusion::filter_view< const YCont, Interface::L > view(cont);
        iPtrCont_ = fusion::fold(cont,
                                 fusion::transform(view, GetConstPtrF()),
                                 JoinIPtrF());
    }
public:
    const YPtrCont& yPtrCont() const { return yPtrCont_; }
    const IPtrCont& iPtrCont() const { return iPtrCont_; }
    template<int I>
    typename result_of::at_c<const YPtrCont, I>::type at() const {
        return fusion::at_c<I>(yPtrCont_);
    }
};

class ZDerived
: public ZBase<mpl::vector1<YIDerived<XDerived, XDerived, XIDerived>>> {
private:
    typedef mpl::vector1<YIDerived<XDerived, XDerived, XIDerived>> Vec;
    YCont yCont_;
public:
    ZDerived() : ZBase<Vec>(), yCont_(YElem<0>::type(1)) {
        initPtrCont(yCont_);
        at<0>();
    }
};

#endif
