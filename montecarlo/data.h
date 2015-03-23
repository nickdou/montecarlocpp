//
//  data.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 3/10/15.
//
//

#ifndef __montecarlocpp__data__
#define __montecarlocpp__data__

#define BOOST_RESULT_OF_USE_TR1

#include <Eigen/Core>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/if.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <map>
#include <iomanip>
#include <iostream>

class Collection
: public boost::array< boost::multi_array_types::size_type, 3 > {
public:
    typedef boost::multi_array_types::size_type Size;
    typedef boost::array<Size, 3> Base;
    typedef Eigen::Matrix<Size, 3, 1> Vector3s;
    Collection() {}
    Collection(const Base& array) : Base(array) {}
    Collection(Size i, Size j, Size k) : Base() {
        operator[](0) = i;
        operator[](1) = j;
        operator[](2) = k;
    }
    template<typename Derived>
    explicit Collection(const Eigen::DenseBase<Derived>& index) : Base() {
        operator[](0) = index(0);
        operator[](1) = index(1);
        operator[](2) = index(2);
    }
    Vector3s vector() {
        return Eigen::Map<Vector3s>(data());
    }
};

template<typename Derived, typename Enable = void>
struct ZeroF {
    Derived operator()() const {
        return Derived::Zero();
    }
};

template<typename T>
struct ZeroF<T, typename boost::enable_if< boost::is_scalar<T> >::type> {
    T operator()() const {
        return T(0);
    }
};

template<typename Derived, typename Enable = void>
struct PrintF {
    Eigen::IOFormat fmt_;
    PrintF(const Eigen::IOFormat& fmt) : fmt_(fmt) {
        fmt_.precision = 0;
    }
    void operator()(std::ostream& os, const Derived* ptr) const {
        os << (*ptr).format(fmt_);
    }
};

template<typename T>
struct PrintF<T, typename boost::enable_if< boost::is_scalar<T> >::type> {
    int width_;
    PrintF(const Eigen::IOFormat& fmt) : width_(fmt.width) {}
    void operator()(std::ostream& os, const T* ptr) const {
        os << std::setw(width_);
        os << *ptr;
    }
};

template<typename T>
struct Vectorizable : boost::is_same<T, Eigen::Vector4d> {};

template<typename T>
struct DataAlloc
: boost::mpl::if_< Vectorizable<T>,
                   Eigen::aligned_allocator<T>,
                   std::allocator<T> > {};

template<typename T>
class Data : public boost::multi_array< T, 3, typename DataAlloc<T>::type > {
public:
    typedef typename DataAlloc<T>::type Allocator;
    typedef boost::multi_array<T, 3, Allocator> Base;
    typedef boost::fortran_storage_order Order;
    typedef typename Base::size_type Size;
    typedef Eigen::Matrix<Size, 3, 1> Vector3s;
private:
    Vector3s shape_;
public:
    Data() : Base(Collection(0,0,0), Order()), shape_(0,0,0) {}
    Data(const Data& data)
    : Base(static_cast<const Base&>(data)), shape_(data.shape_)
    {}
    Data(const Base& arr)
    : Base(shapeColl(arr), Order()), shape_(shapeColl(arr).vector())
    {
        Base::operator=(arr);
    }
    template<typename Coll>
    Data(const Coll& coll)
    : Base(coll, Order()), shape_(coll[0], coll[1], coll[2])
    {
        setZero();
    }
    void setZero() {
        ZeroF<T> zero;
        std::fill(Base::data(), Base::data() + Base::num_elements(), zero());
    }
    Data& operator+=(const Data& data) {
        BOOST_ASSERT_MSG(shape_ == data.shape_, "Data shapes not consistent");
        for (Size i = 0; i < Base::num_elements(); ++i) {
            Base::data()[i] += data.data()[i];
        }
        return *this;
    }
private:
    static Collection shapeColl(const Base& array) {
        const Size* shape = array.shape();
        return Collection(shape[0], shape[1], shape[2]);
    }
    template<bool Row>
    void printArray(std::ostream& os, typename Base::index idx,
                    typename Vector3s::Index r, typename Vector3s::Index c,
                    const Eigen::IOFormat& fmt) const
    {
        PrintF<T> printElem(fmt);
        Size n = 0, rows = shape_(r), cols = shape_(c);
        for (Size i = 0; i < rows; ++i) {
            if (!Row) n = i;
            for (Size j = 0; j < cols; ++j) {
                printElem(os, Base::data() + idx + n);
                os << ' ';
                n += (Row ? 1 : rows);
            }
            os << std::endl;
        }
    }
public:
    void print(std::ostream& os, const Eigen::IOFormat& fmt) const {
        if (Base::num_elements() == 0) return;
        BOOST_ASSERT_MSG(boost::general_storage_order<3>(Base::storage_order())
                         == boost::fortran_storage_order(),
                         "Invalid storage order");
        BOOST_ASSERT_MSG(fmt.precision >= 0, "Full precision not supported");
        BOOST_ASSERT_MSG(fmt.width >= 0, "Column alignment not supported");
        
        os << std::scientific;
        if (fmt.precision) os << std::setprecision(fmt.precision);
        
        typename Vector3s::Index idx = 0;
        switch ((shape_.array() > 1).count()) {
            case 0:
                printArray<true>(os, 0, 0, 1, fmt);
                break;
            case 1:
                (shape_.array() > 1).maxCoeff(&idx);
                printArray<true>(os, 0, (idx+2)%3, idx, fmt);
                break;
            case 2:
                (shape_.array() == 1).maxCoeff(&idx);
                if (idx == 1) {
                    printArray<true>(os, 0, 2, 0, fmt);
                    
                } else {
                    printArray<false>(os, 0, (idx+1)%3, (idx+2)%3, fmt);
                }
                break;
            case 3:
                for (Size k = 0; k < shape_(2); ++k) {
                    if (k > 0) os << std::endl;
                    printArray<false>(os, k * Base::strides()[2], 0, 1, fmt);
                }
                break;
        }
    }
};

//template<typename K, typename T>
//struct FieldAlloc
//: boost::mpl::if_< Vectorizable<T>,
//                   Eigen::aligned_allocator< std::pair<const K, Data<T> > >,
//                   std::allocator< std::pair<const K, Data<T> > > > {};

class Subdomain;

template<typename T>
class Field
: public std::map< const Subdomain*, Data<T> >
//                   std::less<const Subdomain*>,
//                   typename FieldAlloc<const Subdomain*, T>::type >
{
public:
//    typedef std::less<const Subdomain*> Compare;
//    typedef typename FieldAlloc<const Subdomain*, T>::type Allocator;
    typedef std::map<const Subdomain*, Data<T> > Base;
    Field() {}
    Field(const Base& map) : Base(map) {}
    Field& operator+=(const Field& fld) {
        BOOST_ASSERT_MSG(Base::size() == fld.size(),
                         "Cannot add fields of different size");
        typename Base::iterator thisIt = Base::begin();
        typename Base::const_iterator fldIt = fld.begin();
        for (; thisIt != Base::end(); ++thisIt, ++fldIt) {
            BOOST_ASSERT_MSG(thisIt->first == fldIt->first,
                             "Subdomains not the same");
            thisIt->second += fldIt->second;
        }
        return *this;
    }
    void print(std::ostream& os, int precision, int width) const {
        Eigen::IOFormat fmt(precision, width, " ", "", "", "", "[", "]");
        unsigned long i = 0;
        for (typename Base::const_iterator region = Base::begin();
             region != Base::end(); ++region)
        {
            if (i > 0) os << std::endl;
            os << "Region " << ++i << std::endl;
            region->second.print(os, fmt);
        }
    }
};

#endif
