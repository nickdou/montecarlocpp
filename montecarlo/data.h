//
//  data.h
//  montecarlocpp
//
//  Created by Nicholas Dou on 3/10/15.
//
//

#ifndef __montecarlocpp__data__
#define __montecarlocpp__data__

#include <Eigen/Core>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/assert.hpp>
#include <functional>
#include <algorithm>
#include <map>
#include <iomanip>
#include <iostream>

namespace lambda = boost::lambda;

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
    explicit Collection(const Size* shape) : Base() {
        operator[](0) = shape[0];
        operator[](1) = shape[1];
        operator[](2) = shape[2];
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

template<typename T, typename Enable = void>
struct PrintF {
    Eigen::IOFormat fmt;
    PrintF(int w) : fmt(0, w, " ", "", "", "", "[", "]") {}
    void operator()(std::ostream& os, const T* ptr) const {
        os << (*ptr).format(fmt);
    }
};

template<typename T>
struct PrintF<T, typename boost::enable_if< boost::is_scalar<T> >::type> {
    int width;
    PrintF(int w) : width(w) {}
    void operator()(std::ostream& os, const T* ptr) const {
        os << std::setw(width);
        os << *ptr;
    }
};

//template<typename T>
//struct Vectorizable : boost::is_same<T, Eigen::Vector4d> {};
//
//template<typename T>
//struct DataAlloc
//: boost::mpl::if_< Vectorizable<T>,
//                   Eigen::aligned_allocator<T>,
//                   std::allocator<T> > {};

template<typename T>
class Data : public boost::multi_array<T, 3> {
public:
    typedef T Type;
    typedef boost::multi_array<T, 3> Base;
    typedef boost::fortran_storage_order Order;
    typedef typename Base::size_type Size;
    typedef Eigen::Matrix<Size, 3, 1> Vector3s;
public:
    Data() : Base(Collection(0,0,0), Order()) {}
    Data(const Data& data) : Base(static_cast<const Base&>(data)) {}
    Data(const Base& arr) : Base(arr.shapeColl(), Order()) {
        Base::operator=(arr);
    }
    template<typename Coll>
    explicit Data(const Coll& coll) : Base(coll, Order())
    {}
    template<typename Coll>
    explicit Data(const Coll& coll, const T& value) : Base(coll, Order())
    {
        fill(value);
    }
    void fill(const T& value) {
        std::fill(Base::data(), Base::data() + Base::num_elements(), value);
    }
    Collection shapeColl() const {
        return Collection(Base::shape());
    }
    Vector3s shapeVec() const {
        return Eigen::Map<Vector3s>( const_cast<Size*>(Base::shape()) );
    }
    Data& operator+=(const Data& data) {
        BOOST_ASSERT_MSG(shapeVec() == data.shapeVec(),
                         "Data shapes not consistent");
        const T* it = data.data();
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 += *(lambda::var(it)++));
        return *this;
    }
    Data& operator-=(const Data& data) {
        BOOST_ASSERT_MSG(shapeVec() == data.shapeVec(),
                         "Data shapes not consistent");
        const T* it = data.data();
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 -= *(lambda::var(it)++));
        return *this;
    }
    Data& operator*=(const Data& data) {
        BOOST_ASSERT_MSG(shapeVec() == data.shapeVec(),
                         "Data shapes not consistent");
        const T* it = data.data();
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 *= *(lambda::var(it)++));
        return *this;
    }
    Data& operator/=(const Data& data) {
        BOOST_ASSERT_MSG(shapeVec() == data.shapeVec(),
                         "Data shapes not consistent");
        const T* it = data.data();
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 /= *(lambda::var(it)++));
        return *this;
    }
    template<typename O>
    Data& operator+=(const O& operand) {
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 += operand);
        return *this;
    }
    template<typename O>
    Data& operator-=(const O& operand) {
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 -= operand);
        return *this;
    }
    template<typename O>
    Data& operator*=(const O& operand) {
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 *= operand);
        return *this;
    }
    template<typename O>
    Data& operator/=(const O& operand) {
        std::for_each(Base::data(), Base::data() + Base::num_elements(),
                      lambda::_1 /= operand);
        return *this;
    }
    friend Data operator+(const Data& data1, const Data& data2) {
        Data result(data1);
        return result += data2;
    }
    friend Data operator-(const Data& data1, const Data& data2) {
        Data result(data1);
        return result -= data2;
    }
    friend Data operator*(const Data& data1, const Data& data2) {
        Data result(data1);
        return result *= data2;
    }
    friend Data operator/(const Data& data1, const Data& data2) {
        Data result(data1);
        return result /= data2;
    }
    template<typename O>
    friend Data operator+(const Data& data, const O& operand) {
        Data result(data);
        return result += operand;
    }
    template<typename O>
    friend Data operator-(const Data& data, const O& operand) {
        Data result(data);
        return result -= operand;
    }
    template<typename O>
    friend Data operator*(const Data& data, const O& operand) {
        Data result(data);
        return result *= operand;
    }
    template<typename O>
    friend Data operator/(const Data& data, const O& operand) {
        Data result(data);
        return result /= operand;
    }
    template<typename O>
    friend Data operator+(const O& operand, const Data& data) {
        return data + operand;
    }
    template<typename O>
    friend Data operator-(const O& operand, const Data& data) {
        Data result(data);
        std::for_each(result.data(), result.data() + result.num_elements(),
                      lambda::_1 = lambda::ret<T>(operand - lambda::_1));
        return result;
    }
    template<typename O>
    friend Data operator*(const O& operand, const Data& data) {
        return data * operand;
    }
    template<typename O>
    friend Data operator/(const O& operand, const Data& data) {
        Data result(data);
        std::for_each(result.data(), result.data() + result.num_elements(),
                      lambda::_1 = lambda::ret<T>(operand / lambda::_1));
        return result;
    }
    friend std::ostream& operator<<(std::ostream& os, const Data& data) {
        std::ios_base::fmtflags flags = std::cout.flags();
        data.print(os, 9, 16);
        return os << std::setiosflags(flags);
    }
    void print(std::ostream& os, int precision, int width) const {
        if (Base::num_elements() == 0) return;
        BOOST_ASSERT_MSG(boost::general_storage_order<3>(Base::storage_order())
                         == boost::fortran_storage_order(),
                         "Invalid storage order");
        BOOST_ASSERT_MSG(precision >= 0, "Full precision not supported");
        BOOST_ASSERT_MSG(width >= 0, "Column alignment not supported");
        
        os << std::scientific;
        if (precision) os << std::setprecision(precision);
        
        Vector3s vec = shapeVec();
        typename Vector3s::Index idx = 0;
        switch ((vec.array() > 1).count()) {
            case 0:
                printArray<true>(os, 0, 0, 1, width);
                break;
            case 1:
                (vec.array() > 1).maxCoeff(&idx);
                printArray<true>(os, 0, (idx+2)%3, idx, width);
                break;
            case 2:
                (vec.array() == 1).maxCoeff(&idx);
                if (idx == 1) {
                    printArray<true>(os, 0, 2, 0, width);
                    
                } else {
                    printArray<false>(os, 0, (idx+1)%3, (idx+2)%3, width);
                }
                break;
            case 3:
                for (Size k = 0; k < vec(2); ++k) {
                    if (k > 0) os << std::endl;
                    printArray<false>(os, k * Base::strides()[2], 0, 1, width);
                }
                break;
        }
    }
private:
    template<bool Row>
    void printArray(std::ostream& os, typename Base::index idx,
                    typename Vector3s::Index r, typename Vector3s::Index c,
                    int width) const
    {
        PrintF<T> printElem(width);
        Vector3s vec = shapeVec();
        Size n = 0, rows = vec(r), cols = vec(c);
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
};

class Subdomain;

template<typename T>
class Field : public std::map< const Subdomain*, Data<T> > {
public:
    typedef T Type;
    typedef std::map<const Subdomain*, Data<T> > Base;
    typedef typename Base::value_type Value;
    Field() {}
    Field(const Base& map) : Base(map) {}
    Field& operator+=(const Field& fld) {
        BOOST_ASSERT_MSG(Base::size() == fld.size(),
                         "Field shapes not consistent");
        typename BinaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 += lambda::_2);
        std::for_each(Base::begin(), Base::end(), BinaryOpF(fun, fld.begin()));
        return *this;
    }
    Field& operator-=(const Field& fld) {
        BOOST_ASSERT_MSG(Base::size() == fld.size(),
                         "Field shapes not consistent");
        typename BinaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 -= lambda::_2);
        std::for_each(Base::begin(), Base::end(), BinaryOpF(fun, fld.begin()));
        return *this;
    }
    Field& operator*=(const Field& fld) {
        BOOST_ASSERT_MSG(Base::size() == fld.size(),
                         "Field shapes not consistent");
        typename BinaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 *= lambda::_2);
        std::for_each(Base::begin(), Base::end(), BinaryOpF(fun, fld.begin()));
        return *this;
    }
    Field& operator/=(const Field& fld) {
        BOOST_ASSERT_MSG(Base::size() == fld.size(),
                         "Field shapes not consistent");
        typename BinaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 /= lambda::_2);
        std::for_each(Base::begin(), Base::end(), BinaryOpF(fun, fld.begin()));
        return *this;
    }
    template<typename O>
    Field& operator+=(const O& operand) {
        typename UnaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 += operand);
        std::for_each(Base::begin(), Base::end(), UnaryOpF(fun));
        return *this;
    }
    template<typename O>
    Field& operator-=(const O& operand) {
        typename UnaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 -= operand);
        std::for_each(Base::begin(), Base::end(), UnaryOpF(fun));
        return *this;
    }
    template<typename O>
    Field& operator*=(const O& operand) {
        typename UnaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 *= operand);
        std::for_each(Base::begin(), Base::end(), UnaryOpF(fun));
        return *this;
    }
    template<typename O>
    Field& operator/=(const O& operand) {
        typename UnaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 /= operand);
        std::for_each(Base::begin(), Base::end(), UnaryOpF(fun));
        return *this;
    }
    friend Field operator+(const Field& fld1, const Field& fld2) {
        Field result(fld1);
        return result += fld2;
    }
    friend Field operator-(const Field& fld1, const Field& fld2) {
        Field result(fld1);
        return result -= fld2;
    }
    friend Field operator*(const Field& fld1, const Field& fld2) {
        Field result(fld1);
        return result *= fld2;
    }
    friend Field operator/(const Field& fld1, const Field& fld2) {
        Field result(fld1);
        return result /= fld2;
    }
    template<typename O>
    friend Field operator+(const Field& fld, const O& operand) {
        Field result(fld);
        return result += operand;
    }
    template<typename O>
    friend Field operator-(const Field& fld, const O& operand) {
        Field result(fld);
        return result -= operand;
    }
    template<typename O>
    friend Field operator*(const Field& fld, const O& operand) {
        Field result(fld);
        return result *= operand;
    }
    template<typename O>
    friend Field operator/(const Field& fld, const O& operand) {
        Field result(fld);
        return result /= operand;
    }
    template<typename O>
    friend Field operator+(const O& operand, const Field& fld) {
        return fld + operand;
    }
    template<typename O>
    friend Field operator-(const O& operand, const Field& fld) {
        typename UnaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 =
                                      lambda::ret< Data<T> >
                                      (operand - lambda::_1));
        Field result(fld);
        std::for_each(result.begin(), result.end(), UnaryOpF(fun));
        return result;
    }
    template<typename O>
    friend Field operator*(const O& operand, const Field& fld) {
        return fld * operand;
    }
    template<typename O>
    friend Field operator/(const O& operand, const Field& fld) {
        typename UnaryOpF::Lambda fun;
        fun = lambda::ret< Data<T>& >(lambda::_1 =
                                      lambda::ret< Data<T> >
                                      (operand / lambda::_1));
        Field result(fld);
        std::for_each(result.begin(), result.end(), UnaryOpF(fun));
        return result;
    }
    friend std::ostream& operator<<(std::ostream& os, const Field& fld) {
        std::ios_base::fmtflags flags = std::cout.flags();
        fld.print(os, 9, 16);
        return os << std::setiosflags(flags);
    }
    void print(std::ostream& os, int precision, int width) const {
        long i = 0;
        for (typename Base::const_iterator sdom = Base::begin();
             sdom != Base::end(); ++sdom)
        {
            if (i > 0) os << std::endl;
            os << "Subdomain " << i++ << std::endl;
            sdom->second.print(os, precision, width);
        }
    }
private:
    struct BinaryOpF {
        typedef typename Base::value_type Value;
        typedef typename Base::const_iterator Iter;
        typedef boost::function<Data<T>&(Data<T>&, const Data<T>&)> Lambda;
        Lambda functor;
        Iter iterator;
        BinaryOpF(const Lambda& fun, const Iter& it)
        : functor(fun), iterator(it) {}
        void operator() (Value& pair1) {
            const Value& pair2 = *iterator;
            BOOST_ASSERT_MSG(pair1.first == pair2.first,
                             "Subdomains not consistent");
            functor(pair1.second, pair2.second);
            ++iterator;
        }
    };
    struct UnaryOpF {
        typedef typename Base::value_type Value;
        typedef boost::function<Data<T>&(Data<T>&)> Lambda;
        Lambda functor;
        UnaryOpF(const Lambda& fun) : functor(fun) {}
        void operator() (Value& pair) {
            functor(pair.second);
        }
    };
};

template<typename T>
class Statistics {
private:
    struct AccumF {
        Statistics* stats;
        AccumF(Statistics* st) : stats(st) {}
        void operator()(const T& x) {
            stats->add(x);
        }
    };
    long n_;
    T z_, m_, s_;
public:
    Statistics() : n_(0) {}
    Statistics(const T& zero) : n_(0), z_(zero), m_(zero), s_(zero) {}
    T mean() const { return m_; }
    T variance() const { return ( n_ < 2 ? z_ : T(s_/(n_ - 1)) ); }
    void add(const T& x) {
        n_++;
        T d = x - m_;
        m_ += d/n_;
        s_ += d*(x - m_);
    }
    AccumF accumulator() { return AccumF(this); }
    friend std::ostream& operator<<(std::ostream& os, const Statistics& stats) {
        std::ios_base::fmtflags flags = std::cout.flags();
        os << "Mean" << std::endl;
        os << stats.mean() << std::endl << std::endl;
        os << "Variance" << std::endl;
        os << stats.variance();
        return os << std::setiosflags(flags);
    }
};

#endif
