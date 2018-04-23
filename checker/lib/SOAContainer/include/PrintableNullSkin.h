/** @file PrintableNullSkin.h
 *
 * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
 * @date 2016-11-16
 *
 * @brief make underlying tuple printable for whoever needs it
 */

#ifndef PRINTABLENULLSKIN_H
#define PRINTABLENULLSKIN_H

#include <sstream>

/// namespace to encapsulate SOA stuff
namespace SOA {
    /// hide implementation details
    namespace impl {
        template <std::size_t LEN> struct printer;
        /// print a tuple, recursion anchor
        template <>
        struct printer<std::size_t(0)> {
            template <typename T>
            void operator()(std::ostringstream&, const T&) const
            { }
        };
        /// print a tuple, specialisation for size 1
        template <>
        struct printer<std::size_t(1)> {
            template <typename T>
            void operator()(std::ostringstream& os, const T& t) const
            {
                printer<0>()(os, t);
                os << t.template get<0>();
            }
        };
        /// print a tuple, LEN is tuple size
        template <std::size_t LEN>
        struct printer {
            template <typename T>
            void operator()(std::ostringstream& os, const T& t) const
            {
                printer<LEN - 1>()(os, t);
                os << ", " << t.template get<LEN - 1>();
            }
        };
    }
    
    /// make tuple convertible to string
    template <typename T>
    struct PrintableNullSkin : T
    {
        /// constructor - forward to underlying proxy
        using T::T;
        /// assignment operator - forward to underlying proxy
        using T::operator=;
    
        /// conversion to string
        operator std::string() const {
            std::ostringstream os;
            impl::printer<T::parent_type::fields_typelist::size()>()(
                    os, *this);
            return os.str();
        }
    };
    
    //// operator<< on ostream for a PrintableNullSkin<T>
    template <typename T>
    std::ostream& operator<<(std::ostream& os,
            const PrintableNullSkin<T>& printable)
    { return os << "{" << std::string(printable) << "}"; }
} // namespace SOA

#endif // PRINTABLENULLSKIN_H

// vim: sw=4:tw=78:ft=cpp:et
