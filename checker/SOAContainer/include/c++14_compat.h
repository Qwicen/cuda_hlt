/** @file c++14_compat.h
 *
 * @brief some required constructs from C++ 14 implemented using C++ 11
 */

#ifndef CPP14_COMPAT_H
#define CPP14_COMPAT_H

#if __cplusplus > 201103L // well, g++ and icpc have strange values here in c++14 mode
// C++14 - use the standard's facilities
#include <utility>
#elif __cplusplus >= 201103L
// C++11 - implement what's needed
// C++14 Compile-time integer sequences -- this can go once we use C++14...
namespace std {
    template<size_t... indexes>
    struct index_sequence {
        static constexpr size_t size() { return sizeof...(indexes); }
    };

    template<size_t currentIndex, size_t... indexes>
    struct make_index_sequence_helper;

    template<size_t...indexes>
    struct make_index_sequence_helper<0, indexes...> {
        using type = index_sequence<indexes...>;
    };

    template<size_t currentIndex, size_t... indexes>
    struct make_index_sequence_helper {
        using type = typename make_index_sequence_helper<currentIndex - 1,
                currentIndex - 1, indexes...>::type;
    };

    template<size_t N>
    struct make_index_sequence : public make_index_sequence_helper<N>::type
    { };
}
#else // __cplusplus
// not even C++11 support
#error "Your C++ compiler must support at least C++11."
#endif // __cplusplus

// FIXME: what is the defined value of compliant C++17 compilers?
#if __cplusplus > 201402L
#include <type_traits>
#else
namespace std {
    /// little helper for the SFINAE idiom we'll use (not required in C++17)
    template<typename... Ts> struct make_void { using type = void;};
    /// little helper for the SFINAE idiom we'll use (not required in C++17)
    template<typename... Ts> using void_t = typename make_void<Ts...>::type;
}
#endif // __cplusplus

#endif // CPP14_COMPAT_H

// vim: sw=4:tw=78:ft=cpp:et
