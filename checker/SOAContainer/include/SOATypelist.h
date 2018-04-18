/** @file SOATypelist.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-04-10
 * - initial related
 *
 * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
 * @date 2016-11-17
 * - rewrite to use better coding, more C++11 constructs
 */

#ifndef SOATYPELIST_H
#define SOATYPELIST_H

#include <cstdint>
#include <type_traits>

/// namespace to encapsulate SOA stuff
namespace SOA {
    /// namespace for typelist type used by Container and related utilities
    namespace Typelist {
        // forward declarations
        namespace typelist_impl { struct empty_typelist; }
        template <typename HEAD = typelist_impl::empty_typelist, typename... ARGS>
        struct typelist;
    
        namespace typelist_impl {
        /// the empty typelist
        struct empty_typelist {
            /// typelist empty?
            constexpr static bool empty() noexcept { return true; }
            /// size of typelist
            constexpr static std::size_t size() noexcept { return 0; }
            /// no types at any indices!
            template <std::size_t IDX> struct at {};
            /// find a type, return its index, or -1 if not found
            template <typename T, std::size_t = 0>
            constexpr static std::size_t find() noexcept
            { return -1; }
            template <typename T>
            constexpr static std::size_t count() noexcept { return 0; }
        };
    
        /// little switch for typelist
        template <std::size_t LEN, typename... ARGS> struct __typelist {
            using type = typelist<ARGS...>;
        };
        /// little switch for typelist (specialisation)
        template <typename... ARGS> struct __typelist<0, ARGS...> {
            using type = empty_typelist;
        };
        } // namespace typelist_impl
    
        template <typename HEAD, typename... ARGS>
        struct typelist {
            /// first element
            using head_type = HEAD;
            /// typelist empty?
            constexpr static bool empty() noexcept
            { return std::is_same<HEAD, typelist_impl::empty_typelist>::value; }
            /// size of typelist
            constexpr static std::size_t size() noexcept
            { return (!empty()) + sizeof...(ARGS); }
            /// tail typelist
            using tail_types = typename typelist_impl::__typelist<size(), ARGS...>::type;
            /// return type at index IDX
            template <std::size_t IDX, bool PASTEND =
                (IDX >= typelist<HEAD, ARGS...>::size()), int DUMMY = 0>
            struct at;
            /// specialisation: short-circuit reads past end of list early
            template <std::size_t IDX, int DUMMY> struct at<IDX, true, DUMMY> {};
            /// specialisation: specialisation for reading types at valid indices
            template <std::size_t IDX, int DUMMY> struct at<IDX, false, DUMMY>
            { using type = typename tail_types::template at<IDX - 1>::type; };
            /// specialisation: specialisation for reading types at valid indices
            template <int DUMMY> struct at<0, false, DUMMY>
            { using type = head_type; };
            /// find a type, return its index, or -1 if not found
            template <typename T, std::size_t OFS = 0>
            constexpr static std::size_t find() noexcept
            {
                return std::is_same<T, head_type>::value ? OFS :
                    tail_types::template find<T, OFS + 1>();
            }
            /// count how often T appears in the type list
            template <typename T>
            constexpr static std::size_t count() noexcept
            {
                return std::is_same<T, HEAD>::value +
                    tail_types::template count<T>();
            }
            /// little helper to map over the types in the typelist
            template <template <typename ARG> class OP>
            using map_t = typelist<OP<HEAD>, OP<ARGS>...>;
            // make sure we construct only valid typelists
            static_assert(!empty() || (empty() && 0 == size()),
                    "typelist: head empty_typelist with non-empty tail not allowed!");
        };
        template<> struct typelist<typelist_impl::empty_typelist, typelist_impl::empty_typelist> : typelist_impl::empty_typelist {};
    
        /// check basic properties to validate implementation
        namespace __impl_compile_time_tests {
            template <typename T> using add_const_t = const T;
            static_assert(typelist<>::empty(), "implementation error");
            static_assert(typelist<>::size() == 0, "implementation error");
            static_assert(typelist<>::find<float>() == std::size_t(-1),
                    "implementation error");
            static_assert(!typelist<int>::empty(), "implementation error");
            static_assert(typelist<int>::size() == 1, "implementation error");
            static_assert(std::is_same<typelist<int>::at<0>::type, int>::value,
                    "implementation error");
#if !defined(__clang__) && !defined(INTEL_COMPILER) && (\
        __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8))
            static_assert(std::is_same<typelist<const int>,
                    typelist<int>::map_t<add_const_t> >::value,
                    "implementation error");
#endif
            static_assert(typelist<int>::find<int>() == 0, "implementation error");
            static_assert(typelist<int>::find<float>() == std::size_t(-1),
                    "implementation error");
            static_assert(!typelist<int, bool>::empty(), "implementation error");
            static_assert(typelist<int, bool>::size() == 2, "implementation error");
            static_assert(std::is_same<typelist<int, bool>::at<0>::type, int>::value,
                    "implementation error");
            static_assert(std::is_same<typelist<int, bool>::at<1>::type, bool>::value,
                    "implementation error");
            static_assert(typelist<int, bool>::find<int>() == 0, "implementation error");
            static_assert(typelist<int, bool>::find<bool>() == 1, "implementation error");
#if !defined(__clang__) && !defined(INTEL_COMPILER) && (\
        __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8))
            static_assert(std::is_same<typelist<const int, const bool>,
                    typelist<int, bool>::map_t<add_const_t> >::value,
                    "implementation error");
#endif
            static_assert(0 == typelist<int, int, bool>::template count<char>(),
                    "implementation error");
            static_assert(1 == typelist<int, int, bool>::template count<bool>(),
                    "implementation error");
            static_assert(2 == typelist<int, int, bool>::template count<int>(),
                    "implementation error");
        }
    } // namespace Typelist
} // namespace SOA

#endif // SOATYPELIST_H

// vim: sw=4:tw=78:ft=cpp:et
