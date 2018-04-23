/* @file SOADressedTuple.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-05-09
 */

#ifndef SOADRESSEDTUPLE_H
#define SOADRESSEDTUPLE_H

#include <tuple>
#include <type_traits>
#include "c++14_compat.h"

/// namespace to encapsulate SOA stuff
namespace SOA {
    /// implementation details
    namespace impl {
        /** @brief base for SOA::DressedTuple to hold common functionality
         *
         * SOA::View's and SOA::Container's requirements on Dressed tuples
         * are somewhat more stringent than those of the various STL
         * implementations out there in that we want to automatically
         * promote "naked" tuples to DressedTuples if they're compatible,
         * and allow reasonable assignments as well. This will need a
         * bunch of constructors and assignment operators that do the right
         * thing, but will only be enabled (via enable_if) if the tuple
         * elements have the right type. Unfortunately, the STL's version of
         * std::is_assignable and std::is_constructible don't see through the
         * fact that every DressedTuple is also a naked tuple, so we have our
         * own versions here that use the STL's implementation, but see
         * through the various levels of inheritance at the naked tuple
         * underneath. The aim is to only enable our versions of constructors
         * and assignment operators if the underlying tuple's version won't
         * do.
         *
         * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
         * @date 2017-11-15
         */
        class DressedTupleBase {
            protected:
            /// helper to check constructibility of tuples - default: no
            constexpr static inline std::false_type
            _is_constructible(...) noexcept
            { return std::false_type(); }
            /// helper to check constructibility of tuples - depends on tuples
            template <typename... Ts1, typename... Ts2>
            constexpr static inline typename std::enable_if<
                sizeof...(Ts1) == sizeof...(Ts2), std::is_constructible<
                std::tuple<Ts1...>, std::tuple<Ts2...> > >::type
            _is_constructible(const std::tuple<Ts1...>&,
                    const std::tuple<Ts2...>&) noexcept
            {
                return std::is_constructible<
                    std::tuple<Ts1...>, std::tuple<Ts2...> >();
            }
            /// check constructibility of T1 from T2
            template <typename T1, typename T2>
            using is_constructible = typename std::conditional<
                    std::is_constructible<T1, T2>::value, std::false_type,
                    decltype(_is_constructible(std::declval<const T1&>(),
                                std::declval<const T2&>()))>::type;

            /// helper to check assignability of tuples - default: no
            constexpr static inline std::false_type
            _is_assignable(...) noexcept
            { return std::false_type(); }
            /// helper to check tuple's assignability - depends on tuples
            template <typename... Ts1, typename... Ts2>
            constexpr static inline typename std::enable_if<
                sizeof...(Ts1) == sizeof...(Ts2), std::is_assignable<
                std::tuple<Ts1...>, std::tuple<Ts2...> > >::type _is_assignable(
                    const std::tuple<Ts1...>&, const std::tuple<Ts2...>&) noexcept
            {
                return std::is_assignable<
                    std::tuple<Ts1...>, std::tuple<Ts2...> >();
            }
            /// figure out if tuple underlying T1 is assignable from T2
            template <typename T1, typename T2>
            using is_assignable = typename std::conditional<
                    std::is_assignable<T1, T2>::value, std::false_type,
                    decltype(_is_assignable(std::declval<const T1&>(),
                        std::declval<const T2&>()))>::type;
            /// helper to figure out the underlying tuple's size
            template <typename... Ts>
            constexpr static inline std::tuple_size<std::tuple<Ts...> >
            _tuple_size(const std::tuple<Ts...>&) noexcept
            { return std::tuple_size<std::tuple<Ts...> >(); }
            /// figure out the underlying tuple's size
            template <typename T>
            using tuple_size = decltype(
                    _tuple_size(std::declval<const T&>()));
        };
    }
    /** @brief dress std::tuple with the get interface of SOAObjectProxy
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-05-09
     *
     * @tparam TUPLE        an instantiation of std::tuple
     * @tparam CONTAINER    underlying SOAContainer
     */
    template <typename TUPLE, typename CONTAINER>
    class DressedTuple : public impl::DressedTupleBase, public TUPLE
    {
        public:
            /// convenience typedef
            using self_type = DressedTuple<TUPLE, CONTAINER>;
            /// use TUPLE's constructors where possible
            using TUPLE::TUPLE;
            /// use TUPLE's assignment operators where possible
            using TUPLE::operator=;

            /// (copy) assignment from a naked proxy
            DressedTuple& operator=(
                    const typename CONTAINER::naked_proxy& other) noexcept(
                        noexcept(std::declval<TUPLE>().operator=(typename
                                SOA::Typelist::to_tuple<typename
                                CONTAINER::fields_typelist>::value_tuple(other))))
            {
                TUPLE::operator=(typename SOA::Typelist::to_tuple<typename
                        CONTAINER::fields_typelist>::value_tuple(other));
                return *this;
            }

        private:
            /// helper for fallback constructors
            template <std::size_t... IDXS, typename T>
            explicit constexpr DressedTuple(std::index_sequence<IDXS...>,
                    T&& t) noexcept(noexcept(
                            TUPLE(std::get<IDXS>(std::forward<T>(t))...))) :
                TUPLE(std::get<IDXS>(std::forward<T>(t))...)
            {}
            /// helper for fallback assignment operators
            template <std::size_t... IDXS, typename T>
            void assign(std::index_sequence<IDXS...>, T&& t) noexcept(noexcept(
                        std::forward_as_tuple(((std::get<IDXS>(
                                        std::declval<self_type&>()) =
                                    std::get<IDXS>(
                                        std::forward<T>(t))), 0)...)))
            {
                std::forward_as_tuple(((std::get<IDXS>(*this) =
                                std::get<IDXS>(std::forward<T>(t))), 0)...);
            }

        public:
#if (defined(__GNUC__) && (7 > __GNUC__ || (7 == __GNUC__ && \
                __GNUC_MINOR__ <= 1)) && !defined(__clang__))
            // old versions of gcc don't need extra code, only gcc 7.2 or
            // newer does
#elif ((defined(__GNUC__) && !defined(__clang__)) || \
        (defined(__clang__) && 4 <= __clang_major__))
            /// fallback constructor to see past dressed tuples
            template <typename... Ts, typename std::enable_if<
                std::is_constructible<TUPLE, Ts...>::value, int>::type = 0>
            constexpr DressedTuple(Ts&&... ts) noexcept(noexcept(
                        TUPLE(std::declval<Ts>()...))) :
                DressedTuple(std::make_index_sequence<sizeof...(Ts)>(),
                        std::forward_as_tuple(std::forward<Ts>(ts)...))
            {}
#elif (defined(__clang__) && 3 == __clang_major__ && 9 == __clang_minor__)
            /// fallback constructor to see past dressed tuples
            template <typename... Ts, typename DUMMY = typename std::enable_if<
                is_constructible<TUPLE, std::tuple<Ts...> >::value, int>::type>
            constexpr DressedTuple(Ts&&... ts, DUMMY = 0) noexcept(noexcept(
                        TUPLE(std::declval<Ts>()...))) :
                DressedTuple(std::make_index_sequence<sizeof...(Ts)>(),
                        std::forward_as_tuple(std::forward<Ts>(ts)...))
            {}
#elif (defined(__clang__) && 3 == __clang_major__ && 8 == __clang_minor__)
            // no extra code needed for clang 3.8
#endif
            /// fallback constructor to see past dressed tuples
            template <typename T, typename std::enable_if<is_constructible<
                    TUPLE, T>::value, int>::type = 0>
            constexpr DressedTuple(T&& t) noexcept(noexcept(
                DressedTuple(std::make_index_sequence<tuple_size<T>::value>(),
                    std::forward<T>(t)))) :
                DressedTuple(std::make_index_sequence<tuple_size<T>::value>(),
                        std::forward<T>(t))
            {}
            /// fallback assignment operator to see past dressed tuples
            template <typename T>
            typename std::enable_if<is_assignable<TUPLE, T>::value,
                     self_type>::type&
            operator=(T&& t) noexcept(noexcept(std::declval<
                        DressedTuple<TUPLE, CONTAINER> >().assign(
                            std::make_index_sequence<tuple_size<T>::value>(),
                            std::forward<T>(t))))
            {
                if (static_cast<void*>(this) != &t)
                    assign(std::make_index_sequence<tuple_size<T>::value>(),
                            std::forward<T>(t));
                return *this;
            }

            /// provide the member function template get interface of proxies
            template<typename CONTAINER::size_type MEMBERNO>
            auto get() noexcept -> decltype(std::get<MEMBERNO>(std::declval<self_type&>()))
            { return std::get<MEMBERNO>(*this); }

            /// provide the member function template get interface of proxies
            template<typename CONTAINER::size_type MEMBERNO>
            auto get() const noexcept -> decltype(std::get<MEMBERNO>(
                        std::declval<const self_type&>()))
            { return std::get<MEMBERNO>(*this); }

            /// provide the member function template get interface of proxies
            template<typename MEMBER, typename CONTAINER::size_type MEMBERNO =
                CONTAINER::template memberno<MEMBER>()>
            auto get() noexcept -> decltype(std::get<MEMBERNO>(std::declval<self_type&>()))
            {
                static_assert(CONTAINER::template memberno<MEMBER>() ==
                        MEMBERNO, "Called with wrong template argument(s).");
                return std::get<MEMBERNO>(*this);
            }

            /// provide the member function template get interface of proxies
            template<typename MEMBER, typename CONTAINER::size_type MEMBERNO =
                CONTAINER::template memberno<MEMBER>()>
            auto get() const noexcept -> decltype(std::get<MEMBERNO>(
                        std::declval<const self_type&>()))
            {
                static_assert(CONTAINER::template memberno<MEMBER>() ==
                        MEMBERNO, "Called with wrong template argument(s).");
                return std::get<MEMBERNO>(*this);
            }
    };
} // namespace SOA

#endif // SOADRESSEDTUPLE_H

// vim: sw=4:tw=78:ft=cpp:et
