/** @file SOAUtils.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-04-09
 */

#ifndef SOAUTILS_H
#define SOAUTILS_H

#include <tuple>
#include <utility>
#include <cstdint>
#include <type_traits>

#include "c++14_compat.h"

/// namespace to encapsulate SOA stuff
namespace SOA {
    /// various other utilities used by Container
    namespace Utils {
        /// logic "and" of variadic arguments
        constexpr bool ALL() noexcept { return true; }
        template <class HEAD, class... TAIL>
        constexpr bool ALL(HEAD head, TAIL... tail) noexcept
        { return head && ALL(tail...); }
        /// logic "or" of variadic arguments
        constexpr bool ANY() noexcept { return false; }
        template <class HEAD, class... TAIL>
        constexpr bool ANY(HEAD head, TAIL... tail) noexcept
        { return head || ANY(tail...); }

        /// ignores all its arguments
        template <typename... ARGS>
        void ignore(ARGS&&...) noexcept {}

        /// little helper to call callable f with contents of t as arguments
        template <typename F, typename... ARGS, typename RETVAL = decltype(
                std::forward<F>(std::declval<F>())(std::forward<ARGS>(
                        std::declval<ARGS>())...))>
        constexpr typename std::enable_if<!std::is_same<void, RETVAL>::value,
                  RETVAL>::type
        invoke(F&& f, ARGS&&... args) noexcept(
                noexcept(std::forward<F>(f)(std::forward<ARGS>(args)...)))
        { return std::forward<F>(f)(std::forward<ARGS>(args)...); }
        /// little helper to call callable f with contents of t as arguments
        template <typename F, typename... ARGS, typename RETVAL = decltype(
                std::forward<F>(std::declval<F>())(std::forward<ARGS>(
                        std::declval<ARGS>())...))>
        typename std::enable_if<std::is_same<void, RETVAL>::value,
                 RETVAL>::type
        invoke(F&& f, ARGS&&... args) noexcept(
                noexcept(std::forward<F>(f)(std::forward<ARGS>(args)...)))
        { std::forward<F>(f)(std::forward<ARGS>(args)...); }

        // implementation details
        namespace impl {
            template <typename F, typename T, std::size_t... IDXS,
                     typename RETVAL = decltype(SOA::Utils::invoke(std::forward<F>(
                                     std::declval<F>()),
                                 std::get<IDXS>(std::forward<T>(
                                         std::declval<T>()))...))>
            constexpr typename std::enable_if<
                    !std::is_same<void, RETVAL>::value, RETVAL>::type
            apply_impl(F&& f, T&& t, std::index_sequence<IDXS...>) noexcept(
                    noexcept(SOA::Utils::invoke(std::forward<F>(f), std::get<IDXS>(
                                std::forward<T>(t))...)))
            {
                return SOA::Utils::invoke(std::forward<F>(f),
                        std::get<IDXS>(std::forward<T>(t))...);
            }
            template <typename F, typename T, std::size_t... IDXS,
                     typename RETVAL = decltype(SOA::Utils::invoke(std::forward<F>(
                                     std::declval<F>()),
                                 std::get<IDXS>(std::forward<T>(
                                         std::declval<T>()))...))>
            typename std::enable_if<std::is_same<void, RETVAL>::value,
            RETVAL>::type
            apply_impl(F&& f, T&& t, std::index_sequence<IDXS...>) noexcept(noexcept(
                        SOA::Utils::invoke(std::forward<F>(f), std::get<IDXS>(std::forward<T>(t))...)))
            {
                SOA::Utils::invoke(std::forward<F>(f),
                        std::get<IDXS>(std::forward<T>(t))...);
            }
        } // namespace impl

        /// little helper to call callable f with contents of t as arguments
        template<typename F, typename T, typename RETVAL = decltype(
                impl::apply_impl(std::forward<F>(std::declval<F>()),
                    std::forward<T>(std::declval<T>()),
                    std::make_index_sequence<std::tuple_size<typename
                    std::decay<T>::type>::value>())), std::size_t N =
            std::tuple_size<typename std::decay<T>::type>::value>
        constexpr typename std::enable_if<!std::is_same<void, RETVAL>::value, RETVAL>::type
        apply(F&& f, T&& t) noexcept(noexcept(
                    impl::apply_impl(std::forward<F>(f), std::forward<T>(t),
                        std::make_index_sequence<N>())))
        {
            return impl::apply_impl(std::forward<F>(f), std::forward<T>(t),
                        std::make_index_sequence<N>());
        }
        /// little helper to call callable f with contents of t as arguments
        template<typename F, typename T, typename RETVAL = decltype(
                impl::apply_impl(std::forward<F>(std::declval<F>()),
                    std::forward<T>(std::declval<T>()),
                    std::make_index_sequence<std::tuple_size<typename
                    std::decay<T>::type>::value>())), std::size_t N =
            std::tuple_size<typename std::decay<T>::type>::value>
        typename std::enable_if<std::is_same<void, RETVAL>::value, void>::type
        apply(F&& f, T&& t) noexcept(noexcept(
                    impl::apply_impl(std::forward<F>(f), std::forward<T>(t),
                        std::make_index_sequence<N>())))
        {
            impl::apply_impl(std::forward<F>(f), std::forward<T>(t),
                        std::make_index_sequence<N>());
        }

        namespace impl {
            /// helper for the implementation of map
            template <typename F>
            struct map_f {
                F m_f;
                template <typename... ARGS, typename RETVAL = decltype(
                        std::make_tuple(std::forward<F>(std::declval<F>())(
                                std::forward<ARGS>(std::declval<ARGS>()))...))>
                typename std::enable_if<!ANY(std::is_same<void,
                        decltype(std::forward<F>(std::declval<F>())(
                        std::forward<ARGS>(
                            std::declval<ARGS>())))>::value...), RETVAL>::type
                operator()(ARGS&&... args) const noexcept(noexcept(
                            std::make_tuple(m_f(std::forward<ARGS>(args))...)))
                { return std::make_tuple(m_f(std::forward<ARGS>(args))...); }

                template <typename... ARGS>
                typename std::enable_if<ALL(std::is_same<void,
                        decltype(std::forward<F>(std::declval<F>())(
                                std::forward<ARGS>(
                                    std::declval<ARGS>())))>::value...),
                         void>::type
                operator()(ARGS&&... args) const noexcept(noexcept(
                            ignore((m_f(std::forward<ARGS>(args)), 0)...)))
                { ignore((m_f(std::forward<ARGS>(args)), 0)...); }
            };

            /// helper for the implementation of map
            template <typename F>
            map_f<decltype(std::forward<F>(std::declval<F>()))> make_map_f(
                    F&& f) noexcept(noexcept(map_f<decltype(std::forward<F>(
                                    std::declval<F>()))>{std::forward<F>(f)}))
            { return {std::forward<F>(f)}; }
        }

        /// apply functor f to all elements of tuple, return tuple of results
        template <typename F, typename T, typename RETVAL =
            decltype(SOA::Utils::apply(
                    impl::make_map_f(std::forward<F>(std::declval<F>())),
                    std::forward<T>(std::declval<T>())))>
        typename std::enable_if<!std::is_same<RETVAL, void>::value, RETVAL>::type
        map(F&& f, T&& t) noexcept(noexcept(SOA::Utils::apply(impl::make_map_f(
                            std::forward<F>(f)), std::forward<T>(t))))
        {
            return SOA::Utils::apply(
                impl::make_map_f(std::forward<F>(f)), std::forward<T>(t));
        }

        /// apply functor f (returning void) to all elements of tuple
        template <typename F, typename T, typename RETVAL =
            decltype(SOA::Utils::apply(
                    impl::make_map_f(std::forward<F>(std::declval<F>())),
                    std::forward<T>(std::declval<T>())))>
        typename std::enable_if<std::is_same<RETVAL, void>::value, RETVAL>::type
        map(F&& f, T&& t) noexcept(noexcept(SOA::Utils::apply(impl::make_map_f(
                            std::forward<F>(f)), std::forward<T>(t))))
        {
            SOA::Utils::apply(
                    impl::make_map_f(std::forward<F>(f)), std::forward<T>(t));
        }

        /// implementation details of foldl
        namespace foldl_impl {
            /// implementation of foldl
            template <typename FUN, typename TUP, std::size_t LEN>
            struct foldl_impl {
                FUN fun;
                const TUP& tup;
                /// fold over tuple with more than 2 elements
                template <typename INI, std::size_t IDX, std::size_t... IDXS>
                typename std::enable_if<(LEN > 1),
                    typename std::result_of<
                        foldl_impl<FUN, TUP, LEN - 1>(
                            typename std::result_of<FUN(INI,
                                typename std::tuple_element<IDX, TUP>::type)
                            >::type, std::index_sequence<IDXS...>)
                        >::type
                    >::type
                operator()(INI ini,
                        std::index_sequence<IDX, IDXS...>) const noexcept(
                        noexcept(foldl_impl<FUN, TUP, LEN - 1>{fun, tup}(
                                fun(ini, std::get<IDX>(tup)),
                                std::index_sequence<IDXS...>())))
                {
                    return foldl_impl<FUN, TUP, LEN - 1>{fun, tup}(
                            fun(ini, std::get<IDX>(tup)),
                            std::index_sequence<IDXS...>());
                }
                /// foldl over tuple with exactly one element
                template <typename INI, std::size_t IDX>
                typename std::enable_if<1 == LEN,
                    typename std::result_of<
                        FUN(INI,
                                typename std::tuple_element<IDX, TUP>::type)>::type
                    >::type
                operator()(INI ini, std::index_sequence<IDX>) const noexcept(
                        noexcept(fun(ini, std::get<IDX>(tup))))
                { return fun(ini, std::get<IDX>(tup)); }
                /// foldl over tuple with no elements
                template <typename INI>
                INI operator()(INI ini, std::index_sequence<>) const noexcept(
                        noexcept(INI(ini)))
                { return ini; }
            };
        }

        /** @brief foldl (left fold) on tuples
         *
         * Given an index sequence i1, ..., iN, a tuple with elements (e_1, ...,
         * e_N-1, e_N), a fold function fun, and an initial value ini, "fold" the
         * tuple elements given in the index sequence using the functor fun
         * starting from the left, i.e. return fun(fun(fun(...fun(ini, e_i1),
         * ...), e_iN-1), e_iN), or return ini for the empty tuple.
         *
         * @tparam INI  type of initial value
         * @tparam FUN  type of functor used in the fold
         * @tparam TUP  type of the tuple to be folded
         *
         * @param fun   functor used to "fold" tuple
         * @param tup   tuple to fold
         * @param ini   initial value
         *
         * @returns     value of the left fold
         *
         * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
         * @date 2016-12-12
         */
        template <typename INI, typename FUN, typename TUP, std::size_t... IDXS>
        typename std::result_of<foldl_impl::foldl_impl<FUN, TUP,
                 sizeof...(IDXS)>(INI, std::index_sequence<IDXS...>)>::type
        foldl(FUN fun, const TUP& tup, INI ini, std::index_sequence<IDXS...>) noexcept(
                noexcept(foldl_impl::foldl_impl<FUN, TUP, sizeof...(IDXS)>{fun, tup}(
                        ini, std::index_sequence<IDXS...>())))
        {
            return foldl_impl::foldl_impl<FUN, TUP, sizeof...(IDXS)>{fun, tup}(
                    ini, std::index_sequence<IDXS...>());
        }

        /** @brief foldl (left fold) on tuples
         *
         * Given a tuple with elements (e_1, ..., e_N-1, e_N), a fold function
         * fun, and an initial value ini, "fold" the tuple using the functor fun
         * starting from the left, i.e. return fun(fun(fun(...fun(ini, e_1), ...),
         * e_N-1), e_N), or return ini for the empty tuple.
         *
         * @tparam INI  type of initial value
         * @tparam FUN  type of functor used in the fold
         * @tparam TUP  type of the tuple to be folded
         *
         * @param fun   functor used to "fold" tuple
         * @param tup   tuple to fold
         * @param ini   initial value
         *
         * @returns     value of the left fold
         *
         * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
         * @date 2016-12-12
         */
        template <typename INI, typename FUN, typename TUP>
        auto foldl(FUN fun, const TUP& tup, INI ini = INI()) noexcept(
                noexcept(foldl(fun, tup, ini,
                        std::make_index_sequence<std::tuple_size<TUP>::value>()))) ->
            decltype(foldl(fun, tup, ini,
                        std::make_index_sequence<std::tuple_size<TUP>::value>()))
        {
            return foldl(fun, tup, ini,
                    std::make_index_sequence<std::tuple_size<TUP>::value>());
        }

        /// is T a SOA::View?
        template <typename T, typename = void>
        struct is_view : std::false_type {};
        /// is T a SOA::View?
        template <typename T>
        struct is_view<T, std::void_t<typename T::view_tag> > : std::true_type {};

        namespace test {
            struct no_view {};
            struct a_view { using view_tag = struct {}; };
            static_assert(!is_view<no_view>::value &&
                    is_view<a_view>::value,
                    "Implementation bug in is_view");
        }

        /// implementation details for zip
        namespace zip_impl {
            /// little helper for zip
            template <std::size_t IDXINNER, std::size_t... IDXOUTER, typename... OBJS>
            constexpr auto zip(std::index_sequence<IDXOUTER...>, OBJS&&... objs) noexcept(noexcept(
                        std::forward_as_tuple(std::get<IDXINNER>(std::get<IDXOUTER>(
                                    std::forward_as_tuple(std::forward<OBJS>(objs)...)))...))) -> decltype(
                    std::forward_as_tuple(std::get<IDXINNER>(std::get<IDXOUTER>(
                                std::forward_as_tuple(std::forward<OBJS>(objs)...)))...))
            {
                return std::forward_as_tuple(std::get<IDXINNER>(std::get<IDXOUTER>(
                                std::forward_as_tuple(std::forward<OBJS>(objs)...)))...);
            }

            /// little helper for zip
            template <typename... OBJS, std::size_t... IDXOUTER, std::size_t... IDXINNER>
            constexpr auto zip(std::index_sequence<IDXOUTER...> outer,
                    std::index_sequence<IDXINNER...>, OBJS&&... objs) noexcept(noexcept(
                            std::make_tuple(zip<IDXINNER>(
                                    outer, std::forward<OBJS>(objs)...)...))) -> decltype(
                        std::make_tuple(zip<IDXINNER>(
                                outer, std::forward<OBJS>(objs)...)...))
            {
                return std::make_tuple(zip<IDXINNER>(
                            outer, std::forward<OBJS>(objs)...)...);
            }
        }

        /** @brief zip together tuples
         *
         * @tparam OBJS... parameter pack of types of tuples to zip
         *
         * @param objs       tuples to zip
         * @returns          a tuple of zipped tuples
         *
         * Given N tuples with elements (a0, a1, ..., a_M), (b0, b1, ... b_M), ...,
         * returns a tuple ((a0, b0, ...), (a1, b1, ...), ...), i.e. zip "zips" up
         * the contents of the tuples passed as arguments.
         *
         * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
         * @date 2017-01-11
         */
        template <typename... OBJS>
        constexpr auto zip(OBJS&&... objs) noexcept(noexcept(
                    zip_impl::zip(std::make_index_sequence<sizeof...(OBJS)>(),
                        std::make_index_sequence<std::tuple_size<
                        typename std::tuple_element<0, std::tuple<
                        typename std::decay<OBJS>::type...> >::type>::value>(),
                        std::forward<OBJS>(objs)...))) ->
            typename std::enable_if<ALL(!is_view<typename std::remove_cv<
                    typename std::remove_reference<OBJS>::type
                    >::type>::value...), decltype(
                zip_impl::zip(std::make_index_sequence<sizeof...(OBJS)>(),
                    std::make_index_sequence<std::tuple_size<
                    typename std::tuple_element<0, std::tuple<
                    typename std::decay<OBJS>::type...> >::type>::value>(),
                    std::forward<OBJS>(objs)...))>::type
        {
            return zip_impl::zip(std::make_index_sequence<sizeof...(OBJS)>(),
                    std::make_index_sequence<std::tuple_size<
                    typename std::tuple_element<0, std::tuple<
                    typename std::decay<OBJS>::type...> >::type>::value>(),
                    std::forward<OBJS>(objs)...);
        }

        namespace impl {
            /// implementation detail for apply_zip below
            template <typename F>
            struct apply_zip_impl {
                F f;
                template <typename... Ts>
                void operator()(Ts&&... ts) const noexcept(noexcept(
                            ignore((SOA::Utils::apply(
                                        f, std::forward<Ts>(ts)), 0)...)))
                { ignore((SOA::Utils::apply(f, std::forward<Ts>(ts)), 0)...); }
            };
        }

        /// little helper for push_back etc.
        template <typename F, typename... Ts>
        void apply_zip(F&& f, Ts&&... ts) noexcept(noexcept(
                    SOA::Utils::apply(impl::apply_zip_impl<
                        decltype(std::forward<F>(f))>{
                        std::forward<F>(f)}, zip(std::forward<Ts>(ts)...))))
        {
            SOA::Utils::apply(impl::apply_zip_impl<
                    decltype(std::forward<F>(f))>{
                    std::forward<F>(f)}, zip(std::forward<Ts>(ts)...));
        }

    } // namespace Utils
} // namespace SOA

#endif // SOAUTILS_H

// vim: sw=4:tw=78:ft=cpp:et
