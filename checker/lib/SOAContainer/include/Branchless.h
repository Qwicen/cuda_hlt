/** @file Branchless.h
 *
 * help with branchless programming by providing a branchless substitute
 * of the ternary operator, "cond ? vTrue : vFalse"
 *
 * @date 2016-02-22
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 */
#ifndef BRANCHLESS_H
#define BRANCHLESS_H

#include <utility>
#include <type_traits>

// play a few tricks so this performs well at low optimisation levels
#undef alwaysinline
#ifdef __GNUC__
#define alwaysinline __attribute__((always_inline))
#else
#define alwaysinline inline
#endif

/// implementation details for sel operator (see below)
namespace SelImplementation {
    /// operate on 64 bits at a time by default
    template <unsigned SZ> struct IntType { using type = uint64_t; };
    /// specialisations for smaller objects
    template <> struct IntType<1> { using type = uint8_t; };
    /// specialisations for smaller objects
    template <> struct IntType<2> { using type = uint16_t; };
    /// specialisations for smaller objects
    template <> struct IntType<3> { using type = uint32_t; };
    /// specialisations for smaller objects
    template <> struct IntType<4> { using type = uint32_t; };

    /// force inlining and unrolling at low optimisation levels
    template <unsigned N>
    struct doBlend {
        template <typename BB1, typename BB2>
        static alwaysinline BB1& doit(
            BB1& bb1, const BB2& bb2, bool c) noexcept
        {
            // tried all sorts of things here, but this seems to be one of
            // the best options, producing reasonable code on all compilers,
            // and vectorising well when used in loops
            using I = typename BB1::int_type;
            bb1.m_i[N - 1] =
                (bb1.m_i[N - 1] & -I(c)) | (bb2.m_i[N - 1] & ~(-I(c)));
            return doBlend<N - 1>::doit(bb1, bb2, c);
        }
    };

    /// force inlining and unrolling at low optimisation levels (recursion anchor)
    template <>
    struct doBlend<0u> {
        template <typename BB1, typename BB2>
        static alwaysinline BB1& doit(
            BB1& bb1, const BB2&, bool) noexcept
        { return bb1; }
    };

    /// allow bitwise operations on representation of objects of type T
    template <typename T, typename I = typename IntType<sizeof(T)>::type>
    union bitblender {
            // this class does open brain surgery on your data: ensure all goes
            // well by checking T is simple enough with a static_assert
            //
            // FIXME: compilers have not caught up enough with C++ 11 to support
            // std::is_trivially_copyable yet - to be changed soon
            //static_assert(std::is_trivially_copyable<T>::value,
            static_assert(std::is_trivial<T>::value, // FIXME: for now, use this...
                    "In function T sel(bool cond, T valIfTrue, T valIfFalse): "
                    "Type T must be trivially copyable, or else!");
            /// how many integers of type I do we need to cover a T
            enum { size = sizeof(T) / sizeof(I) + bool(sizeof(T) % sizeof(I)) };
        using int_type = I;
            /// object of type T...
            T m_t;
            /// ... overlaid with a bunch of integers
            I m_i[size];
        /// build from a T
        constexpr alwaysinline bitblender(const T& t) noexcept : m_t(t)
        { }

            /// blend together two objects; c selects where to take bits from
            template <typename U>
            alwaysinline typename std::enable_if<size == U::size,
        bitblender<T, I> >::type& blend(const U& o, bool c) noexcept
            { return doBlend<size>::doit(*this, o, c); }
    };

    /// little helper for the SFINAE idiom we'll use (not required in C++17)
    template<typename... Ts> struct make_void { using type = void;};
    /// little helper for the SFINAE idiom we'll use (not required in C++17)
    template<typename... Ts> using void_t = typename make_void<Ts...>::type;

    /// type property: can not call sel on type
    template<typename T, typename = void>
    struct has_sel : std::false_type {};
    /// type property: can call sel on type
    template <typename T>
    struct has_sel<T, void_t<decltype((*static_cast<typename
        std::decay<T>::type *>(nullptr)).sel(true, *static_cast<typename
        std::decay<T>::type *>(nullptr)))> > : std::true_type {};
}

/** @brief branchless substitute for ternary operator ("cond ? vtrue : vFalse")
 *
 * @tparam T     type to work with, must provide a sel method
 *
 * @param cond   condition; true selects the second, false the third argument
 * @param vTrue  value to return when cond is true
 * @param vFalse value to return when cond is false
 *
 * @returns vTrue if cond is true, vFalse otherwise
 *
 * @note: In contrast to the ternary operator, sel DOES NOT DO SHORT-CIRCUIT
 * EVALUATION, so both vTrue and vFalse are fully evaluated before the
 * selection is made. Code such as "iptr ? *iptr : 0" where iptr is a pointer
 * to an integer will segfault with sel, since the sel call needs to evaluate
 * the arguments before the call can be made.
 *
 * This version of the sel routine works when the type T provides a sel
 * method which takes care of selecting members according to the condition
 * cond. A simple example is given below:
 *
 * @code
 * struct polarpoint {
 *     float m_r, m_phi;
 *     // non-trivial constructor
 *     // have to implement sel to provide branchless selection
 *     polarpoint(float x, float y) :
 *         m_r(std::sqrt(x * x + y * y)), m_phi(std::atan2(y, x))
 *     { }
 *     // assign other to this if cond is false
 *     polarpoint& sel(bool cond, const polarpoint* elseVal)
 *     {
 *         sel(cond, m_r, elseVal.m_r); // branchless selection of members!
 *         sel(cond, m_phi, elseVal.m_phi);
 *         return *this;
 *     }
 * };
 *
 * // create two points p and q with different radii
 * polarpoint p(1, 1), q(2, 1);
 *
 * // select r to be the point with the larger radius
 * polarpoint r = sel(p.m_r > q.m_r, p, q);
 * @endcode
 *
 * This closely models the swap method many containers implement when
 * convenient to allow cheap swapping. In the same sense, if a type provides
 * a sel method, it indicates its ability to do branchless selection.
 *
 * If your type only has trivial constructors/destructors, copy- and move-
 * constructors and -assignment operators, and the same is true for all
 * members of the type, you don't have to provide a sel method, and the
 * overload just below will succeed.
 *
 * @date 2016-02-22
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 */
template <typename T>
alwaysinline typename std::enable_if<SelImplementation::has_sel<T>::value, T>::type
sel(bool cond, T vTrue, const T& vFalse) noexcept
{
    // defer to a type's sel method, if available
    return vTrue.sel(cond, std::forward<T>(vFalse));
}

/** @brief branchless substitute for ternary operator ("cond ? vtrue : vFalse")
 *
 * @tparam T     type to work with, must satisfy std::is_trivially_copyable
 *
 * @param cond   condition; true selects the second, false the third argument
 * @param vTrue  value to return when cond is true
 * @param vFalse value to return when cond is false
 *
 * @returns vTrue if cond is true, vFalse otherwise
 *
 * @note: In contrast to the ternary operator, sel DOES NOT DO SHORT-CIRCUIT
 * EVALUATION, so both vTrue and vFalse are fully evaluated before the
 * selection is made. Code such as "iptr ? *iptr : 0" where iptr is a pointer
 * to an integer will segfault with sel, since the sel call needs to evaluate
 * the arguments before the call can be made.
 *
 * This version of the routine works for trivial types, i.e. types which use
 * the compiler-generated versions of constructors, destructors, copy-/move-
 * constructors and -assignment operators; the same must be true for all of
 * the members of T. The reason for this restriction is that the code treats
 * an object in memory as a bit pattern to be constructed/destroyed anywhere
 * in memory at will, which only works if the objects are simple enough. If
 * you know your type survives this kind of treatment, but is non-trivial in
 * the sense given above, you may provide a sel method calling sel on all
 * members (much like the swap method for containers), which will get used by
 * the overload above.
 *
 * If the type T provides a sel method, it will be preferred over this
 * implementation.
 *
 * @date 2016-02-22
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 */
template <typename T>
inline typename std::enable_if<!SelImplementation::has_sel<T>::value &&
    std::is_trivial<T>::value, T>::type
sel(bool cond, T vTrue, const T& vFalse) noexcept
{
    // do surgery on raw bits in memory, and use that to "blend" the result
    using SelImplementation::bitblender;
    // return the "blended" result...
    return bitblender<T>(vTrue).blend(bitblender<T>(vFalse), cond).m_t;
}

/** @brief branchful substitute for ternary operator ("cond ? vtrue : vFalse")
 *
 * @tparam T     type to work with
 *
 * @param cond   condition; true selects the second, false the third argument
 * @param vTrue  value to return when cond is true
 * @param vFalse value to return when cond is false
 *
 * @returns vTrue if cond is true, vFalse otherwise
 *
 * @note: In contrast to the ternary operator, sel DOES NOT DO SHORT-CIRCUIT
 * EVALUATION, so both vTrue and vFalse are fully evaluated before the
 * selection is made. Code such as "iptr ? *iptr : 0" where iptr is a pointer
 * to an integer will segfault with sel, since the sel call needs to evaluate
 * the arguments before the call can be made.
 *
 * This overload is for the unfortunate few cases that do not provide a sel
 * method or are trivial (see the previous two overloads), and falls back on
 * the ternary operator in its implementation. Unless you're very lucky, or
 * your compiler is very clever, you will not really get branchless code from
 * this overload. It is provided so generic code doesn't fail in non-obvious
 * ways when changing types.
 *
 * @date 2016-02-22
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 */
template <typename T>
alwaysinline typename std::enable_if<!SelImplementation::has_sel<T>::value &&
    !std::is_trivial<T>::value, T>::type
sel(bool cond, T vTrue, const T& vFalse) noexcept
{ return cond ? vTrue : vFalse; }

#undef alwaysinline
#endif // BRANCHLESS_H

// vim: sw=4:tw=78:ft=cpp
