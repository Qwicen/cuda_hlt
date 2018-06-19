/** @file SOAView.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-04-10
 */

#ifndef SOAVIEW_H
#define SOAVIEW_H

#include <limits>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <type_traits>

#include "boost/range/iterator_range.hpp"

#include "SOATypelist.h"
#include "SOATypelistUtils.h"
#include "SOAObjectProxy.h"
#include "SOADressedTuple.h"
#include "SOAIterator.h"
#include "SOAUtils.h"

#include "SOAField.h"
#include "SOASkin.h"

// forward decl.
template < template <typename...> class CONTAINER,
         template <typename> class SKIN, typename... FIELDS>
class _Container;

/// namespace to encapsulate SOA stuff
namespace SOA {
    /** @brief skin class for View which does nothing and preserved the
     * raw proxy interface.
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-04-21
     */
    template <typename NAKEDPROXY>
    struct NullSkin : NAKEDPROXY
    {
        /// constructor(s) - forward to underlying proxy
        using NAKEDPROXY::NAKEDPROXY;
        /// assignment operator - forward to underlying proxy
        using NAKEDPROXY::operator=;
    };


    /// more _View implementation details
    namespace impl {
        /// decay T&& into T, leave (const) T(&) unchanged
        template <typename T>
        struct remove_rvalue_reference { using type = T; };
        /// decay T&& into T, leave (const) T(&) unchanged (specialisation)
        template <typename T>
        struct remove_rvalue_reference<const T&> { using type = const T&; };
        /// decay T&& into T, leave (const) T(&) unchanged (specialisation)
        template <typename T>
        struct remove_rvalue_reference<T&> { using type = T&; };
        /// decay T&& into T, leave (const) T(&) unchanged (specialisation)
        template <typename T>
        struct remove_rvalue_reference<T&&> { using type = T; };

        /// move everything but lvalue references
        template <typename T, typename R>
        const R& move_if_not_lvalue_reference(const T&, const R& r) { return r; }
        /// move everything but lvalue references
        template <typename T, typename R>
        R& move_if_not_lvalue_reference(T&, R& r) { return r; }
        /// move everything but lvalue references
        template <typename T, typename R>
        R&& move_if_not_lvalue_reference(T&&, R&& r) { return std::move(r); }

        /// helper to allow flexibility in how fields are supplied
        template <class STORAGE, template <typename> class SKIN,
                 class... TYPELISTORFIELDS>
        struct ViewFieldsFromTypelistOrTemplateParameterPackHelper {
            using type = _View<STORAGE, SKIN, TYPELISTORFIELDS...>;
        };
        /// helper to allow flexibility in how fields are supplied
        template <class STORAGE, template <typename> class SKIN,
                 class... FIELDS, class... EXTRA>
        struct ViewFieldsFromTypelistOrTemplateParameterPackHelper<
            STORAGE, SKIN, SOA::Typelist::typelist<FIELDS...>, EXTRA... >
        {
            static_assert(!sizeof...(EXTRA), "typelist or variadic, not both");
            using type = _View<STORAGE, SKIN, FIELDS...>;
        };
        /// helper to allow flexibility in how fields are supplied
        template <class STORAGE, template <typename> class SKIN>
        struct ViewFieldsFromTypelistOrTemplateParameterPackHelper<STORAGE, SKIN>
        {
            using type = typename
                ViewFieldsFromTypelistOrTemplateParameterPackHelper<
                STORAGE, SKIN, typename SKIN<dummy>::fields_typelist>::type;
        };
        template <class STORAGE, template <typename> class SKIN, typename... FIELDS>
        using View = typename
            impl::ViewFieldsFromTypelistOrTemplateParameterPackHelper<
            STORAGE, SKIN, FIELDS...>::type;

        /// for use by std::swap
        template <typename STORAGE,
                 template <typename> class SKIN,
                 typename... FIELDS>
        void swap(_View<STORAGE, SKIN, FIELDS...>& a,
                _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                    noexcept(a.swap(b)))
        { a.swap(b); }
    }

    /** @brief SOA view for objects with given fields (SOA storage)
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-04-10
     *
     * @tparam STORAGE      type to store the underlying ranges
     * @tparam SKIN         "skin" to dress the the interface of the object
     *                      proxying the content elemnts; this can be used to
     *                      augment the interface provided by the get<fieldtag>()
     *                      syntax with something more convenient; NullSkin leaves
     *                      the raw interface intact
     * @tparam FIELDS...    list of "field tags" describing names an types
     *                      of members (can be omitted if SKIN contains a type
     *                      fields_typelist that lists the types of fields)
     *
     * This templated class represents an object view of a number of ranges of
     * equal size with the given list of fields.  Objects are not stored as such,
     * but each of object's fields is taken from the corresponding range,
     * effectively creating a structure-of-arrays (SOA) layout which is
     * advantageous for vectorisation of algorithms. To illustrate the SOA layout,
     * first consider the normal array-of-structures (AOS) layout:
     *
     * @code
     * class Point {
     *     private:
     *         float m_x;
     *         float m_y;
     *     public:
     *         Point(float x, float y) : m_x(x), m_y(y) { }
     *         float x() const noexcept { return m_x; }
     *         float y() const noexcept { return m_y; }
     *         void setX(float x) noexcept { m_x = x; }
     *         void setY(float y) noexcept { m_y = y; }
     *         // plus some routines that do more than just setting/getting members
     *         float r2() const noexcept
     *         { return m_x * m_x + m_y * m_y; }
     * };
     *
     * using AOSPoints = std::vector<Point>;
     * using AOSPoint = Point&;
     * @endcode
     *
     * The memory layout in the example above will be x of element 0, y of element
     * 0, x of element 1, y of element 1, x of element 2, and so on.
     *
     * For the equivalent example in SOA layout, you'd have to do:
     *
     * @code
     * #include "View.h"
     *
     * // first declare member "tags" which describe the members of the notional
     * // struct (which will never exist in memory - SOA layout!)
     *  namespace PointFields {
     *     using namespace SOA::Typelist;
     *     // since we can have more than one member of the same type in our
     *     // SOA object, we have to do some typedef gymnastics so the compiler
     *     // can tell them apart
     *     struct x : wrap_type<float> { };
     *     struct y : wrap_type<float> { };
     * };
     *
     * // define the "skin", i.e. the outer guise that the naked members "wear"
     * // to make interaction with the class nice
     * template <typename NAKEDPROXY>
     * class SOAPoint : public NAKEDPROXY {
     *     public:
     *         /// define the fields (members) typelist
     *         using fields_typelist =
     *             SOA::Typelist::typelist<PointFields::x, PointFields::y>;
     *         /// forward constructor to NAKEDPROXY's constructor
     *         using NAKEDPROXY::NAKEDPROXY;
     *         /// assignment operator - forward to underlying proxy
     *         using NAKEDPROXY::operator=;
     *
     *         // add your own constructors/assignment operators here
     *
     *         float x() const noexcept
     *         { return this-> template get<PointFields::x>(); }
     *         float y() const noexcept
     *         { return this-> template get<PointFields::y>(); }
     *         void setX(float x) noexcept
     *         { this-> template get<PointFields::x>() = x; }
     *         void setY(float y) noexcept
     *         { this-> template get<PointFields::y>() = y; }
     *
     *         // again, something beyond plain setters/getters
     *         float r2() const noexcept { return x() * x() + y() * y(); }
     * };
     *
     * // define the SOA container type
     * using SOAPoints = View<
     *         // underlying type for storage
     *         std::tuple<std::vector<float>&, std::vector<float>&>,
     *         SOAPoint>;   // skin to "dress" the tuple of fields with
     * // define the SOAPoint itself
     * using SOAPoint = typename SOAPoints::proxy;
     * @endcode
     *
     * The code is very similar to the AOS layout example above, but the memory
     * layout is very different. Internally, the container has two std::vectors,
     * one which holds all the x "members" of the point structure, and one which
     * holds all the "y" members.
     *
     * Despite all the apparent complexity in defining this container, the use of
     * the contained points is practically the same in both cases. For example,
     * consider a piece of code the "normalises" all points to have unit length:
     *
     * @code
     * // normalise to unit length for the old-style AOSPoint
     * for (AOSPoint p: points) {
     *     auto ir = 1 / std::sqrt(p.r2());
     *     p.setX(p.x() * ir), p.setY(p.y() * ir);
     * }
     * @endcode
     *
     * The only change required is that a different data type is used in
     * conjunction with the SOAPoint implementation from above:
     *
     * @code
     * // normalise to unit length
     * for (SOAPoint p: points) {
     *     auto ir = 1 / std::sqrt(p.r2());
     *     p.setX(p.x() * ir), p.setY(p.y() * ir);
     * }
     * @endcode
     *
     * It is important to realise that there's nothing like the struct Point in the
     * AOS example above - the notion of a point very clearly exists, but the
     * View will not create such an object at any point in time. Instead,
     * the container provides an interface to access fields with the get<field_tag>
     * syntax, or via tuples (conversion to or assignment from tuples is
     * implemented), if simultaneous access to all fields is desired.
     *
     * Apart from these points (which are dictated by the SOA layout and efficiency
     * considerations), the class tries to follow the example of the interface of
     * std::vector as closely as possible.
     *
     * When using the View class, one must distinguish between elements
     * (and references to elements) which are stored outside the container and
     * those that are stored inside the container:
     *
     * - When stored inside the container, the element itself does not exist as
     *   such because of the SOA storage constraint; (const) references and
     *   pointers are implemented with instances of SKINned ObjectProxy and
     *   SOA(Const)Ptr. On the outside, these classes look and feel like
     *   references and pointers, but set up memory access to members/fields such
     *   that the SOA memory layout is preserved.
     * - When a single element is stored outside of the container (a temporary),
     *   the SOA layout doesn't make sense (it's just a single element, after
     *   all), and it is stored as a SKINned std::tuple with the right members.
     *   The value_type, value_reference and const_value_reference typedefs name
     *   the type of these objects.
     *
     * This distinction becomes important when using or implementing generic
     * algorithms (like types in the functor passed to std::sort) which sometimes
     * create a temporary holding a single element outside the container. For
     * illustration, here's how one would sort by increasing y in the SOAPoint
     * example from above:
     *
     * @code
     * SOAPoints& c = get_points_from_elsewhere();
     * std::sort(c.begin(), c.end(),
     *     [] (decltype(c)::value_const_reference a,
     *         decltype(c)::value_const_reference b)
     *     { return a.y() < b.y(); });
     * @endcode
     *
     * Since directly constructing a View is a bit stressful, there's a little
     * helper called make_soaview which takes some of the sting out of
     * constructing Views (in much the same way that std::make_pair and
     * std::make_tuple do).
     */
    template <class STORAGE, template <typename> class SKIN, typename... FIELDS>
    class View : public impl::View<STORAGE, SKIN, FIELDS...>
    {
        using impl::View<STORAGE, SKIN, FIELDS...>::View;
    };
    /** @brief construct a _View from a skin and a bunch of ranges
     *
     * @tparam SKIN         type of skin class to use
     * @tparam RANGES       types of the ranges supplied
     *
     * @param ranges        ranges from which to construct a _View
     *
     * @returns a View of the ranges given
     *
     * @code
     * std::vector<float> vx, vy;
     * // fill vx, vy somehow - same number of elements
     * using field_x = struct : SOA::Typelist::wrap_type<float> {};
     * using field_y = struct : SOA::Typelist::wrap_type<float> {};
     * template <typename NAKEDPROXY>
     * class SOAPoint : public NAKEDPROXY {
     *     public:
     *         /// define a list of "data member" fields
     *         using fields_typelist = SOA::Typelist::typelist<field_x, field_y>;
     *
     *         /// pull in the standard constructors and assignment operators
     *         using NAKEDPROXY::NAKEDPROXY;
     *         using NAKEDPROXY::operator=;
     *         /// if you define your own, they go here
     *
     *         float x() const noexcept
     *         { return this-> template get<field_x>(); }
     *         float y() const noexcept
     *         { return this-> template get<field_y>(); }
     *         float& x() noexcept
     *         { return this-> template get<field_x>(); }
     *         float& y() noexcept
     *         { return this-> template get<field_y>(); }
     *         float r2() const noexcept { return x() * x() + y() * y(); }
     * };
     * // construct a _View from vx, vy
     * auto view = make_soaview<SOAPoint>(vx, vy);
     * const float angle = 42.f / 180.f * M_PI;
     * const auto s = std::sin(angle), c = std::cos(angle);
     * for (auto p: view) {
     *     if (p.r2() > 1) continue;
     *     // rotate points within the unit circle by given angle
     *     std::tie(p.x(), p.y()) = std::make_pair(
     *         c * p.x() + s * p.y(), -s * p.x() + c * p.y());
     * }
     * @endcode
     */
    template <template <typename> class SKIN, typename... RANGES>
    View<std::tuple<
        typename impl::remove_rvalue_reference<RANGES>::type...>, SKIN>
    make_soaview(RANGES&&... ranges)
    {
        return View<std::tuple<
            typename impl::remove_rvalue_reference<RANGES>::type...>,
                     SKIN>(std::forward<RANGES>(ranges)...);
    }

    /** @brief create a new View from given view, including given fields
     *
     * @tparam FIELDS...        fields to extract
     * @tparam VIEW             View from which to extract
     * @tparam ARGS...          nothing, or types of two iterators (first,last(
     *
     * @param view              View from which to extract given fields
     * @param args...           either empty, or two iterators (first, last(
     *
     * @returns View of the given fields (and given range)
     *
     * @note This will only work with the new-style convienent SOAFields and
     * SOASkins defined via SOAFIELD* and SOASKIN* macros.
     *
     * Example:
     * @code
     * SOAFIELD(x, float);
     * SOAFIELD(y, float);
     * SOAFIELD(z, float);
     * SOASKIN(SOAPoint, f_x, f_y, f_z);
     * Container<std::vector, SOAPoint> c = get_from_elsewhere();
     * // create a view of only the x and y fields in c
     * auto view = view<f_x, f_y>(c);
     * // as before, but only for the first 16 elements
     * auto subview = view<f_x, f_y>(c, c.begin(), c.begin() + 16);
     * @endcode
     */
    template <typename... FIELDS, typename VIEW, typename... ARGS,
        template <class> class SKIN =
        SOA::impl::SOASkinCreatorSimple<FIELDS...>::template type>
    auto view(VIEW&& view, ARGS&&... args) -> decltype(
        make_soaview<SKIN>(impl::move_if_not_lvalue_reference(view,
                view.template range<FIELDS>(std::forward<ARGS>(args)...))...))
    {
        return make_soaview<SKIN>(impl::move_if_not_lvalue_reference(view,
                    view.template range<FIELDS>(
                        std::forward<ARGS>(args)...))...);
    }
    namespace impl {
        /// helper for the view variant that takes a skin
        template <template <class> class SKIN, typename VIEW,
                 typename... FIELDS, typename... ARGS>
        auto view_skin(VIEW&& view, SOA::Typelist::typelist<FIELDS...>,
                ARGS&&... args) -> decltype(make_soaview<SKIN>(
                        impl::move_if_not_lvalue_reference(view, view.template
                            range<FIELDS>(std::forward<ARGS>(args)...))...))
        {
            return make_soaview<SKIN>(impl::move_if_not_lvalue_reference(view,
                        view.template range<FIELDS>(
                            std::forward<ARGS>(args)...))...);
        }
    }
    /** @brief create a new View from given view, including given fields
     *
     * @tparam SKIN             custom skin (and list of fields to extract)
     * @tparam VIEW             View from which to extract
     * @tparam ARGS...          nothing, or types of two iterators (first,last(
     * @tparam FIELDS...        fields to extract
     *
     * @param view              View from which to extract given fields
     * @param args...           either empty, or two iterators (first, last(
     *
     * @returns View of the given fields (and given range)
     *
     * @note This will only work with the new-style convienent SOAFields and
     * SOASkins defined via SOAFIELD* and SOASKIN* macros.
     *
     * Example:
     * @code
     * SOAFIELD(x, float);
     * SOAFIELD(y, float);
     * SOAFIELD(z, float);
     * SOASKIN(SOAPoint, f_x, f_y, f_z);
     * Container<std::vector, SOAPoint> c = get_from_elsewhere();
     * // create a view of only the x and y fields in c
     * auto view = view<f_x, f_y>(c);
     * // as before, but only for the first 16 elements
     * auto subview = view<f_x, f_y>(c, c.begin(), c.begin() + 16);
     * @endcode
     */
    template <template <class> class SKIN, typename VIEW, typename... ARGS>
    auto view(VIEW&& view, ARGS&&... args) -> decltype(
            impl::view_skin<SKIN>(std::forward<VIEW>(view),
                typename SKIN<impl::dummy>::fields_typelist(),
                std::forward<ARGS>(args)...))
    {
        return impl::view_skin<SKIN>(std::forward<VIEW>(view),
                typename SKIN<impl::dummy>::fields_typelist(),
                std::forward<ARGS>(args)...);
    }
    /// the class that really implements View
    template <class STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    class _View {
        private:
            /// little helper to find the type contained in a range
            template <typename RANGE>
            struct contained_type {
                using type = decltype(*std::begin(std::declval<RANGE>()));
            };
            /// hide verification of FIELDS inside struct or doxygen gets confused
            struct fields_verifier {
                // storing objects without state doesn't make sense
                static_assert(1 <= sizeof...(FIELDS),
                    "need to supply at least one field");
                /// little helper to verify the FIELDS template parameter
                template <typename T>
                struct is_pod_or_wrapped : std::integral_constant<bool,
                    std::is_pod<T>::value ||
                    SOA::Typelist::is_wrapped<T>::value> {};
                // and check that all fields are either pod or wrapped
                static_assert(SOA::Utils::ALL(
                            is_pod_or_wrapped<FIELDS>::value...),
                    "Fields should be either plain old data (POD) or "
                    "wrapped types.");

                // make sure storage size agrees with number of fields
                static_assert(std::tuple_size<STORAGE>::value ==
                        sizeof...(FIELDS),
                        "Number of fields does not match storage.");
                /// little helper: check if element in a tuple F matches field
                template <size_t N, class T, typename FIELD>
                struct verify_storage_element : public std::integral_constant<bool,
                    std::is_same<typename contained_type<
                        typename std::tuple_element<N, T>::type>::type,
                        SOA::Typelist::unwrap_t<FIELD> >::value ||
                    std::is_same<typename std::remove_cv<typename contained_type<
                        typename std::tuple_element<N, T>::type>::type>::type,
                        SOA::Typelist::unwrap_t<FIELD> >::value ||
                    std::is_same<typename std::remove_cv<
                        typename std::remove_reference<typename contained_type<
                        typename std::tuple_element<N, T>::type>::type>::type>::type,
                        SOA::Typelist::unwrap_t<FIELD> >::value> {};

                /// little helper verifying the storage matches the fields
                template <size_t N, class T, typename... ARGS>
                struct verify_storage;
                /// specialisation for > 1 field
                template <size_t N, class T, typename HEAD, typename... TAIL>
                struct verify_storage<N, T, HEAD, TAIL...> : public
                    std::integral_constant<bool,
                        verify_storage_element<N, T, HEAD>::value &&
                        verify_storage<N + 1, T, TAIL...>::value> { };
                /// specialisation for one field
                template <size_t N, class T, typename HEAD>
                struct verify_storage<N, T, HEAD> : public
                    std::integral_constant<bool,
                        verify_storage_element<N, T, HEAD>::value> { };
                /// specialisation for empty tuples
                template <size_t N, typename... ARGS>
                struct verify_storage<N, std::tuple<>, ARGS...> :
                    public std::false_type { };
                // make sure the storage matches the fields provided
                static_assert(verify_storage<0, STORAGE, FIELDS...>::value,
                        "Type of provided storage must match fields.");
            };

            /// work out if what is contained in a range is constant
            template <typename RANGE>
            struct is_contained_constant : std::integral_constant<bool,
                std::is_const<RANGE>::value || std::is_const<
                    typename contained_type<RANGE>::type>::value> { };

            /// is any field a constant range?
            template <typename... ARGS>
            struct _is_any_field_constant : std::integral_constant<bool,
                SOA::Utils::ANY(is_contained_constant<ARGS>::value...)> { };

            /// helper for _is_any_field_constant: extract parameter pack
            template <template <typename...> class T, typename... ARGS>
            constexpr static bool is_any_field_constant(const T<ARGS...>*) noexcept
            { return _is_any_field_constant<ARGS...>::value; }

            /// record if the _View should be a const one
            enum {
                is_constant = _View<STORAGE, SKIN, FIELDS...
                    >::is_any_field_constant(static_cast<const STORAGE*>(nullptr))
            };

        public:
            /// type to tag this as a SOA::View
            using view_tag = struct {};
            /// type to represent sizes and indices
            using size_type = std::size_t;
            /// type to represent differences of indices
            using difference_type = std::ptrdiff_t;
            /// type to represent container itself
            using self_type = _View<STORAGE, SKIN, FIELDS...>;
            /// typelist with the given fields
            using fields_typelist = SOA::Typelist::typelist<FIELDS...>;
            /// type of the storage backend
            using SOAStorage = STORAGE;

            /// convenience function to return member number given member tag type
            template <typename MEMBER>
            static constexpr size_type memberno() noexcept
            { return fields_typelist::template find<MEMBER>(); }

        protected:
            /** @brief a tag to be used when calling ObjectProxy's constructor
             *
             * The idea is to forbid user code to call ObjectProxy's constructor
             * that takes a SOAStorage* and an index into it; this constructor is
             * dangerous, since it can be used to break const-correctness. Since
             * older versions of gcc/clang cannot see through the complex interplay
             * of typedefs, inheritance, templates and friend declarations (and
             * insist that that constructor is protected, despite a friend
             * declaration), we work around that limitation by requiring safe code
             * to supply a dummy struct of a type that the user cannot instantiate
             * or even see.
             */
            using its_safe_tag = struct {};
            /// implementation details
            struct impl_detail {
                /// little helper to check range sizes at construction time
                struct rangeSizeCheckHelper {
                    size_type m_sz;
                    template <typename T>
                    void operator()(const T& range) const {
                        if (m_sz != size_type(range.size()))
                            throw std::logic_error("_View: range sizes must match!");
                    }
                };
                /// little helper for assign(count, val)
                struct assignHelper {
                    size_type m_cnt;
                    template <typename T, typename V>
                    void operator()(std::tuple<T&, const V&> t) const noexcept(noexcept(
                            std::get<0>(t).assign(m_cnt, std::get<1>(t))))
                    { std::get<0>(t).assign(m_cnt, std::get<1>(t)); }
                };
                template <typename IT, size_type IDX>
                struct tuple_element_iterator {
                    IT m_it;
                    tuple_element_iterator& operator++() noexcept(
                            noexcept(++m_it))
                    { ++m_it; return *this; }
                    auto operator*() const noexcept(
                            noexcept(std::get<IDX>(*m_it))) -> decltype(
                            std::get<IDX>(*m_it))
                    { return std::get<IDX>(*m_it); }
                };
                template <typename IT, size_type... IDXS>
                static std::tuple<tuple_element_iterator<IT, IDXS>...>
                make_tuple_element_iterators(IT it, std::index_sequence<IDXS...>)
                {
                    return std::tuple<tuple_element_iterator<IT, IDXS>...>(
                            tuple_element_iterator<IT, IDXS...>{it});
                }
                template <size_type N, typename IT>
                static auto make_tuple_element_iterators(IT it) -> decltype(
                        make_tuple_element_iterators(it,
                            std::make_index_sequence<N>()))
                {
                    return make_tuple_element_iterators(it,
                            std::make_index_sequence<N>());
                }
                /// little helper for assign(first, last)
                struct assignHelper2 {
                    template <typename R, typename IT>
                    void operator()(std::tuple<R&, IT>& t) const noexcept(noexcept(
                                *std::get<0>(t).begin() = *std::get<1>(t)))
                    {
                        auto& it = std::get<1>(t), itend = std::get<2>(t);
                        auto& jt = std::get<0>(t).begin();
                        while (itend != it) {
                            *jt = *it;
                            ++jt, ++it;
                        }
                    }
                };
                /// helper to check contents of fields typelist
                template <typename FIELD>
                struct fields_contains : std::integral_constant<bool,
                    -1 != fields_typelist::template find<FIELD>()> {};
            };

        protected:
            /// storage backend
            SOAStorage m_storage;

            /// (naked) tuple type used as values
            using naked_value_tuple_type = typename SOA::Typelist::to_tuple<
                fields_typelist>::value_tuple;
            /// (naked) tuple type used as reference
            using naked_reference_tuple_type = typename SOA::Typelist::to_tuple<
                fields_typelist>::reference_tuple;
            /// (naked) tuple type used as const reference
            using naked_const_reference_tuple_type = typename SOA::Typelist::to_tuple<
                fields_typelist>::const_reference_tuple;

            _View() {}

        public:
            /// (notion of) type of the contained objects
            using value_type = SKIN<SOA::DressedTuple<
                naked_value_tuple_type, self_type> >;
            /// (notion of) reference to value_type (outside container)
            using value_reference = SKIN<SOA::DressedTuple<
                naked_reference_tuple_type, self_type> >;
            /// (notion of) const reference to value_type (outside container)
            using value_const_reference = SKIN<SOA::DressedTuple<
                naked_const_reference_tuple_type, self_type> >;

        public:
            /// naked proxy type (to be given a "skin" later)
            using naked_proxy = SOA::ObjectProxy<self_type>;
            friend naked_proxy;
            /// corresponding _Containers are friends
            template < template <typename...> class CONTAINER,
                     template <typename> class SKIN2, typename... FIELDS2>
            friend class _Container;
            /// type of proxy
            using proxy = SKIN<naked_proxy>;
            friend proxy;
            /// pointer to contained objects
            using pointer = SOA::Iterator<proxy>;
            friend pointer;
            /// iterator type
            using iterator = pointer;
            /// reference to contained objects
            using reference = proxy;
            /// reference to contained objects
            using const_reference = const reference;
            /// const pointer to contained objects
            using const_pointer = SOA::ConstIterator<proxy>;
            friend const_pointer;
            /// const iterator type
            using const_iterator = const_pointer;
            /// reverse iterator type
            using reverse_iterator = std::reverse_iterator<iterator>;
            /// const reverse iterator type
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;

            /// constructor from the underlying storage
            _View(const SOAStorage& other) :
                m_storage(other) { }
            /// constructor from the underlying storage
            _View(SOAStorage&& other) :
                m_storage(std::move(other)) { }
            /// constructor from a list of ranges
            template <typename... RANGES, typename std::enable_if<
                sizeof...(RANGES) == sizeof...(FIELDS), int>::type = 0>
            _View(RANGES&&... ranges) :
                m_storage(std::forward<RANGES>(ranges)...)
            {
                // verify size of ranges
                SOA::Utils::map(typename impl_detail::rangeSizeCheckHelper{
                            size_type(std::get<0>(m_storage).size())}, m_storage);
            }
            /// copy constructor
            _View(const self_type& other) = default;
            /// move constructor
            _View(self_type&& other) = default;

            /// assignment from other _View
            self_type& operator=(const self_type& other) = default;
            /// move-assignment from other _View
            self_type& operator=(self_type&& other) = default;

            /// return if the container is empty
            bool empty() const noexcept(noexcept(std::get<0>(m_storage).empty()))
            { return std::get<0>(m_storage).empty(); }
            /// return the size of the container
            size_type size() const noexcept(noexcept(
                        std::get<0>(m_storage).size()))
            { return std::get<0>(m_storage).size(); }

            /// access specified element
            typename std::enable_if<!is_constant, reference>::type
            operator[](size_type idx) noexcept
            { return reference{ &m_storage, idx, its_safe_tag() }; }
            /// access specified element (read access only)
            const_reference operator[](size_type idx) const noexcept
            {
                return const_reference{ &const_cast<SOAStorage&>(m_storage), idx,
                    its_safe_tag()  };
            }
            /// access specified element with out of bounds checking
            typename std::enable_if<!is_constant, reference>::type
            at (size_type idx)
            {
                if (idx < size()) return operator[](idx);
                else throw std::out_of_range("out of bounds");
            }
            /// access specified element with out of bounds checking (read-only)
            const_reference at(size_type idx) const
            {
                if (idx < size()) return operator[](idx);
                else throw std::out_of_range("out of bounds");
            }

            /// access first element (non-empty container)
            typename std::enable_if<!is_constant, reference>::type
            front() noexcept { return operator[](0); }
            /// access first element (non-empty container, read-only)
            const_reference front() const noexcept { return operator[](0); }
            /// access last element (non-empty container)
            typename std::enable_if<!is_constant, reference>::type
            back() noexcept(noexcept(
                        std::declval<self_type>().size()))
            { return operator[](size() - 1); }
            /// access last element (non-empty container, read-only)
            const_reference back() const noexcept(noexcept(
                        std::declval<self_type>().size()))
            { return operator[](size() - 1); }

            /// iterator pointing to first element
            typename std::enable_if<!is_constant, iterator>::type
            begin() noexcept
            { return iterator{ &m_storage, 0, its_safe_tag() }; }
            /// iterator pointing one element behind the last element
            typename std::enable_if<!is_constant, iterator>::type
            end() noexcept
            { return iterator{ &m_storage, size(), its_safe_tag() }; }

            /// const iterator pointing to first element
            const_iterator begin() const noexcept
            {
                return const_iterator{ const_cast<SOAStorage*>(&m_storage), 0,
                    its_safe_tag() };
            }
            /// const iterator pointing one element behind the last element
            const_iterator end() const noexcept
            {
                return const_iterator{ const_cast<SOAStorage*>(&m_storage), size(),
                    its_safe_tag() };
            }

            /// const iterator pointing to first element
            const_iterator cbegin() const noexcept { return begin(); }
            /// const iterator pointing one element behind the last element
            const_iterator cend() const noexcept { return end(); }

            /// get begin iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            typename std::enable_if<!is_constant, decltype(
                    std::get<MEMBERNO>(std::declval<SOAStorage>()).begin())>::type
            begin() noexcept
            { return std::get<MEMBERNO>(m_storage).begin(); }
            /// get end iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            typename std::enable_if<!is_constant, decltype(
                    std::get<MEMBERNO>(std::declval<SOAStorage>()).end())>::type
            end() noexcept
            { return std::get<MEMBERNO>(m_storage).end(); }

            /// get begin iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            typename std::enable_if<!is_constant, decltype(
                    std::get<memberno<MEMBER>()>(std::declval<SOAStorage>()).begin())>::type
            begin() noexcept
            { return std::get<memberno<MEMBER>()>(m_storage).begin(); }
            /// get end iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            typename std::enable_if<!is_constant, decltype(
                    std::get<memberno<MEMBER>()>(std::declval<SOAStorage>()).end())>::type
            end() noexcept
            { return std::get<memberno<MEMBER>()>(m_storage).end(); }

            /// get begin iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto begin() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).begin())
            { return std::get<MEMBERNO>(m_storage).begin(); }
            /// get end iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto end() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).end())
            { return std::get<MEMBERNO>(m_storage).end(); }

            /// get begin iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto begin() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).begin())
            { return std::get<memberno<MEMBER>()>(m_storage).begin(); }
            /// get end iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto end() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).end())
            { return std::get<memberno<MEMBER>()>(m_storage).end(); }

            /// get begin iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto cbegin() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).cbegin())
            { return std::get<MEMBERNO>(m_storage).cbegin(); }
            /// get end iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto cend() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).cend())
            { return std::get<MEMBERNO>(m_storage).cend(); }

            /// get begin iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto cbegin() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).cbegin())
            { return std::get<memberno<MEMBER>()>(m_storage).cbegin(); }
            /// get end iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto cend() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).cend())
            { return std::get<memberno<MEMBER>()>(m_storage).cend(); }


            /// iterator pointing to first element
            typename std::enable_if<!is_constant, reverse_iterator>::type
            rbegin() noexcept { return reverse_iterator(end()); }
            /// iterator pointing one element behind the last element
            typename std::enable_if<!is_constant, reverse_iterator>::type
            rend() noexcept { return reverse_iterator(begin()); }

            /// const iterator pointing to first element
            const_reverse_iterator rbegin() const noexcept
            { return const_reverse_iterator(end()); }
            /// const iterator pointing one element behind the last element
            const_reverse_iterator rend() const noexcept
            { return const_reverse_iterator(begin()); }

            /// const iterator pointing to first element
            const_reverse_iterator crbegin() const noexcept { return rbegin(); }
            /// const iterator pointing one element behind the last element
            const_reverse_iterator crend() const noexcept { return rend(); }

            /// get begin iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            typename std::enable_if<!is_constant, decltype(
                    std::get<MEMBERNO>(std::declval<SOAStorage>()).rbegin())>::type
            rbegin() noexcept
            { return std::get<MEMBERNO>(m_storage).rbegin(); }
            /// get end iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            typename std::enable_if<!is_constant, decltype(
                    std::get<MEMBERNO>(std::declval<SOAStorage>()).rend())>::type
            rend() noexcept
            { return std::get<MEMBERNO>(m_storage).rend(); }

            /// get begin iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            typename std::enable_if<!is_constant, decltype(
                    std::get<memberno<MEMBER>()>(std::declval<SOAStorage>()).rbegin())>::type
            rbegin() noexcept
            { return std::get<memberno<MEMBER>()>(m_storage).rbegin(); }
            /// get end iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            typename std::enable_if<!is_constant, decltype(
                    std::get<memberno<MEMBER>()>(std::declval<SOAStorage>()).rend())>::type
            rend() noexcept
            { return std::get<memberno<MEMBER>()>(m_storage).rend(); }

            /// get begin iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto rbegin() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).rbegin())
            { return std::get<MEMBERNO>(m_storage).rbegin(); }
            /// get end iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto rend() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).rend())
            { return std::get<MEMBERNO>(m_storage).rend(); }

            /// get begin iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto rbegin() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).rbegin())
            { return std::get<memberno<MEMBER>()>(m_storage).rbegin(); }
            /// get end iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto rend() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).rend())
            { return std::get<memberno<MEMBER>()>(m_storage).rend(); }

            /// get begin iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto crbegin() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).crbegin())
            { return std::get<MEMBERNO>(m_storage).crbegin(); }
            /// get end iterator of storage vector for member MEMBERNO
            template <size_type MEMBERNO>
            auto crend() const noexcept -> decltype(
                    std::get<MEMBERNO>(m_storage).crend())
            { return std::get<MEMBERNO>(m_storage).crend(); }

            /// get begin iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto crbegin() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).crbegin())
            { return std::get<memberno<MEMBER>()>(m_storage).crbegin(); }
            /// get end iterator of storage vector for member with tag MEMBER
            template <typename MEMBER>
            auto crend() const noexcept -> decltype(
                    std::get<memberno<MEMBER>()>(m_storage).crend())
            { return std::get<memberno<MEMBER>()>(m_storage).crend(); }

            /// return a const reference to the underlying SOA storage range MEMBERNO
            template <size_type MEMBERNO>
            auto range() const noexcept -> decltype(
                        boost::make_iterator_range(
                            std::get<MEMBERNO>(m_storage).begin(),
                            std::get<MEMBERNO>(m_storage).end()))
            {
                return boost::make_iterator_range(
                        std::get<MEMBERNO>(m_storage).begin(),
                        std::get<MEMBERNO>(m_storage).end());
            }
            /// return a reference to the underlying SOA storage range MEMBERNO
            template <size_type MEMBERNO>
            auto range() noexcept -> decltype(
                        boost::make_iterator_range(
                            std::get<MEMBERNO>(m_storage).begin(),
                            std::get<MEMBERNO>(m_storage).end()))
            {
                return boost::make_iterator_range(
                        std::get<MEMBERNO>(m_storage).begin(),
                        std::get<MEMBERNO>(m_storage).end());
            }
            /// return a const reference to the underlying SOA storage range MEMBER
            template <typename MEMBER>
            auto range() const noexcept -> decltype(boost::make_iterator_range(
                            std::get<memberno<MEMBER>()>(m_storage).begin(),
                            std::get<memberno<MEMBER>()>(m_storage).end()))
            {
                return boost::make_iterator_range(
                        std::get<memberno<MEMBER>()>(m_storage).begin(),
                        std::get<memberno<MEMBER>()>(m_storage).end());
            }
            /// return a reference to the underlying SOA storage range MEMBER
            template <typename MEMBER>
            auto range() noexcept -> decltype(boost::make_iterator_range(
                            std::get<memberno<MEMBER>()>(m_storage).begin(),
                            std::get<memberno<MEMBER>()>(m_storage).end()))
            {
                return boost::make_iterator_range(
                        std::get<memberno<MEMBER>()>(m_storage).begin(),
                        std::get<memberno<MEMBER>()>(m_storage).end());
            }
            /// return a reference to the underlying SOA storage range MEMBERNO
            template <size_type MEMBERNO>
            auto range(iterator first, iterator last) noexcept -> decltype(
                        boost::make_iterator_range(
                            std::get<MEMBERNO>(m_storage).begin() + (first - std::declval<iterator>()),
                            std::get<MEMBERNO>(m_storage).begin() + (last - std::declval<iterator>())))
            {
                return boost::make_iterator_range(
                        std::get<MEMBERNO>(m_storage).begin() + (first - begin()),
                        std::get<MEMBERNO>(m_storage).begin() + (last - begin()));
            }
            /// return a const reference to the underlying SOA storage range MEMBERNO
            template <size_type MEMBERNO>
            auto range(const_iterator first, const_iterator last) const noexcept -> decltype(
                        boost::make_iterator_range(
                            std::get<MEMBERNO>(m_storage).begin() + (first - std::declval<const_iterator>()),
                            std::get<MEMBERNO>(m_storage).begin() + (last - std::declval<const_iterator>())))
            {
                return boost::make_iterator_range(
                        std::get<MEMBERNO>(m_storage).begin() + (first - begin()),
                        std::get<MEMBERNO>(m_storage).begin() + (last - begin()));
            }
            /// return a reference to the underlying SOA storage range MEMBER
            template <typename MEMBER>
            auto range(iterator first, iterator last) noexcept -> decltype(
                        boost::make_iterator_range(
                            std::get<memberno<MEMBER>()>(m_storage).begin() + (first - std::declval<iterator>()),
                            std::get<memberno<MEMBER>()>(m_storage).begin() + (last - std::declval<iterator>())))
            {
                return boost::make_iterator_range(
                        std::get<memberno<MEMBER>()>(m_storage).begin() + (first - begin()),
                        std::get<memberno<MEMBER>()>(m_storage).begin() + (last - begin()));
            }
            /// return a const reference to the underlying SOA storage range MEMBER
            template <typename MEMBER>
            auto range(const_iterator first, const_iterator last) const noexcept -> decltype(
                        boost::make_iterator_range(
                            std::get<memberno<MEMBER>()>(m_storage).begin() + (first - std::declval<const_iterator>()),
                            std::get<memberno<MEMBER>()>(m_storage).begin() + (last - std::declval<const_iterator>())))
            {
                return boost::make_iterator_range(
                        std::get<memberno<MEMBER>()>(m_storage).begin() + (first - begin()),
                        std::get<memberno<MEMBER>()>(m_storage).begin() + (last - begin()));
            }

            /// assign the vector to contain count copies of val
            void assign(size_type count, const value_type& val)
            {
                if (size() >= count) {
                    std::stringstream str;
                    str << "In " << __func__ << " (" << __FILE__ << ", line " <<
                        __LINE__ << "): count must not exceed length of range.";
                    throw std::out_of_range(str.str());
                }
                SOA::Utils::map(typename impl_detail::assignHelper{count},
                        SOA::Utils::zip(m_storage, val),
                        std::make_index_sequence<sizeof...(FIELDS)>());
            }

            /// assign the vector from a range of elements in another container
            template <typename IT>
            void assign(IT first, IT last)
            {
                if (size() >= std::distance(first, last)) {
                    std::stringstream str;
                    str << "In " << __func__ << " (" << __FILE__ << ", line " <<
                        __LINE__ << "): supplied range too large.";
                    throw std::out_of_range(str.str());
                }
                SOA::Utils::map(typename impl_detail::assignHelper2(),
                        SOA::Utils::zip(m_storage,
                            impl_detail::template make_tuple_element_iterators<
                                sizeof...(FIELDS)>(first),
                            impl_detail::template make_tuple_element_iterators<
                                sizeof...(FIELDS)>(last)),
                        std::make_index_sequence<sizeof...(FIELDS)>());
            }

            /// swap contents of two containers
            void swap(self_type& other) noexcept(
                    noexcept(std::swap(m_storage, other.m_storage)))
            { std::swap(m_storage, other.m_storage); }

            /** @brief create a new view
             *
             * @tparam FIELDSORSKIN...  list of fields, or new skin
             * @tparam ARGS...          empty, or a type of pair of iterators
             *
             * @param args              none, or two iterators
             *
             * Examples:
             * @code
             * namespace XYZPoint {
             *     SOAFIELD_TRIVIAL(x, x, float);
             *     SOAFIELD_TRIVIAL(y, y, float);
             *     SOAFIELD_TRIVIAL(z, z, float);
             *     SOASKIN_TRIVIAL(Skin, x, y, z);
             * }
             * // fill container somehow
             * SOAContainer<std::vector, XYZPoint::Skin> c = getPoints();
             * // create a view with x and y fields
             * auto viewxy = c.view<XYZPoint::x, XYZPoint::y>();
             * // create a view with x and y fields, but only first half of
             * // elements
             * auto viewxyhalf = c.view<XYZPoint::x, XYZPoint::y>(
             *                          c.begin(), c.begin() + c.size() / 2);
             * // create a view with a custom skin
             * SOASKIN(RPhiSkin, XYZPoint::x, XYZPoint::y) {
             *     SOASKIN_INHERIT_DEFAULT_METHODS(RPhiSkin);
             *     float r() const
             *     {
             *         return std::sqrt(this->x() * this->x() +
             *             this->y() * this->y());
             *     }
             *     float phi() const
             *     { return std::atan2(this->y(), this->x()); }
             * };
             * auto customview = c.view<RPhiSkin>();
             * @endcode
             */
            template <typename... FIELDS2, typename... ARGS>
            auto view(ARGS&&... args) -> typename std::enable_if<!impl::is_skin<FIELDS2...>(), decltype(
                    SOA::view<FIELDS2...>(*this, std::forward<ARGS>(args)...))>::type
            { return SOA::view<FIELDS2...>(*this, std::forward<ARGS>(args)...); }
            template <template <class> class SKIN2, typename... ARGS>
            auto view(ARGS&&... args) -> typename std::enable_if<impl::is_skin<SKIN2>(), decltype(
                    SOA::view<SKIN2>(*this, std::forward<ARGS>(args)...))>::type
            { return SOA::view<SKIN2>(*this, std::forward<ARGS>(args)...); }
            /** @brief create a new view
             *
             * @tparam FIELDSORSKIN...  list of fields, or new skin
             * @tparam ARGS...          empty, or a type of pair of iterators
             *
             * @param args              none, or two iterators
             *
             * Examples:
             * @code
             * namespace XYZPoint {
             *     SOAFIELD_TRIVIAL(x, x, float);
             *     SOAFIELD_TRIVIAL(y, y, float);
             *     SOAFIELD_TRIVIAL(z, z, float);
             *     SOASKIN_TRIVIAL(Skin, x, y, z);
             * }
             * // fill container somehow
             * SOAContainer<std::vector, XYZPoint::Skin> c = getPoints();
             * // create a view with x and y fields
             * auto viewxy = c.view<XYZPoint::x, XYZPoint::y>();
             * // create a view with x and y fields, but only first half of
             * // elements
             * auto viewxyhalf = c.view<XYZPoint::x, XYZPoint::y>(
             *                          c.begin(), c.begin() + c.size() / 2);
             * // create a view with a custom skin
             * SOASKIN(RPhiSkin, XYZPoint::x, XYZPoint::y) {
             *     SOASKIN_INHERIT_DEFAULT_METHODS(RPhiSkin);
             *     float r() const
             *     {
             *         return std::sqrt(this->x() * this->x() +
             *             this->y() * this->y());
             *     }
             *     float phi() const
             *     { return std::atan2(this->y(), this->x()); }
             * };
             * auto customview = c.view<RPhiSkin>();
             * @endcode
             */
            template <typename... FIELDS2, typename... ARGS>
            auto view(ARGS&&... args) const -> typename std::enable_if<!impl::is_skin<FIELDS2...>(), decltype(
                    SOA::view<FIELDS2...>(*this, std::forward<ARGS>(args)...))>::type
            { return SOA::view<FIELDS2...>(*this, std::forward<ARGS>(args)...); }
            template <template <class> class SKIN2, typename... ARGS>
            auto view(ARGS&&... args) const -> typename std::enable_if<impl::is_skin<SKIN2>(), decltype(
                    SOA::view<SKIN2>(*this, std::forward<ARGS>(args)...))>::type
            { return SOA::view<SKIN2>(*this, std::forward<ARGS>(args)...); }
    };

    namespace impl {
        /// helper class to compare _Views (field by field)
        template <template <typename> class COMP, std::size_t N> class compare {
            private:
                /// compare field N - 1 element by element
                template <typename T>
                static bool doit(const T& a, const T& b) noexcept(noexcept(
                            COMP<decltype(a.front().template get<N - 1>())>()(
                                    a.front().template get<N - 1>(),
                                    b.front().template get<N - 1>())))
                {
                    auto last = std::min(a.size(), b.size());
                    for (auto it = a.template cbegin<N - 1>(),
                            jt = a.template cbegin<N - 1>() + last,
                            kt = b.template cbegin<N - 1>();
                            jt != it; ++it, ++kt) {
                        if (!COMP<decltype(*it)>()(*it, *kt)) return false;
                    }
                    return true;
                }
            public:
                /// trigger comparison of fields 0, ..., N - 1
                template <typename T>
                bool operator()(const T& a, const T& b) const noexcept(noexcept(
                            compare<COMP, N - 1>()(a, b)) && noexcept(doit(a, b)))
                { return compare<COMP, N - 1>()(a, b) && doit(a, b); }
        };

        /// helper class to compare _Views, specialised N = 1
        template <template <typename> class COMP> class compare<COMP, 1> {
            private:
                /// compare field 0 element by element
                template <typename T>
                static bool doit(const T& a, const T& b) noexcept(noexcept(
                            COMP<decltype(a.front().template get<0>())>()(
                                    a.front().template get<0>(),
                                    b.front().template get<0>())))
                {
                    auto last = std::min(a.size(), b.size());
                    for (auto it = a.template cbegin<0>(),
                            jt = a.template cbegin<0>() + last,
                            kt = b.template cbegin<0>();
                            jt != it; ++it, ++kt) {
                        if (!COMP<decltype(*it)>()(*it, *kt)) return false;
                    }
                    return true;
                }
            public:
                /// trigger comparison of field 0
                template <typename T>
                bool operator()(const T& a, const T& b) const
                    noexcept(noexcept(doit(a, b)))
                { return doit(a, b); }
        };
    }

    /// compare two _Views for equality
    template <typename STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    bool operator==(const _View<STORAGE, SKIN, FIELDS...>& a,
            const _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                noexcept(a.size()) && noexcept(
                    impl::compare<std::equal_to,
                    _View<STORAGE, SKIN, FIELDS...
                    >::fields_typelist::size()>()(a, b)))
    {
        if (a.size() != b.size()) return false;
        // compare one field at a time
        return impl::compare<std::equal_to,
               _View<STORAGE, SKIN, FIELDS...
                   >::fields_typelist::size()>()(a, b);
    }

    /// compare two _Views for inequality
    template <typename STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    bool operator!=(const _View<STORAGE, SKIN, FIELDS...>& a,
            const _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                noexcept(a.size()) && noexcept(
                    impl::compare<std::not_equal_to,
                    _View<STORAGE, SKIN, FIELDS...
                    >::fields_typelist::size()>()(a, b)))
    {
        if (a.size() != b.size()) return true;
        // compare one field at a time
        return impl::compare<std::not_equal_to,
               _View<STORAGE, SKIN, FIELDS...
                   >::fields_typelist::size()>()(a, b);
    }

    /// compare two _Views lexicographically using <
    template <typename STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    bool operator<(const _View<STORAGE, SKIN, FIELDS...>& a,
            const _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                noexcept(std::lexicographical_compare(a.cbegin(), a.cend(),
                        b.cbegin(), b.cend(),
                        std::less<decltype(a.front())>())))
    {
        return std::lexicographical_compare(a.cbegin(), a.cend(),
                b.cbegin(), b.cend(), std::less<decltype(a.front())>());
    }

    /// compare two _Views lexicographically using >
    template <typename STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    bool operator>(const _View<STORAGE, SKIN, FIELDS...>& a,
            const _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                noexcept(b < a))
    { return b < a; }

    /// compare two _Views lexicographically using <=
    template <typename STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    bool operator<=(const _View<STORAGE, SKIN, FIELDS...>& a,
            const _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                noexcept(!(a > b)))
    { return !(a > b); }

    /// compare two _Views lexicographically using >=
    template <typename STORAGE,
        template <typename> class SKIN, typename... FIELDS>
    bool operator>=(const _View<STORAGE, SKIN, FIELDS...>& a,
            const _View<STORAGE, SKIN, FIELDS...>& b) noexcept(
                noexcept(!(a < b)))
    { return !(a < b); }


    namespace impl {
        /// helper to zip a number of views
        template <std::size_t N> struct _zipper;
        /// specialisation for the hard case: zip two views
        template <> struct _zipper<std::size_t(2)> {
            template <class V1, class V2, typename... FIELDS1, typename... FIELDS2,
                template <class> class SKIN =
                SOA::impl::SOASkinCreatorSimple<FIELDS1..., FIELDS2...>::template type>
            static auto _doIt(V1&& v1, V2&& v2,
                    SOA::Typelist::typelist<FIELDS1...>,
                    SOA::Typelist::typelist<FIELDS2...>) -> decltype(
                        make_soaview<SKIN>(
                        impl::move_if_not_lvalue_reference(
                            std::forward<V1>(v1), v1.template range<FIELDS1>())...,
                        impl::move_if_not_lvalue_reference(
                            std::forward<V2>(v2), v2.template range<FIELDS2>())...))
            {
                return make_soaview<SKIN>(
                        impl::move_if_not_lvalue_reference(
                            std::forward<V1>(v1), v1.template range<FIELDS1>())...,
                        impl::move_if_not_lvalue_reference(
                            std::forward<V2>(v2), v2.template range<FIELDS2>())...);
            }
            template <typename V1, typename V2,
                     typename VV1 = typename std::remove_cv<
                         typename std::remove_reference<V1>::type>::type,
                     typename VV2 = typename std::remove_cv<
                         typename std::remove_reference<V2>::type>::type>
            static auto doIt(V1&& v1, V2&& v2) -> decltype(
                    _doIt(std::forward<V1>(v1), std::forward<V2>(v2),
                        typename VV1::fields_typelist(),
                        typename VV2::fields_typelist()))
            {
                return _doIt(std::forward<V1>(v1), std::forward<V2>(v2),
                        typename VV1::fields_typelist(),
                        typename VV2::fields_typelist());
            }
        };
        /// specialisation: "zipping" a single view is a no-op
        template <> struct _zipper<std::size_t(1)> {
            template <typename VIEW1>
            static VIEW1 doIt(const VIEW1& v)
            { return v; }
            template <typename VIEW1>
            static VIEW1 doIt(VIEW1& v)
            { return v; }
            template <typename VIEW1>
            static VIEW1 doIt(VIEW1&& v)
            { return std::move(v); }
        };
        /// general case: zip views one-by-one
        template <std::size_t N> struct _zipper {
            static_assert(N > 2, "Specialisations broken.");
            template <typename VIEW1, typename... VIEWS>
            static auto doIt(VIEW1&& v1, VIEWS&&... views) -> decltype(
                    _zipper<2>::doIt(std::forward<VIEW1>(v1),
                        _zipper<N - 1>::doIt(std::forward<VIEWS>(views)...)))
            {
                return _zipper<2>::doIt(std::forward<VIEW1>(v1),
                        _zipper<N - 1>::doIt(std::forward<VIEWS>(views)...));
            }
        };
    }

    /** @brief zip a number of Views
     *
     * @tparam VIEWS...     types of views to zip
     * @param views...      views to zip
     *
     * @returns a zipped view, containing all the fields in the input
     *
     * @note This will only work with the new-style convienent SOAFields and
     * SOASkins defined via SOAFIELD* and SOASKIN* macros.
     *
     * Example:
     * @code
     * SOAFIELD(x, float);
     * SOAFIELD(y, float);
     * SOAFIELD(z, float);
     * SOASKIN(SOAPoint, f_x, f_y, f_z);
     * Container<std::vector, SOAPoint> c = get_from_elsewhere();
     * // create two views
     * auto v1 = view<f_x, f_y>(c);
     * auto v2 = view<f_z>(c);
     * // zip two views
     * auto zip_view = zip(v1, v2);
     * @endcode
     */
    template <typename... VIEWS>
    auto zip(VIEWS&&... views) -> typename std::enable_if<SOA::Utils::ALL(
            sizeof...(VIEWS) > 0, SOA::Utils::is_view<typename std::remove_cv<
            typename std::remove_reference<VIEWS>::type>::type>::value...),
         decltype(impl::_zipper<sizeof...(VIEWS)>::doIt(
                     std::forward<VIEWS>(views)...))>::type
    {
        return impl::_zipper<sizeof...(VIEWS)>::doIt(
                std::forward<VIEWS>(views)...);
    }
} // namespace SOA

// [FIXME]: document
// NB. C++'s funny rules on argument dependent lookup mean that this one
// has to live in the global namespace to be found reliably. To avoid
// trouble, the function is constrained tightly with a suitable enable_if
template <template <class> class SKIN, typename... VIEWS>
auto zip(VIEWS&&... views) -> typename std::enable_if<SOA::Utils::ALL(
        SOA::impl::is_skin<SKIN>(), SOA::Utils::is_view<
        typename std::remove_cv<typename std::remove_reference<
        VIEWS>::type>::type>::value...), decltype(
        zip(std::forward<VIEWS>(views)...).template view<SKIN>())>::type
{ return zip(std::forward<VIEWS>(views)...).template view<SKIN>(); }

#endif // SOAVIEW_H

// vim: sw=4:tw=78:ft=cpp:et
