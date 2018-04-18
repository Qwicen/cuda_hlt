/** @file Container.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-04-10
 */

#ifndef SOACONTAINER_H
#define SOACONTAINER_H

#include "SOAView.h"

namespace SOA {
    // implementation details
    namespace impl {
        /// little helper for indexing to implement clear()
        struct clearHelper {
            template <typename... ARGS>
            void operator()(ARGS&... args) const noexcept(
                    noexcept(SOA::Utils::ignore((args.clear(), 0)...)))
            { SOA::Utils::ignore((args.clear(), 0)...); }
        };
        /// little helper for indexing to implement pop_back()
        struct pop_backHelper {
            template <typename... ARGS>
            void operator()(ARGS&... args) const noexcept(
                    noexcept(SOA::Utils::ignore((args.pop_back(), 0)...)))
            { SOA::Utils::ignore((args.pop_back(), 0)...); }
        };
        /// little helper for indexing to implement shrink_to_fit()
        struct shrink_to_fitHelper {
            template <typename... ARGS>
            void operator()(ARGS&... args) const noexcept(
                    noexcept(SOA::Utils::ignore((args.shrink_to_fit(), 0)...)))
            { SOA::Utils::ignore((args.shrink_to_fit(), 0)...); }
        };
        /// little helper for indexing to implement reserve()
        template <typename size_type>
        struct reserveHelper {
            size_type m_sz;
            template <typename T>
                void operator()(T& obj) const noexcept(
                        noexcept(obj.reserve(m_sz)))
                { obj.reserve(m_sz); }
        };
        /// little helper for indexing to implement capacity()
        template <typename size_type>
        struct capacityHelper {
            template <typename T>
                size_type operator()(const T& obj) const noexcept(
                        noexcept(obj.capacity()))
                { return obj.capacity(); }
        };
        /// little helper for indexing to implement max_size()
        template <typename size_type>
        struct max_sizeHelper {
            template <typename T>
                size_type operator()(const T& obj) const noexcept(
                        noexcept(obj.max_size()))
                { return obj.max_size(); }
        };
        /// little helper for push_back
        struct push_backHelper {
            template <typename T, typename V>
            void operator()(T&& t, V&& v) const noexcept(
                    noexcept(t.push_back(std::forward<V>(v))))
            { t.push_back(std::forward<V>(v)); }
        };
        /// little helper for insert(it, val)
        template <typename size_type>
        struct insertHelper {
            size_type m_idx;
            template <typename T, typename V>
            void operator()(T&& t, V&& v) const noexcept(
                    noexcept(t.insert(t.begin() + m_idx, std::forward<V>(v))))
            { t.insert(t.begin() + m_idx, std::forward<V>(v)); }
        };
        /// little helper for insert(it, count, val)
        template <typename size_type>
        struct insertHelper2 {
            size_type m_idx;
            size_type m_cnt;
            template <typename T, typename V>
            void operator()(T&& t, V&& v) const noexcept(
                    noexcept(t.insert(t.begin() + m_idx, m_cnt, std::forward<V>(v))))
            { t.insert(t.begin() + m_idx, m_cnt, std::forward<V>(v)); }
        };
        /// little helper for resize(sz) and resize(sz, val)
        template <typename size_type>
        struct resizeHelper {
            size_type m_sz;
            template <typename T>
            void operator()(T& obj) const noexcept(
                    noexcept(obj.resize(m_sz)))
            { obj.resize(m_sz); }
            template <typename T, typename V>
            void operator()(T&& t, V&& val) const noexcept(
                    noexcept(std::forward<T>(t).resize(m_sz, std::forward<V>(val))))
            { std::forward<T>(t).resize(m_sz, std::forward<V>(val)); }
        };
        /// little helper for emplace_back(args...)
        struct emplace_backHelper {
            template <typename T, typename V>
            void operator()(T&& t, V&& v) const noexcept(
                    noexcept(std::forward<T>(t).emplace_back(std::forward<V>(v))))
            { std::forward<T>(t).emplace_back(std::forward<V>(v)); }
        };
        /// little helper for emplace(pos, args...)
        template <typename size_type>
        struct emplaceHelper {
            size_type m_idx;
            template <typename T, typename V>
            void operator()(T&& t, V&& v) const noexcept(
                    noexcept(t.emplace(t.begin() + m_idx, std::forward<V>(v))))
            { t.emplace(t.begin() + m_idx, std::forward<V>(v)); }
        };
        /// little helper for erase(it)
        template <typename size_type>
        struct eraseHelper {
            size_type m_idx;
            template <typename T>
            void operator()(T& obj) const noexcept(noexcept(
                    obj.erase(obj.begin() + m_idx)))
            { obj.erase(obj.begin() + m_idx); }
        };
        /// little helper for erase(first, last)
        template <typename size_type>
        struct eraseHelper_N {
            size_type m_idx;
            size_type m_len;
            template <typename T>
            void operator()(T& obj) const noexcept(noexcept(
                    obj.erase(obj.begin() + m_idx,
                        obj.begin() + m_idx + m_len)))
            { obj.erase(obj.begin() + m_idx, obj.begin() + m_idx + m_len); }
        };
        /// little helper for assign(count, val)
        template <typename size_type>
        struct assignHelper {
            size_type m_cnt;
            template <typename T, typename V>
            void operator()(T& t, const V& v) const noexcept(noexcept(
                        t.assign(m_cnt, v)))
            { t.assign(m_cnt, v); }
        };

    } // namespace impl

    /// the implementation behind Container
    template <template <typename...> class CONTAINER,
        template <typename> class SKIN, typename... FIELDS>
    class _Container : public SOA::View<
                typename SOA::Typelist::to_tuple<SOA::Typelist::typelist<FIELDS...>
                         >::template container_tuple<CONTAINER>,
                SKIN, FIELDS...>
    {
        private:
            /// hide verification of FIELDS inside struct or doxygen gets confused
            struct fields_verifier {
                // storing objects without state doesn't make sense
                static_assert(1 <= sizeof...(FIELDS),
                    "need to supply at least one field");
                /// little helper to verify the FIELDS template parameter
                template <typename... ARGS>
                struct verify_fields;
                /// specialisation for more than one field
                template <typename HEAD, typename... TAIL>
                struct verify_fields<HEAD, TAIL...>
                {
                    /// true if HEAD and TAIL verify okay
                    enum {
                        value = (std::is_pod<HEAD>::value ||
                            SOA::Typelist::is_wrapped<
                            HEAD>::value) && verify_fields<TAIL...>::value
                    };
                };
                /// specialisation for one field
                template <typename HEAD>
                struct verify_fields<HEAD>
                {
                    /// true if HEAD verifies okay
                    enum {
                        value = std::is_pod<HEAD>::value ||
                            SOA::Typelist::is_wrapped<HEAD>::value
                    };
                };
                // make sure fields are either POD or wrapped types
                static_assert(verify_fields<FIELDS...>::value,
                    "Fields should be either plain old data (POD) or "
                    "wrapped types.");
            };

            /// give a short and convenient name to base class
            using BASE = SOA::View<
                typename SOA::Typelist::to_tuple<SOA::Typelist::typelist<FIELDS...>
                         >::template container_tuple<CONTAINER>,
                SKIN, FIELDS...>;
        public:
            /// type for sizes
            using size_type = typename BASE::size_type;
            /// type for distances between iterators
            using difference_type = typename BASE::difference_type;
            /// pointer type
            using pointer = typename BASE::pointer;
            /// const pointer type
            using const_pointer = typename BASE::const_pointer;
            /// iterator type
            using iterator = typename BASE::iterator;
            /// const iterator type
            using const_iterator = typename BASE::const_iterator;
            /// reference type
            using reference = typename BASE::reference;
            /// const reference type
            using const_reference = typename BASE::const_reference;
            /// reverse iterator type
            using reverse_iterator = typename BASE::reverse_iterator;
            /// const reverse iterator type
            using const_reverse_iterator = typename BASE::const_reverse_iterator;
            /// (notion of) type of the contained objects
            using value_type = typename BASE::value_type;
            /// (notion of) reference to value_type (outside container)
            using value_reference = typename BASE::value_reference;
            /// (notion of) const reference to value_type (outside container)
            using value_const_reference = typename BASE::value_const_reference;
            /// type of the underlying storage
            using SOAStorage = typename BASE::SOAStorage;
            /// type to represent container itself
            using self_type = _Container<CONTAINER, SKIN, FIELDS...>;
            /// typelist with the given fields
            using fields_typelist = typename BASE::fields_typelist;

            /// convenience function to return member number given member tag type
            template <typename MEMBER>
            static constexpr size_type memberno() noexcept
            { return fields_typelist::template find<MEMBER>(); }

        protected:
            /// (naked) tuple type used as values
            using naked_value_tuple_type =
                typename BASE::naked_value_tuple_type;
            /// (naked) tuple type used as reference
            using naked_reference_tuple_type =
                typename BASE::naked_reference_tuple_type;
            /// (naked) tuple type used as const reference
            using naked_const_reference_tuple_type =
                typename BASE::naked_const_reference_tuple_type;
            /// use the parent's its_safe_tag (make sure this stays protected)
            using its_safe_tag = typename BASE::its_safe_tag;

        public:
            /// default constructor
            _Container() : BASE() { }
            /// copy constructor
            _Container(const self_type& other) = default;
            /// move constructor
            _Container(self_type&& other) = default;
            /// assignment from other _Container
            self_type& operator=(const self_type& other) = default;
            /// move-assignment from other _Container
            self_type& operator=(self_type&& other) = default;

            /// fill container with count copies of val
            _Container(size_type count, const value_type& val) : BASE()
            {
                reserve(count);
                assign(count, val);
            }
            /// fill container with count (default constructed) elements
            _Container(size_type count) : _Container(count, value_type()) { }
            /// fill container with elements from other container
            template <typename IT>
            _Container(IT first, IT last) : BASE()
            {
                reserve(std::distance(first, last));
                assign(first, last);
            }

            /// std::initializer_list constructor
            _Container(std::initializer_list<naked_value_tuple_type> listing) :
                BASE()
            {
                reserve(listing.size());
                assign(listing.begin(), listing.end());
            }

        public:
            /// clear the container
            void clear() noexcept(noexcept(
                        SOA::Utils::apply(impl::clearHelper(),
                            std::declval<SOAStorage&>())))
            { SOA::Utils::apply(impl::clearHelper(), this->m_storage); }

            /// pop the last element off the container
            void pop_back() noexcept(noexcept(
                        SOA::Utils::apply(impl::pop_backHelper(),
                            std::declval<SOAStorage&>())))
            { SOA::Utils::apply(impl::pop_backHelper(), this->m_storage); }

            /// shrink the underlying storage of the container to fit its size
            void shrink_to_fit() noexcept(noexcept(
                        SOA::Utils::apply(impl::shrink_to_fitHelper(),
                            std::declval<SOAStorage&>())))
            {
                SOA::Utils::apply(impl::shrink_to_fitHelper(),
                        this->m_storage);
            }

            /// reserve space for at least sz elements
            void reserve(size_type sz) noexcept(noexcept(
                SOA::Utils::map(
                        typename SOA::impl::reserveHelper<size_type>{sz},
                        std::declval<self_type*>()->m_storage)))
            {
                SOA::Utils::map(
                        typename SOA::impl::reserveHelper<size_type>{sz}, this->m_storage);
            }

            /// return capacity of container
            size_type capacity() const
            {
                return SOA::Utils::foldl<size_type>(
                        [] (size_type a, size_type b) noexcept
                        { return std::min(a, b); },
                        SOA::Utils::map(
                            typename SOA::impl::capacityHelper<size_type>(), this->m_storage),
                        std::numeric_limits<size_type>::max());
            }

            /// return maximal size of container
            size_type max_size() const
            {
                return SOA::Utils::foldl<size_type>(
                        [] (size_type a, size_type b) noexcept
                        { return std::min(a, b); },
                        SOA::Utils::map(
                            typename SOA::impl::max_sizeHelper<size_type>(), this->m_storage),
                        std::numeric_limits<size_type>::max());
            }

            /// resize container (use default-constructed values if container grows)
            void resize(size_type sz) noexcept(noexcept(
                        SOA::Utils::map(
                            typename SOA::impl::resizeHelper<size_type>{sz},
                            std::declval<SOAStorage&>())))
            {
                SOA::Utils::map(
                        typename SOA::impl::resizeHelper<size_type>{sz}, this->m_storage);
            }

            /// resize the container (append val if the container grows)
            void resize(size_type sz, const value_type& val) noexcept(noexcept(
                SOA::Utils::apply_zip(typename SOA::impl::resizeHelper<size_type>{sz},
                        std::declval<SOAStorage&>(), val)))
            {
                SOA::Utils::apply_zip(typename SOA::impl::resizeHelper<size_type>{sz},
                        this->m_storage, val);
            }

            /// push an element at the back of the array
            template <typename T>
            typename std::enable_if<std::is_constructible<value_type,
                     T>::value>::type
            push_back(T&& val) noexcept(noexcept(SOA::Utils::apply_zip(
                            impl::push_backHelper(),
                            std::declval<SOAStorage&>(),
                            std::forward<T>(val))))
            {
                SOA::Utils::apply_zip(
                        impl::push_backHelper(), this->m_storage,
                        std::forward<T>(val));
            }

            /// insert a value at the given position
            template <typename T>
            typename std::enable_if<std::is_constructible<value_type,
                     T>::value, iterator>::type
            insert(const_iterator pos, T&& val) noexcept(
                    noexcept(SOA::Utils::apply_zip(
                            impl::insertHelper<size_type>{0},
                            std::declval<SOAStorage&>(),
                            std::forward<T>(val))))
            {
                assert((*pos).m_storage == &this->m_storage);
                SOA::Utils::apply_zip(
                        impl::insertHelper<size_type>{pos.m_proxy.m_index},
                        this->m_storage, std::forward<T>(val));
                return iterator{ pos.m_proxy.m_storage, pos.m_proxy.m_index,
                    its_safe_tag() };
            }

            /// insert count copies of value at the given position
            iterator insert(const_iterator pos, size_type count,
                const value_type& val) noexcept(
                    noexcept(SOA::Utils::apply_zip(
                            impl::insertHelper2<size_type>{0, 0},
                            std::declval<SOAStorage&>(), val)))
            {
                assert((*pos).m_storage == &this->m_storage);
                SOA::Utils::apply_zip(
                        impl::insertHelper2<size_type>{pos.m_proxy.m_index, count},
                        this->m_storage, val);
                return iterator{ pos.m_proxy.m_storage, pos.m_proxy.m_index,
                    its_safe_tag() };
            }

            /// insert elements between first and last at position pos
            template <typename IT>
            iterator insert(const_iterator pos, IT first, IT last)
            {
                // FIXME: terrible implementation!!!
                // for iterators which support multiple passes over the data, this
                // should fill field by field
                // moreover, if we can determine the distance between first and
                // last, we should make a hole of the right size to avoid moving
                // data more than once
                iterator retVal(pos.m_proxy.m_storage, pos.m_proxy.m_index,
                        its_safe_tag());
                while (first != last) { insert(pos, *first); ++first; ++pos; }
                return retVal;
            }

            /// erase an element at the given position
            iterator erase(const_iterator pos) noexcept(noexcept(
                        SOA::Utils::map(
                            typename SOA::impl::eraseHelper<size_type>{
                                    static_cast<size_type>(pos - pos)},
                            std::declval<self_type*>()->m_storage)))
            {
                assert((*pos).m_storage == &this->m_storage);
                SOA::Utils::map(
                        typename SOA::impl::eraseHelper<size_type>{ pos.m_proxy.m_index },
                        this->m_storage);
                return iterator{ pos.m_proxy.m_storage, pos.m_proxy.m_index,
                    its_safe_tag() };
            }

            /// erase elements from first to last
            iterator erase(const_iterator first, const_iterator last) noexcept(
                    noexcept(
                        SOA::Utils::map(
                            typename SOA::impl::eraseHelper_N<size_type>{
                                static_cast<size_type>(first - first),
                                static_cast<size_type>(last - first)},
                                std::declval<self_type*>()->m_storage)))
            {
                assert((*first).m_storage == &this->m_storage);
                assert((*last).m_storage == &this->m_storage);
                SOA::Utils::map(
                        typename SOA::impl::eraseHelper_N<size_type>{
                                first.m_proxy.m_index,
                                static_cast<size_type>(last - first)},
                            this->m_storage);
                return iterator{ first.m_proxy.m_storage, first.m_proxy.m_index,
                    its_safe_tag() };
            }

            /// assign the vector to contain count copies of val
            void assign(size_type count, const value_type& val) noexcept(noexcept(
                        SOA::Utils::apply_zip(
                            SOA::impl::assignHelper<size_type>{count},
                            std::declval<SOAStorage&>(), val)))
            {
                SOA::Utils::apply_zip(
                        SOA::impl::assignHelper<size_type>{count},
                        this->m_storage, val);
            }

            /// assign the vector from a range of elements in another container
            template <typename IT>
            void assign(IT first, IT last) noexcept(
                    noexcept(std::declval<self_type*>()->empty()) &&
                    noexcept(std::declval<self_type*>()->clear()) &&
                    noexcept(std::declval<self_type*>()->insert(
                            std::declval<self_type*>()->begin(), first, last)))
            {
                if (!BASE::empty()) clear();
                // naively, one would use a reserve(distance(first, last)) here,
                // but I'm not sure how this will work for various kinds of
                // special iterators - callers are expected to reserve beforehand
                // if the required size is known
                insert(BASE::begin(), first, last);
            }

            /// construct new element at end of container (in-place) from args
            template <typename... ARGS, typename std::enable_if<
                SOA::Utils::ALL(std::is_convertible<ARGS,
                        SOA::Typelist::unwrap_t<FIELDS> >::value...)>::type* = nullptr>
            void emplace_back(ARGS&&... args) noexcept(noexcept(
                SOA::Utils::apply_zip(SOA::impl::emplace_backHelper{}, std::declval<SOAStorage&>(),
                        std::forward_as_tuple(std::forward<ARGS>(args)...))))
            {
                static_assert(std::is_constructible<naked_value_tuple_type,
                        ARGS...>::value || std::is_constructible<value_type,
                        ARGS...>::value, "Wrong arguments to emplace_back.");
                SOA::Utils::apply_zip(SOA::impl::emplace_backHelper{}, this->m_storage,
                        std::forward_as_tuple(std::forward<ARGS>(args)...));
            }

            /// construct a new element at the end of container from naked_value_tuple_type
            void emplace_back(naked_value_tuple_type&& val) noexcept(noexcept(
                SOA::Utils::apply_zip(SOA::impl::emplace_backHelper{}, std::declval<SOAStorage&>(),
                        std::forward<naked_value_tuple_type>(val))))
            {
                SOA::Utils::apply_zip(SOA::impl::emplace_backHelper{}, this->m_storage,
                        std::forward<naked_value_tuple_type>(val));
            }

            /// construct a new element at the end of container from value_type
            void emplace_back(value_type&& val) noexcept(noexcept(
                SOA::Utils::apply_zip(SOA::impl::emplace_backHelper{}, std::declval<SOAStorage&>(),
                        std::forward<value_type>(val))))
            {
                SOA::Utils::apply_zip(SOA::impl::emplace_backHelper{}, this->m_storage,
                        std::forward<value_type>(val));
            }

            /// construct a new element by calling the user-defined constructor of the proxy object
            template <typename... ARGS, typename std::enable_if<
                !SOA::Utils::ALL(std::is_convertible<ARGS,
                        SOA::Typelist::unwrap_t<FIELDS> >::value...)>::type* = nullptr>
            void emplace_back(ARGS&&... args) noexcept(noexcept(
                        std::declval<self_type>().push_back(value_type(
                                std::forward<ARGS>(args)...))))
            { this->push_back(value_type(std::forward<ARGS>(args)...)); }

            /// construct element by calling user-defined constructor of proxy
            template <typename... ARGS, typename std::enable_if<
                sizeof...(ARGS) != sizeof...(FIELDS)>::type* = nullptr>
            void emplace_back(ARGS&&... args) noexcept(noexcept(
                        std::declval<self_type>().push_back(value_type(
                                std::forward<ARGS>(args)...))))
            { this->push_back(value_type(std::forward<ARGS>(args)...)); }

            /// construct new element at position pos (in-place) from args
            template <typename... ARGS>
            iterator emplace(const_iterator pos, ARGS&&... args) noexcept(
                    noexcept(SOA::Utils::apply_zip(
                            SOA::impl::emplaceHelper<size_type>{ pos.m_proxy.m_index },
                            std::declval<SOAStorage&>(),
                            std::forward_as_tuple(std::forward<ARGS>(args)...))))
            {
                static_assert(std::is_constructible<naked_value_tuple_type,
                        ARGS...>::value || std::is_constructible<value_type,
                        ARGS...>::value, "Wrong arguments to emplace.");
                assert(&this->m_storage == (*pos).m_storage);
                SOA::Utils::apply_zip(
                        SOA::impl::emplaceHelper<size_type>{ pos.m_proxy.m_index },
                        this->m_storage,
                        std::forward_as_tuple(std::forward<ARGS>(args)...));
                return iterator{ pos.m_proxy.m_storage, pos.m_proxy.m_index,
                    its_safe_tag() };
            }

            /// construct a new element at position pos from naked_value_tuple_type
            iterator emplace(const_iterator pos,
                naked_value_tuple_type&& val) noexcept(noexcept(
                        SOA::Utils::apply_zip(
                            SOA::impl::emplaceHelper<size_type>{ pos.m_proxy.m_index },
                            std::declval<SOAStorage&>(),
                            std::forward<naked_value_tuple_type>(val))))
            {
                SOA::Utils::apply_zip(
                        SOA::impl::emplaceHelper<size_type>{ pos.m_proxy.m_index },
                        this->m_storage,
                        std::forward<naked_value_tuple_type>(val));
                return iterator{ pos.m_proxy.m_storage, pos.m_proxy.m_index,
                    its_safe_tag() };
            }

            /// construct a new element at position pos from value_type
            iterator emplace(const_iterator pos,
                value_type&& val) noexcept(noexcept(
                        SOA::Utils::apply_zip(
                            SOA::impl::emplaceHelper<size_type>{ pos.m_proxy.m_index },
                            std::declval<SOAStorage&>(),
                            std::forward<value_type>(val))))
            {
                SOA::Utils::apply_zip(
                        SOA::impl::emplaceHelper<size_type>{ pos.m_proxy.m_index },
                        this->m_storage,
                        std::forward<value_type>(val));
                return iterator{ pos.m_proxy.m_storage, pos.m_proxy.m_index,
                    its_safe_tag() };
            }
    };

    /// more _Container implementation details
    namespace _ContainerImpl {
        /// helper to allow flexibility in how fields are supplied
        template <template <typename...> class CONTAINER,
                 template <typename> class SKIN,
                 class... TYPELISTORFIELDS>
        struct ContainerFieldsFromTypelistOrTemplateParameterPackHelper {
            using type = _Container<CONTAINER, SKIN, TYPELISTORFIELDS...>;
        };
        /// helper to allow flexibility in how fields are supplied
        template <template <typename...> class CONTAINER,
                 template <typename> class SKIN,
                 class... FIELDS, class... EXTRA>
        struct ContainerFieldsFromTypelistOrTemplateParameterPackHelper<
            CONTAINER, SKIN, SOA::Typelist::typelist<FIELDS...>, EXTRA... >
        {
            static_assert(!sizeof...(EXTRA), "typelist or variadic, not both");
            using type = _Container<CONTAINER, SKIN, FIELDS...>;
        };
        /// helper to allow flexibility in how fields are supplied
        template <template <typename...> class CONTAINER,
                 template <typename> class SKIN>
        struct ContainerFieldsFromTypelistOrTemplateParameterPackHelper<CONTAINER, SKIN>
        {
            using type = typename
                ContainerFieldsFromTypelistOrTemplateParameterPackHelper<
                CONTAINER, SKIN, typename SKIN<SOA::impl::dummy>::fields_typelist>::type;
        };
        template <template <typename...> class CONTAINER, template <typename> class SKIN,
                 typename... FIELDS>
        using Container = typename
            _ContainerImpl::ContainerFieldsFromTypelistOrTemplateParameterPackHelper<
            CONTAINER, SKIN, FIELDS...>::type;
    }

    /** @brief container class for objects (SOA storage)
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-04-10
     *
     * @tparam CONTAINER    underlying container type (anything that follows
     *                      std::vector's interface is fine)
     * @tparam SKIN         "skin" to dress the the interface of the object
     *                      proxying the content elemnts; this can be used to
     *                      augment the interface provided by the get<fieldtag>()
     *                      syntax with something more convenient; NullSkin leaves
     *                      the raw interface intact
     * @tparam FIELDS...    list of "field tags" describing names an types
     *                      of members (can be omitted if SKIN contains a type
     *                      fields_typelist that lists the types of fields)
     *
     * This class represents a container of objects with the given list of fields,
     * the objects are not stored as such, but each of object's fields gets its own
     * storage array, effectively creating a structure-of-arrays (SOA) layout which
     * is advantageous for vectorisation of algorithms. To illustrate the SOA
     * layout, first consider the normal array-of-structures (AOS) layout:
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
     * #include "Container.h"
     *
     * // first declare member "tags" which describe the members of the notional
     * // struct (which will never exist in memory - SOA layout!)
     *  namespace PointFields {
     *     using namespace SOA::Typelist;
     *     // since we can have more than one member of the same type in our
     *     // SOA object, we have to do some typedef gymnastics so the compiler
     *     // can tell them apart
     *     using x = struct : public wrap_type<float> {};
     *     using y = struct : public wrap_type<float> {};
     * };
     *
     * // define the "skin", i.e. the outer guise that the naked members "wear"
     * // to make interaction with the class nice
     * template <typename NAKEDPROXY>
     * class SOAPoint : public NAKEDPROXY {
     *     public:
     *         /// declare "member fields"
     *         using fields_typelist = SOA::Typelist::typelist<
     *             PointFields::x, PointFields::y>;
     *
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
     * using SOAPoints = _Container<
     *         std::vector, // underlying type for each field
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
     * _Container will not create such an object at any point in time. Instead,
     * the container provides an interface to access fields with the get<field_tag>
     * syntax, or via tuples (conversion to or assignment from tuples is
     * implemented), if simultaneous access to all fields is desired.
     *
     * Apart from these points (which are dictated by the SOA layout and efficiency
     * considerations), the class tries to follow the example of the interface of
     * std::vector as closely as possible.
     *
     * When using the _Container class, one must distinguish between elements
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
     */
    template <template <typename...> class CONTAINER, template <typename> class SKIN,
             typename... FIELDS>
    class Container : public _ContainerImpl::Container<CONTAINER, SKIN, FIELDS...>
    {
        using _ContainerImpl::Container<CONTAINER, SKIN, FIELDS...>::Container;
    };
} // namespace SOA

#endif // SOACONTAINER_H

// vim: sw=4:tw=78:ft=cpp:et
