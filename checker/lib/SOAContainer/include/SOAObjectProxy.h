/** @file SOAObjectProxy.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-04-10
 */

#ifndef SOAOBJECTPROXY_H
#define SOAOBJECTPROXY_H

#include <tuple>
#include <memory>

#include "SOATypelist.h"
#include "SOATypelistUtils.h"
#include "SOAUtils.h"
#include "c++14_compat.h"

/// namespace to encapsulate SOA stuff
namespace SOA {
    // forward declarations
    template <typename PROXY> class ConstIterator;
    template <typename PROXY> class Iterator;
    template <class STORAGE,
             template <typename> class SKIN, typename... FIELDS>
    class _View;
    template <template <typename...> class CONTAINER,
             template <typename> class SKIN, typename... FIELDS>
    class _Container;
} // namespace SOA

namespace SOA {
    /** @brief proxy object for the elements stored in the container.
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-04-10
     *
     * Conceptually, the _Container contains a collection of objects
     * which have some data members. To optimise the data access patterns
     * in memory, the _Container doesn't store the objects themselves,
     * but containers which each store a different member. That means the
     * conceptual objects mentioned above do not exist as such. The
     * ObjectProxy class stands in for these objects, and provides
     * support for accessing members, assigment of the whole conceptual
     * object from a tuple of its members, and similar functionality.
     */
    template <typename PARENTCONTAINER>
    class ObjectProxy {
        public:
            /// type of parent container
            using parent_type = PARENTCONTAINER;
            /// type to refer to this type
            using self_type = ObjectProxy<PARENTCONTAINER>;
            /// type to hold the distance between two iterators
            using difference_type = typename parent_type::difference_type;
            /// type to hold the size of a container
            using size_type = typename parent_type::size_type;
            /// type to which ObjectProxy converts and can be assigned from
            using value_type = typename parent_type::value_type;
            /// type for tuple of references to members
            using reference = typename parent_type::value_reference;
            /// type for tuple of const references to members
            using const_reference = typename parent_type::value_const_reference;
            /// type of a pointer
            using pointer = SOA::Iterator<typename parent_type::proxy>;
            /// type of a const pointer
            using const_pointer = SOA::ConstIterator<typename parent_type::proxy>;

        protected:
            /// type used by the parent container to hold the SOA data
            using SOAStorage = typename parent_type::SOAStorage;
            /// typelist of fields
            using fields_typelist = typename parent_type::fields_typelist;

            SOAStorage* m_storage;  ///< underlying SOA storage of members
            size_type m_index;      ///< index into underlying SOA storage

            // _Container is allowed to invoke the private constructor
            friend parent_type;
            // so is the pointer/iterator type
            friend pointer;
            // and the const pointer/iterator type
            friend const_pointer;
            /// corresponding _Containers are friends
            template <template <typename...> class CONTAINER,
                     template <typename> class SKIN, typename... FIELDS>
            friend class _Container;

        public:
            /// constructor is private, but parent container is a friend
            explicit ObjectProxy(
                    SOAStorage* storage, size_type index,
                    typename parent_type::its_safe_tag) noexcept :
                m_storage(storage), m_index(index)
            {
                /* FIXME: This used to be a protected constructor, with the
                 * appropriate friend declarations, but apparently only the
                 * latest C++ compilers saw through that, and compiled things
                 * without a problem. Therefore, the constructor became
                 * public, and an explicit one, and it now has a dummy
                 * argument of type _SOAView<...>::its_safe_tag that is
                 * private to _SOAView, and a few of its friends. That way,
                 * user code cannot call this constructor, since they cannot
                 * supply the third argument, which essentially amounts to the
                 * same kind of protection that the old version enjoyed.
                 */
            }

        private:
            /// little helper to implement conversion to tuple
            struct helper {
                size_type m_idx;
                /// convert to tuple of values
                template <typename T, std::size_t... IDX>
                auto to_value(const T& obj, std::index_sequence<IDX...>) const
                    noexcept(noexcept(std::make_tuple(
                                    std::get<IDX>(obj)[m_idx]...)))
                    -> decltype(std::make_tuple(std::get<IDX>(obj)[m_idx]...))
                { return std::make_tuple(std::get<IDX>(obj)[m_idx]...); }
                /// convert to tuple of references
                template <typename T, std::size_t... IDX>
                auto to_reference(T& obj, std::index_sequence<IDX...>) const
                    noexcept(noexcept(std::tie(std::get<IDX>(obj)[m_idx]...)))
                    -> decltype(std::tie(std::get<IDX>(obj)[m_idx]...))
                { return std::tie(std::get<IDX>(obj)[m_idx]...); }
                /// convert to tuple of const references
                template <typename T, std::size_t... IDX>
                auto to_const_reference(
                        const T& obj, std::index_sequence<IDX...>) const
                    noexcept(noexcept(std::tie(std::get<IDX>(obj)[m_idx]...)))
                    -> decltype(std::tie(std::get<IDX>(obj)[m_idx]...))
                { return std::tie(std::get<IDX>(obj)[m_idx]...); }
            };

            struct swapHelper {
                size_type m_idx1;
                size_type m_idx2;
                template <typename T>
                void operator()(std::tuple<T&, T&> obj) const noexcept(
                        noexcept(std::swap(std::get<0>(obj)[m_idx1],
                                std::get<1>(obj)[m_idx2])))
                {
                    std::swap(std::get<0>(obj)[m_idx1],
                            std::get<1>(obj)[m_idx2]);
                }
            };

        public:
            /// default constructor
            ObjectProxy() noexcept = default;
            /// copy constructor
            ObjectProxy(const self_type& other) noexcept = default;
            /// move constructor
            ObjectProxy(self_type&& other) noexcept = default;

            /// convert to tuple of member contents
            operator value_type() const noexcept(noexcept(
                        helper{ m_index }.to_value(*m_storage,
                            std::make_index_sequence<
                            fields_typelist::size()>())))
            {
                return helper{ m_index }.to_value(*m_storage,
                        std::make_index_sequence<fields_typelist::size()>());
            }
            /// convert to tuple of references to members
            operator reference() noexcept(noexcept(
                        helper{ m_index }.to_reference(*m_storage,
                            std::make_index_sequence<
                            fields_typelist::size()>())))
            {
                return helper{ m_index }.to_reference(*m_storage,
                        std::make_index_sequence<fields_typelist::size()>());
            }
            /// convert to tuple of const references to members
            operator const_reference() const noexcept(noexcept(
                        helper{ m_index }.to_const_reference(*m_storage,
                            std::make_index_sequence<
                            fields_typelist::size()>())))
            {
                return helper{ m_index }.to_const_reference(*m_storage,
                        std::make_index_sequence<
                        fields_typelist::size()>());
            }

            /// assign from tuple of member contents
            self_type& operator=(const value_type& other) noexcept(noexcept(
                        reference(std::declval<self_type>()) == other))
            { reference(*this) = other; return *this; }

            /// assign from tuple of member contents (move semantics)
            self_type& operator=(value_type&& other) noexcept(noexcept(
                        reference(std::declval<self_type>()) == std::move(other)))
            { reference(*this) = std::move(other); return *this; }

            /// assign from tuple of member contents
            self_type& operator=(const reference& other) noexcept(noexcept(
                        reference(std::declval<self_type>()) == other))
            { reference(*this) = other; return *this; }

            /// assign from tuple of member contents (move semantics)
            self_type& operator=(reference&& other) noexcept(noexcept(
                        reference(std::declval<self_type>()) == std::move(other)))
            { reference(*this) = std::move(other); return *this; }

            /// assign from tuple of member contents
            self_type& operator=(const const_reference& other) noexcept(
                    noexcept(reference(std::declval<self_type>()) == other))
            { reference(*this) = other; return *this; }

            /// assignment operator (value semantics)
            self_type& operator=(const self_type& other) noexcept(noexcept(
                        reference(std::declval<self_type>()) = const_reference(other)))
            {
                if (other.m_storage != m_storage || other.m_index != m_index)
                    reference(*this) = const_reference(other);
                return *this;
            }

            /// move assignment operator (value semantics)
            self_type& operator=(self_type&& other) noexcept(noexcept(
                        reference(std::declval<self_type>()) = std::move(reference(other))))
            {
                if (other.m_storage != m_storage || other.m_index != m_index)
                    reference(*this) = std::move(reference(other));
                return *this;
            }

            /// assignment (pointer-like semantics)
            self_type& assign(const self_type& other) noexcept
            {
                if (this != std::addressof(other))
                    m_storage = other.m_storage, m_index = other.m_index;
                return *this;
            }
            /// move assignment (pointer-like semantics)
            self_type& assign(self_type&& other) noexcept
            {
                if (this != std::addressof(other))
                    m_storage = std::move(other.m_storage),
                              m_index = std::move(other.m_index);
                return *this;
            }

            /// access to member by number
            template <size_type MEMBERNO>
            auto get() noexcept -> decltype(
                    std::get<MEMBERNO>(*m_storage)[m_index])
            { return std::get<MEMBERNO>(*m_storage)[m_index]; }
            /// access to member by "member tag"
            template <typename MEMBER, size_type MEMBERNO =
                PARENTCONTAINER::template memberno<MEMBER>()>
            auto get() noexcept -> decltype(
                    std::get<MEMBERNO>(*m_storage)[m_index])
            {
                static_assert(PARENTCONTAINER::template memberno<MEMBER>() ==
                        MEMBERNO, "Called with wrong template argument(s).");
                return std::get<MEMBERNO>(*m_storage)[m_index];
            }
            /// access to member by number (read-only)
            template <size_type MEMBERNO>
            auto get() const noexcept -> decltype(
                    std::get<MEMBERNO>(*m_storage)[m_index])
            { return std::get<MEMBERNO>(*m_storage)[m_index]; }
            /// access to member by "member tag" (read-only)
            template <typename MEMBER, size_type MEMBERNO =
                PARENTCONTAINER::template memberno<MEMBER>()>
            auto get() const noexcept -> decltype(
                    std::get<MEMBERNO>(*m_storage)[m_index])
            {
                static_assert(PARENTCONTAINER::template memberno<MEMBER>() ==
                        MEMBERNO, "Called with wrong template argument(s).");
                return std::get<MEMBERNO>(*m_storage)[m_index];
            }

            /// swap the contents of two ObjectProxy instances
            void swap(self_type& other) noexcept(noexcept(
                        SOA::Utils::map(swapHelper{ m_index, other.m_index },
                        SOA::Utils::zip(*m_storage, *other.m_storage))))
            {
                SOA::Utils::map(swapHelper{ m_index, other.m_index },
                        SOA::Utils::zip(*m_storage, *other.m_storage));
            }

            /// comparison (equality)
            bool operator==(const value_type& other) const noexcept
            { return const_reference(*this) == other; }
            /// comparison (inequality)
            bool operator!=(const value_type& other) const noexcept
            { return const_reference(*this) != other; }
            /// comparison (less than)
            bool operator<(const value_type& other) const noexcept
            { return const_reference(*this) < other; }
            /// comparison (greater than)
            bool operator>(const value_type& other) const noexcept
            { return const_reference(*this) > other; }
            /// comparison (less than or equal to)
            bool operator<=(const value_type& other) const noexcept
            { return const_reference(*this) <= other; }
            /// comparison (greater than or equal to)
            bool operator>=(const value_type& other) const noexcept
            { return const_reference(*this) >= other; }

            /// comparison (equality)
            bool operator==(const self_type& other) const noexcept
            { return const_reference(*this) == const_reference(other); }
            /// comparison (inequality)
            bool operator!=(const self_type& other) const noexcept
            { return const_reference(*this) != const_reference(other); }
            /// comparison (less than)
            bool operator<(const self_type& other) const noexcept
            { return const_reference(*this) < const_reference(other); }
            /// comparison (greater than)
            bool operator>(const self_type& other) const noexcept
            { return const_reference(*this) > const_reference(other); }
            /// comparison (less than or equal to)
            bool operator<=(const self_type& other) const noexcept
            { return const_reference(*this) <= const_reference(other); }
            /// comparison (greater than or equal to)
            bool operator>=(const self_type& other) const noexcept
            { return const_reference(*this) >= const_reference(other); }

            /// return pointer to element pointed to be this proxy
            pointer operator&() noexcept;
            /// return const pointer to element pointed to be this proxy
            const_pointer operator&() const noexcept;
    };

    /// comparison (equality)
    template <typename T>
    bool operator==(const typename ObjectProxy<T>::value_type& a,
            const ObjectProxy<T>& b) noexcept { return b == a; }
    /// comparison (inequality)
    template <typename T>
    bool operator!=(const typename ObjectProxy<T>::value_type& a,
            const ObjectProxy<T>& b) noexcept { return b != a; }
    /// comparison (less than)
    template <typename T>
    bool operator<(const typename ObjectProxy<T>::value_type& a,
            const ObjectProxy<T>& b) noexcept { return b > a; }
    /// comparison (greater than)
    template <typename T>
    bool operator>(const typename ObjectProxy<T>::value_type& a,
            const ObjectProxy<T>& b) noexcept { return b < a; }
    /// comparison (less than or equal to)
    template <typename T>
    bool operator<=(const typename ObjectProxy<T>::value_type& a,
            const ObjectProxy<T>& b) noexcept { return b >= a; }
    /// comparison (greater than or equal to)
    template <typename T>
    bool operator>=(const typename ObjectProxy<T>::value_type& a,
            const ObjectProxy<T>& b) noexcept { return b <= a; }

    /// helper for std::swap for ObjectProxy<T>
    template <typename T>
    void swap(ObjectProxy<T> a, ObjectProxy<T> b) noexcept(
            noexcept(a.swap(b)))
    { a.swap(b); }
} // namespace SOA

#include "SOAIterator.h"

namespace SOA {
    template <typename PARENTCONTAINER>
    typename ObjectProxy<PARENTCONTAINER>::pointer
    ObjectProxy<PARENTCONTAINER>::operator&() noexcept
    {
        return pointer{ m_storage, m_index,
            typename parent_type::its_safe_tag() };
    }

    template <typename PARENTCONTAINER>
    typename ObjectProxy<PARENTCONTAINER>::const_pointer
    ObjectProxy<PARENTCONTAINER>::operator&() const noexcept
    {
        return const_pointer{ m_storage, m_index,
            typename parent_type::its_safe_tag() };
    }
} // namespace SOA

#endif // SOAOBJECTPROXY_H

// vim: sw=4:tw=78:ft=cpp:et
