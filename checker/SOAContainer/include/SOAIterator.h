/** @file SOAIterator.h
 *
 * @author Manuel Schiller <Manuel.Schiller@cern.ch>
 * @date 2015-10-02
 */

#ifndef SOAITERATOR_H
#define SOAITERATOR_H

#include <tuple>
#include <iterator>
#include <ostream>

namespace SOA {
    // forward decls
    template <typename PROXY>
    class ConstIterator;
    template<typename T>
    std::ostream& operator<<(std::ostream&, const ConstIterator<T>&);
} // namespace SOA
template < template <typename...> class CONTAINER,
    template <typename> class SKIN, typename... FIELDS>
class _Container;

/// namespace to encapsulate SOA stuff
namespace SOA {
    /** @brief class mimicking a const pointer to pointee inidicated by PROXY
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-05-03
     *
     * @tparam PROXY        proxy class
     */
    template <typename PROXY>
    class ConstIterator
    {
        protected:
            /// parent type (underlying container)
            using parent_type = typename PROXY::parent_type;
            // parent container is a friend
            friend parent_type;
            /// corresponding _Containers are friends
            template < template <typename...> class CONTAINER,
                     template <typename> class SKIN, typename... FIELDS>
            friend class _Container;
            /// parent's proxy type
            using proxy = PROXY;
            // underlying "dressed" proxy is friend as well
            friend proxy;
            /// parent's naked proxy type
            using naked_proxy = typename parent_type::naked_proxy;
            // underlying "naked" proxy is friend as well
            friend naked_proxy;

            proxy m_proxy; ///< pointee


        public:
            /// convenience using = for our own type
            using self_type = ConstIterator<proxy>;
            /// import value_type from proxy
            using value_type = typename proxy::value_type;
            /// import size_type from proxy
            using size_type = typename proxy::size_type;
            /// import difference_type from proxy
            using difference_type = typename proxy::difference_type;
            /// using = for reference to pointee
            using reference = const proxy;
            /// using = for const reference to pointee
            using const_reference = const proxy;
            /// using = for pointer
            using pointer = typename proxy::pointer;
            /// using = for const pointer
            using const_pointer = typename proxy::const_pointer;
            /// iterator category
            using iterator_category = std::random_access_iterator_tag;

        protected:
            /// constructor building proxy in place
            explicit ConstIterator(typename proxy::SOAStorage* storage,
                    size_type index, typename proxy::parent_type::its_safe_tag
                    safe) noexcept : m_proxy(storage, index, safe) { }

        public:
            /// default constructor (nullptr equivalent)
            ConstIterator() noexcept : ConstIterator(nullptr, 0) { }

            /// copy constructor
            ConstIterator(const self_type& other) noexcept = default;
            /// move constructor
            ConstIterator(self_type&& other) noexcept = default;

            /// assignment
            self_type& operator=(const self_type& other) noexcept
            { m_proxy.assign(other.m_proxy); return *this; }
            /// assignment (move semantics)
            self_type& operator=(self_type&& other) noexcept
            { m_proxy.assign(std::move(other.m_proxy)); return *this; }

            /// deference pointer (*p)
            reference operator*() noexcept
            { return m_proxy; }
            /// deference pointer (*p)
            const_reference operator*() const noexcept
            { return m_proxy; }
            /// deference pointer (p->blah)
            reference* operator->() noexcept
            { return std::addressof(m_proxy); }
            /// deference pointer (p->blah)
            const_reference* operator->() const noexcept
            { return std::addressof(m_proxy); }

            /// (pre-)increment
            self_type& operator++() noexcept
            { ++m_proxy.m_index; return *this; }
            /// (pre-)decrement
            self_type& operator--() noexcept
            { --m_proxy.m_index; return *this; }
            /// (post-)increment
            self_type operator++(int) noexcept
            { self_type retVal(*this); ++m_proxy.m_index; return retVal; }
            /// (post-)decrement
            self_type operator--(int) noexcept
            { self_type retVal(*this); --m_proxy.m_index; return retVal; }
            /// advance by dist elements
            self_type& operator+=(difference_type dist) noexcept
            { m_proxy.m_index += dist; return *this; }
            /// "retreat" by dist elements
            self_type& operator-=(difference_type dist) noexcept
            { m_proxy.m_index -= dist; return *this; }
            /// advance by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type&
                    >::type operator+=(T dist) noexcept
            { m_proxy.m_index += dist; return *this; }
            /// "retreat" by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type&
                    >::type operator-=(T dist) noexcept
            { m_proxy.m_index -= dist; return *this; }
            /// advance by dist elements
            self_type operator+(difference_type dist) const noexcept
            { return self_type(*this) += dist; }
            /// "retreat" by dist elements
            self_type operator-(difference_type dist) const noexcept
            { return self_type(*this) -= dist; }
            /// advance by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type
                    >::type operator+(T dist) const noexcept
            { return self_type(*this) += dist; }
            /// "retreat" by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type
                    >::type operator-(T dist) const noexcept
            { return self_type(*this) -= dist; }
            /// distance between two pointers
            difference_type operator-(const self_type& other) const noexcept
            {
                // give warning about buggy code subtracting pointers from
                // different containers (ill-defined operation on this pointer
                // class), use plain C style assert here
                assert(m_proxy.m_storage &&
                        m_proxy.m_storage == other.m_proxy.m_storage);
#if !defined(BREAKACTIVELY) && !defined(NDEBUG)
                return (m_proxy.m_index - other.m_proxy.m_index);
#else
                // return distance if pointers from same container, else return
                // ridiculously large value in the hopes of badly breaking
                // ill-behaved client code (when asserts are disabled)
                return (m_proxy.m_storage &&
                        m_proxy.m_storage == other.m_proxy.m_storage) ?
                    (m_proxy.m_index - other.m_proxy.m_index) :
                    std::numeric_limits<difference_type>::max();
#endif
            }

            /// indexing
            reference operator[](size_type idx) noexcept
            { return { m_proxy.m_storage, m_proxy.m_index + idx }; }
            /// indexing
            const_reference operator[](size_type idx) const noexcept
            { return { m_proxy.m_storage, m_proxy.m_index + idx }; }

            /// comparison (equality)
            bool operator==(const self_type& other) const noexcept
            {
                return m_proxy.m_index == other.m_proxy.m_index &&
                    m_proxy.m_storage == other.m_proxy.m_storage;
            }
            /// comparison (inequality)
            bool operator!=(const self_type& other) const noexcept
            {
                return m_proxy.m_index != other.m_proxy.m_index ||
                    m_proxy.m_storage != other.m_proxy.m_storage;
            }
            /// comparison (less than)
            bool operator<(const self_type& other) const noexcept
            {
                return m_proxy.m_storage < other.m_proxy.m_storage ? true :
                    other.m_proxy.m_storage < m_proxy.m_storage ? false :
                    m_proxy.m_index < other.m_proxy.m_index;
            }
            /// comparison (greater than)
            bool operator>(const self_type& other) const noexcept
            {
                return m_proxy.m_storage < other.m_proxy.m_storage ? false :
                    other.m_proxy.m_storage < m_proxy.m_storage ? true :
                    other.m_proxy.m_index < m_proxy.m_index;
            }
            /// comparison (less than or equal to)
            bool operator<=(const self_type& other) const noexcept
            {
                return m_proxy.m_storage < other.m_proxy.m_storage ? true :
                    other.m_proxy.m_storage < m_proxy.m_storage ? false :
                    m_proxy.m_index <= other.m_proxy.m_index;
            }
            /// comparison (greater than or equal to)
            bool operator>=(const self_type& other) const noexcept
            {
                return m_proxy.m_storage < other.m_proxy.m_storage ? false :
                    other.m_proxy.m_storage < m_proxy.m_storage ? true :
                    other.m_proxy.m_index <= m_proxy.m_index;
            }
            /// check for validity (if (ptr) or if (!ptr) idiom)
            operator bool() const noexcept
            {
                return m_proxy.m_storage &&
                    m_proxy.m_index < std::get<0>(*m_proxy.m_storage).size();
            }

        protected:
            /// give access to underlying storage pointer
            auto storage() const noexcept -> decltype(&*m_proxy.m_storage)
            { return &*m_proxy.m_storage; }
            /// give access to index into storage
            auto index() const noexcept -> decltype(m_proxy.m_index)
            { return m_proxy.m_index; }
            /// make operator<< friend to allow calling storage() and index()
            template <typename T>
            friend std::ostream& operator<<(std::ostream&, const ConstIterator<T>&);
    };

    /** @brief class mimicking a pointer to pointee inidicated by PROXY
     *
     * @author Manuel Schiller <Manuel.Schiller@cern.ch>
     * @date 2015-05-03
     *
     * @tparam PROXY        proxy class
     */
    template <typename PROXY>
    class Iterator : public ConstIterator<PROXY>
    {
        private:
            /// parent type (underlying container)
            using parent_type = typename PROXY::parent_type;
            // parent container is a friend
            friend parent_type;
            /// parent's proxy type
            using proxy = PROXY;
            // underlying "dressed" proxy is friend as well
            friend proxy;
            /// corresponding _Containers are friends
            template < template <typename...> class CONTAINER,
                     template <typename> class SKIN, typename... FIELDS>
            friend class _Container;
            /// parent's naked proxy type
            using naked_proxy = typename parent_type::naked_proxy;
            // underlying "naked" proxy is friend as well
            friend naked_proxy;

        public:
            /// convenience using = for our own type
            using self_type = Iterator<proxy>;
            /// import value_type from proxy
            using value_type = typename proxy::value_type;
            /// import size_type from proxy
            using size_type = typename proxy::size_type;
            /// import difference_type from proxy
            using difference_type = typename proxy::difference_type;
            /// using = for reference to pointee
            using reference = proxy;
            /// using = for const reference to pointee
            using const_reference = const proxy;
            /// using = for pointer
            using pointer = Iterator<proxy>;
            /// using = for const pointer
            using const_pointer = ConstIterator<proxy>;
            /// iterator category
            using iterator_category = std::random_access_iterator_tag;

        private:
            /// constructor building proxy in place
            explicit Iterator(typename proxy::parent_type::SOAStorage* storage,
                    size_type index, typename proxy::parent_type::its_safe_tag safe) noexcept :
                ConstIterator<proxy>(storage, index, safe) { }

        public:
            /// default constructor (nullptr equivalent)
            Iterator() noexcept : Iterator(nullptr, 0) { }

            /// copy constructor
            Iterator(const self_type& other) noexcept = default;
            /// move constructor
            Iterator(self_type&& other) noexcept = default;

            /// assignment
            self_type& operator=(const self_type& other) noexcept
            { ConstIterator<proxy>::operator=(other); return *this; }
            /// assignment (move semantics)
            self_type& operator=(self_type&& other) noexcept
            { ConstIterator<proxy>::operator=(std::move(other)); return *this; }

            /// deference pointer (*p)
            reference operator*() noexcept
            { return ConstIterator<proxy>::m_proxy; }
            /// deference pointer (*p)
            const_reference operator*() const noexcept
            { return ConstIterator<proxy>::m_proxy; }
            /// deference pointer (p->blah)
            reference* operator->() noexcept
            { return std::addressof(ConstIterator<proxy>::m_proxy); }
            /// deference pointer (p->blah)
            const_reference* operator->() const noexcept
            { return std::addressof(ConstIterator<proxy>::m_proxy); }

            /// (pre-)increment
            self_type& operator++() noexcept
            { ConstIterator<proxy>::operator++(); return *this; }
            /// (pre-)decrement
            self_type& operator--() noexcept
            { ConstIterator<proxy>::operator--(); return *this; }
            /// (post-)increment
            self_type operator++(int) noexcept
            { self_type retVal(*this); operator++(); return retVal; }
            /// (post-)decrement
            self_type operator--(int) noexcept
            { self_type retVal(*this); operator--(); return retVal; }
            /// advance by dist elements
            self_type& operator+=(difference_type dist) noexcept
            { ConstIterator<proxy>::operator+=(dist); return *this; }
            /// "retreat" by dist elements
            self_type& operator-=(difference_type dist) noexcept
            { ConstIterator<proxy>::operator-=(dist); return *this; }
            /// advance by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type
                    >::type operator+=(T dist) noexcept
            { ConstIterator<proxy>::operator+=(dist); return *this; }
            /// "retreat" by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type
                    >::type operator-=(T dist) noexcept
            { ConstIterator<proxy>::operator-=(dist); return *this; }
            /// advance by dist elements
            self_type operator+(difference_type dist) const noexcept
            { return self_type(*this) += dist; }
            /// "retreat" by dist elements
            self_type operator-(difference_type dist) const noexcept
            { return self_type(*this) -= dist; }
            /// advance by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type
                    >::type operator+(T dist) const noexcept
            { return self_type(*this) += dist; }
            /// "retreat" by dist elements
            template <typename T>
            typename std::enable_if<
                    std::is_integral<T>::value &&
                    std::is_convertible<T, difference_type>::value, self_type
                    >::type operator-(T dist) const noexcept
            { return self_type(*this) -= dist; }
            /// return distance between two pointers
            difference_type operator-(
                    const ConstIterator<proxy>& other) const noexcept
            { return ConstIterator<proxy>::operator-(other); }

            /// indexing
            reference operator[](size_type idx) noexcept
            { return { ConstIterator<proxy>::m_proxy.m_storage,
                         Iterator<proxy>::m_proxy.m_index + idx }; }
            /// indexing
            const_reference operator[](size_type idx) const noexcept
            { return { ConstIterator<proxy>::m_proxy.m_storage,
                         Iterator<proxy>::m_proxy.m_index + idx }; }
    };

    /// implement integer + Iterator
    template <typename PROXY, typename T>
    typename std::enable_if<
        std::is_integral<T>::value && std::is_convertible<T,
            typename Iterator<PROXY>::difference_type>::value,
        Iterator<PROXY> >::type
        operator+(T dist, const Iterator<PROXY>& other) noexcept
    { return other + dist; }

    /// implement integer + ConstIterator
    template <typename PROXY, typename T>
    typename std::enable_if<
        std::is_integral<T>::value && std::is_convertible<T,
            typename ConstIterator<PROXY>::difference_type>::value,
        ConstIterator<PROXY> >::type
        operator+(T dist, const ConstIterator<PROXY>& other) noexcept
    { return other + dist; }

    /// operator<< for supporting idioms like "std::cout << it" (mostly debugging)
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const ConstIterator<T>& it) {
        os << "(" << it.storage() << ", " << it.index() << ")";
        return os;
    }
} // namespace SOA

#endif // SOAITERATOR_H

// vim: sw=4:tw=78:ft=cpp:et
