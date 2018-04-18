/** @file AlignedAllocator.h
 *
 * @brief allocator class that allows classes like std::vector to make
 * guarantees about memory alignment.
 *
 * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
 * @date 2017-03-28
 */

#ifndef ALIGNEDALLOCATOR_H
#define ALIGNEDALLOCATOR_H

/** @brief aligned allocator
 *
 * @author Manuel Schiller <Manuel.Schiller@glasgow.ac.uk>
 * @date 2017-03-28
 *
 * This class allows for allocation of aligned memory for classes like
 * std::vector. It is used this way:
 *
 * @code
 * #include <vector>
 * #include "AlignedAllocator.h"
 * template <typename T> using aligned_vector =
 *     std::vector<T, AlignedAllocator<T, 64> >;
 * aligned_vector<int> v;
 * v.push_back(42); // more operations
 * @endcode
 *
 * The class has the following template parameters:
 *
 * @tparam T            type for which to allocate memory
 * @tparam ALIGN        alignment in bytes
 *
 * ALIGN must be a power of two between 1 and 128 bytes. (Larger alignment
 * requirements are better satisfied with a more complex allocator.)
 */
template <typename T, std::size_t ALIGN>
class AlignedAllocator {
    private:
        // a few sanity checks...
        static_assert(ALIGN > 0, "ALIGN must be positive.");
        static_assert(ALIGN <= 128, "ALIGN must be 128 or smaller.");
        static_assert(0 == (ALIGN & (ALIGN - 1)),
                "ALIGN must be a power of 2.");
        static_assert(ALIGN >= ((sizeof(T) < 16) ? sizeof(T) : 16),
                "ALIGN not suitable for type T");

        std::allocator<char> alloc; ///< underlying allocator

    public:
        /// constructor
        AlignedAllocator() {}
        /// copy constructor
        AlignedAllocator(const AlignedAllocator<T, ALIGN>&) {}
        /// rebinding constructor
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, ALIGN>&) {}

        /// type being allocated
        using value_type = T;
        /// type of pointer
        using pointer = T*;
        /// type of const pointer
        using const_pointer = const T*;
        /// type of reference
        using reference = T&;
        /// type of const_reference
        using const_reference = const T&;
        /// type used for sizes
        using size_type = std::size_t;
        /// type used for (pointer) differences
        using difference_type = std::ptrdiff_t;
        /// whether to propagate allocator on move assignment of containers
        using propagate_on_container_move_assignment = std::true_type;
        /// allocator type for a different underlying type -- U instead of T
        template <typename U>
        struct rebind { using other = AlignedAllocator<U, ALIGN>; };
        // C++17
        // using is_always_equal = typename std::allocator<T>::is_always_equal;

        /// allocate (aligned) space for n objects of type T
        pointer __attribute__((aligned(ALIGN))) allocate(
                size_type n, const void* hint = nullptr)
        {
            char* p = alloc.allocate(n * sizeof(T) + ALIGN, hint);
            p += ALIGN;
            auto adj = std::ptrdiff_t(p) & (ALIGN - 1);
            p -= adj;
            *(p - 1) = adj;
            return (T*) p;
        }

        /// deallocate allocated space
        void deallocate(pointer p, std::size_t n)
        {
            auto adj = *(((unsigned char*) p) - 1);
            char* q = ((char*) p) + adj - ALIGN;
            alloc.deallocate(q, n * sizeof(T) + ALIGN);
        }

        /// maximum size representable by size_type
        constexpr size_type max_size() const noexcept
        { return std::numeric_limits<size_type>::max(); }

        /// construct a T at memory location p
        void construct(pointer p, const_reference val) { new(p) T(val); }
        /// construct a U at memory location p
        template <typename U, typename... ARGS>
        void construct(U* p, ARGS&&... args) { new(p) U(std::forward<ARGS>(args)...); }
        /// destroy a T at memory location p
        void destroy(pointer p) { p->~T(); }
        /// destroy a U at memory location p
        template <typename U>
        void destroy(U* p) { p->~U(); }

        /// return the address of what reference references
        pointer address(reference r) const noexcept { return &r; }
        /// return the address of what const reference references
        const_pointer address(const_reference r) const noexcept { return &r; }
};

/// convenience typedef for 64 byte alignment
template <typename T>
using CacheLineAlignedAllocator = AlignedAllocator<T, 64>;

#endif // ALIGNEDALLOCATOR_H

// vim: sw=4:tw=78:ft=cpp:et
