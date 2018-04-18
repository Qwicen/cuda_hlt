/** @file LHCbID.h
 *
 * @brief encapsulate an LHCbID
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 */

#pragma once

#include <array>
#include <cstdint>

#include "boost/range/iterator_range.hpp"

/// encapsulate an LHCbID
class LHCbID {
    private:
        uint32_t m_id;

    public:
        constexpr LHCbID(uint32_t id) : m_id(id) {}

        LHCbID() = default;
        LHCbID(const LHCbID& other) = default;
        LHCbID(LHCbID&& other) = default;
        LHCbID& operator=(const LHCbID& other) = default;
        LHCbID& operator=(LHCbID&& other) = default;

        /// convert back to integer
        constexpr operator uint32_t() const noexcept { return m_id; }
        /// ordering of LHCbIDs
        constexpr bool operator==(const LHCbID& other) const noexcept
        { return m_id == other.m_id; }
        /// ordering of LHCbIDs
        constexpr bool operator!=(const LHCbID& other) const noexcept
        { return m_id != other.m_id; }
        /// ordering of LHCbIDs
        constexpr bool operator<(const LHCbID& other) const noexcept
        { return m_id < other.m_id; }
        /// ordering of LHCbIDs
        constexpr bool operator<=(const LHCbID& other) const noexcept
        { return m_id <= other.m_id; }
        /// ordering of LHCbIDs
        constexpr bool operator>(const LHCbID& other) const noexcept
        { return m_id > other.m_id; }
        /// ordering of LHCbIDs
        constexpr bool operator>=(const LHCbID& other) const noexcept
        { return m_id >= other.m_id; }

        // FIXME: ultimately, more methods are needed to e.g. get Velo sensor
        // numbers for hits etc.
};

/// how many LHCbIDs for small collections
enum { maxLHCbIDs = 128 };
/// a small collection of LHCbIDs
class SomeLHCbIDs : public std::array<LHCbID, maxLHCbIDs>
{
    public:
        SomeLHCbIDs() = default;
        SomeLHCbIDs(const SomeLHCbIDs&) = default;
        SomeLHCbIDs(SomeLHCbIDs&&) = default;
        SomeLHCbIDs& operator=(const SomeLHCbIDs&) = default;
        SomeLHCbIDs& operator=(SomeLHCbIDs&&) = default;
        template <typename IT>
        SomeLHCbIDs(IT first, IT last)
        {
            assert(maxLHCbIDs >= std::distance(first, last));
            for (auto it = this->begin(); last != first; ++first, ++it)
                *it = *first;
        }
};
/// a range of LHCbIDs from a small collection
using SomeLHCbIDRange = boost::iterator_range<SomeLHCbIDs::iterator>;
/// a const range of LHCbIDs from a small collection
using ConstSomeLHCbIDRange = boost::iterator_range<SomeLHCbIDs::const_iterator>;

// vim: sw=4:tw=78:ft=cpp:et
