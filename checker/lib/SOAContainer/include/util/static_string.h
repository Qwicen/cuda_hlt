/** @file static_string.h
 *
 * @brief compile-time strings
 */

// From http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
#ifndef STATIC_STRING_H
#define STATIC_STRING_H

#include <cstddef>
#include <stdexcept>
#include <cstring>
#include <ostream>

// portability
#undef CONSTEXPR11_TN
#undef CONSTEXPR14_TN
#undef NOEXCEPT_TN
#ifndef _MSC_VER
#  if __cplusplus < 201103
#    define CONSTEXPR11_TN
#    define CONSTEXPR14_TN
#    define NOEXCEPT_TN
#  elif __cplusplus < 201402
#    define CONSTEXPR11_TN constexpr
#    define CONSTEXPR14_TN
#    define NOEXCEPT_TN noexcept
#  else
#    define CONSTEXPR11_TN constexpr
#    define CONSTEXPR14_TN constexpr
#    define NOEXCEPT_TN noexcept
#  endif
#else  // _MSC_VER
#  if _MSC_VER < 1900
#    define CONSTEXPR11_TN
#    define CONSTEXPR14_TN
#    define NOEXCEPT_TN
#  elif _MSC_VER < 2000
#    define CONSTEXPR11_TN constexpr
#    define CONSTEXPR14_TN
#    define NOEXCEPT_TN noexcept
#  else
#    define CONSTEXPR11_TN constexpr
#    define CONSTEXPR14_TN constexpr
#    define NOEXCEPT_TN noexcept
#  endif
#endif  // _MSC_VER

// namespace encapsulating these utilities
namespace util {
    class static_string
    {
	const char* const p_;
	const std::size_t sz_;

	public:
	using const_iterator = const char*;

	template <std::size_t N>
	CONSTEXPR11_TN static_string(const char(&a)[N]) NOEXCEPT_TN :
	    p_(a), sz_(N-1)
	{}

	CONSTEXPR11_TN static_string(
		const char* p, std::size_t N) NOEXCEPT_TN :
	    p_(p), sz_(N)
	{}

	CONSTEXPR11_TN const char* data() const NOEXCEPT_TN
	{ return p_; }
	CONSTEXPR11_TN std::size_t size() const NOEXCEPT_TN
	{ return sz_; }

	CONSTEXPR11_TN const_iterator begin() const NOEXCEPT_TN
	{ return p_; }
	CONSTEXPR11_TN const_iterator end()   const NOEXCEPT_TN
	{ return p_ + sz_; }

	CONSTEXPR11_TN char operator[](std::size_t n) const
	{
	    return n < sz_ ? p_[n] :
		throw std::out_of_range("static_string");
	}
    };

    inline std::ostream& operator<<(
	    std::ostream& os, static_string const& s)
    { return os.write(s.data(), s.size()); }
} // namespace util

#undef CONSTEXPR11_TN
#undef CONSTEXPR14_TN
#undef NOEXCEPT_TN

#endif // STATIC_STRING_H

// vim: sw=4:tw=78:ft=cpp
