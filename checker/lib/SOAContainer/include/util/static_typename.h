/** @file static_typename.h
 *
 * @brief compile-time type names
 */

// From http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
#ifndef STATIC_TYPENAME_H
#define STATIC_TYPENAME_H

#include "util/static_string.h"

// encapsulate functionality
namespace util {
    /// returns a compile-time string with the name of type T
    template <class T>
    CONSTEXPR14_TN static_string type_name()
    {
#ifdef __clang__
	static_string p = __PRETTY_FUNCTION__;
	return static_string(p.data() + 31, p.size() - 31 - 1);
#elif defined(__GNUC__)
	static_string p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
	return static_string(p.data() + 36, p.size() - 36 - 1);
#  else
	return static_string(p.data() + 46, p.size() - 46 - 1);
#  endif
#elif defined(_MSC_VER)
	static_string p = __FUNCSIG__;
	return static_string(p.data() + 38, p.size() - 38 - 7);
#endif
    }
} // namespace util

#endif // STATIC_TYPENAME_H

// vim: sw=4:tw=78:ft=cpp
