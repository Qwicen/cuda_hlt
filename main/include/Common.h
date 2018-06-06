#pragma once

#include <vector>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

// MC Check
#ifdef MC_CHECK
constexpr bool do_mc_check = true;
#else
constexpr bool do_mc_check = false;
#endif

/**
 * Generic StrException launcher
 */
class StrException : public std::exception
{
public:
    std::string s;
    StrException(std::string ss) : s(ss) {}
    ~StrException() throw () {} // Updated
    const char* what() const throw() { return s.c_str(); }
};
