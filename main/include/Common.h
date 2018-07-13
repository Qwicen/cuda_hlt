#pragma once

#include <vector>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include "SystemOfUnits.h"

// MC Check
#if defined(MC_CHECK)
constexpr bool mc_check_enabled = true;
#else
constexpr bool mc_check_enabled = false;
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
