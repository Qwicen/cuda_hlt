#pragma once

#include <vector>
#include <set>
#include <type_traits>
#include <iostream>
#include <cstdint>
#include <cstdio>

#include "SystemOfUnits.h"

/**
 * Generic StrException launcher
 */
struct StrException : public std::exception {
  std::string s;
  StrException(std::string ss) : s(ss) {}
  ~StrException() throw() {} // Updated
  const char* what() const throw() { return s.c_str(); }
};
