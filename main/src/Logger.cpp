#include "Logger.h"

namespace logger {
  Logger ll;
}

std::ostream& logger::logger(int requestedLogLevel)
{
  if (logger::ll.verbosityLevel >= requestedLogLevel) {
    return std::cout;
  }
  else {
    return logger::ll.discardStream;
  }
}
