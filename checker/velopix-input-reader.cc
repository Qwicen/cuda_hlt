/** @file velopix-input-reader.cc
 *
 * @brief unit test: reader of velopix input files
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-08
 */

#include "velopix-input-reader.h"

static void test() {
    std::vector<VelopixEvent> events = VelopixEventReader::readFolder("../input");
    for (auto& event : events) {
        event.print();
    }
}

int main()
{
    test();
    return 0;
}
