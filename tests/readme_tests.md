Catch2 tests
============
How to add a new test
------------
[Catch2 documentation](https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md)  
In order to add a new test called `TestName`, it is necessary to put `TestName` into `UNIT_TEST_LIST` in [tests/CMakeLists.txt](CMakeLists.txt).
```
set(UNIT_TEST_LIST
  MuonFeaturesExtraction
  ...
  TestName)
```
Create a new file inside the `tests` directory named `TestName.test.cu`.  
Then add the following include to a file:
```cpp
#include "catch.hpp"
```
Now we are ready to add test to the project.

How to build it
------------
```
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
make
```

How to run it
------------
```
ctest { runs all the tests }
ctest -R test_name { runs specific test }
./tests/test_name.test { runs specific test and shows failed assertions }
./tests/test_name.test -s { runs specific test and shows all assertions }
```