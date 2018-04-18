include_directories(include)
include_directories(SOAContainer/include)

add_subdirectory(SOAContainer)
add_subdirectory(src)
add_subdirectory(tests)


file(GLOB headers include/*)
