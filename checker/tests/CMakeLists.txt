add_executable(velopix-input-reader velopix-input-reader.cc ${headers})
add_executable(velo-phi-drdz2 velo-phi-drdz2.cc ${headers})
target_link_libraries(velopix-input-reader libvelophidrdz2)
target_link_libraries(velo-phi-drdz2 libvelophidrdz2)
