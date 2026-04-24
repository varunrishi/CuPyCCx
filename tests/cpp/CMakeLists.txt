find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_cupyccx
    test_ccd.cpp
    test_integrals.cpp
)

target_link_libraries(test_cupyccx PRIVATE
    cupyccx_core
    GTest::gtest_main
)

target_include_directories(test_cupyccx PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(test_cupyccx)
