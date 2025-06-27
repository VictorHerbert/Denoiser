#include "test.cuh"

#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>

std::vector<std::pair<std::string, TestStatus(*)()>> test_functions;

void test(){
    size_t max_size = 0;
    for(auto [str, func] : test_functions)
        max_size = std::max(max_size, str.size());

    std::cout << "Start Testing\n" << std::endl;
    for(auto [str, func] : test_functions){
        TestStatus status = func();
        std::cout << "Test " << std::left <<std::setfill('.') << std::setw(max_size+3) << str << to_string(status) << std::endl;
    }
    std::cout << "\nFinish Testing" << std::endl;
}

std::string to_string(TestStatus id){
    switch (id) {
        case TestStatus::SUCCESS: return "SUCCESS";
        case TestStatus::FAIL: return "FAIL";
        case TestStatus::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        default:          return "Unknown";
    }
}