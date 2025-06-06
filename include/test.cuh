#ifndef TEST_H
#define TEST_H

#define TEST_FUNC(func) \
    std::cout << #func << "\t"; \
    if(func()) \
        std::cout << "PASSED" << std::endl; \
    else \
        std::cout << "FAILED" << std::endl; \


void test();

#endif