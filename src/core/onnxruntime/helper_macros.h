/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-14 18:55:58
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-04-13 21:29:41
 **************************************************************/

#ifndef HELPER_MACROS_H  // NOLINT
#define HELPER_MACROS_H

#include <iostream>
#include <map>
#include <stdio.h>
#include <sys/time.h>

#define APP_DEBUG

namespace cuda_ops {

#define APP_BOOL(x)                                                                                                    \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::cout << "Assertion `" #x "` failed";                                                                  \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define APP_CHECK(x)                                                                                                   \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::cout << "Assertion `" #x "` failed";                                                                  \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define APP_CHECK_V2(x, val)                                                                                           \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::cout << "Assertion `" #x "` failed, " << val;                                                         \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define APP_ASSERT(x)                                                                                                  \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::cout << "Assertion `" #x "` failed";                                                                  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define APP_PRINTF(format, ...)                                                                                        \
    do {                                                                                                               \
        struct timeval tv;                                                                                             \
        gettimeofday(&tv, nullptr);                                                                                    \
        auto now_micros = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;                                     \
        auto now_seconds = static_cast<time_t>(now_micros / 1000000);                                                  \
        auto micros_remainder = static_cast<int32_t>(now_micros % 1000000);                                            \
        const size_t kTimeBufferSize{30};                                                                              \
        char time_buffer[kTimeBufferSize];                                                                             \
        strftime(time_buffer, kTimeBufferSize, "%Y-%m-%d %H:%M:%S", localtime(&now_seconds));                          \
        char file_ch[256];                                                                                             \
        sprintf(file_ch, "%s", __FILE__);                                                                              \
        std::string file_str(file_ch);                                                                                 \
        auto pos = file_str.rfind("/");                                                                                \
        auto file_str_short = file_str.substr(pos + 1);                                                                \
        fprintf(stderr,                                                                                                \
                "%s.%06d: %s %s:%d] " format,                                                                          \
                time_buffer,                                                                                           \
                micros_remainder,                                                                                      \
                "I",                                                                                                   \
                file_str_short.c_str(),                                                                                \
                __LINE__,                                                                                              \
                ##__VA_ARGS__);                                                                                        \
    } while (0)

}  // namespace cuda_ops

#endif  // HELPER_MACROS_H
