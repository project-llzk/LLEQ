#pragma once

#include <cstddef>
#include <memory>

template <std::size_t N> class Arena {
  std::byte *buffer = new std::byte[N];
  void *current = buffer;
  std::size_t capacity = N;
  Arena *next = nullptr;

public:
  template <class T, class... Args> T *alloc(Args... args) {
    std::size_t align = alignof(T);

    if (std::align(align, sizeof(T), current, capacity)) {
      T *result = std::launder(reinterpret_cast<T *>(current));
      new (result) T(args...);

      current = static_cast<std::byte *>(current) + sizeof(T);
      capacity -= sizeof(T);
      return result;
    }

    next = new Arena<N>;
    return next->alloc<T>(args...);
  }

  ~Arena() { ::operator delete[](buffer); }
};
