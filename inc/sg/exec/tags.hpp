#pragma once

// sg::exec: Tags and core customization points.

#include <utility>

namespace sg::exec {

struct sender_t {};
struct receiver_t {};
struct operation_state_t {};
struct scheduler_t {};

// The completion tags name the channels in completion signatures. Calling one
// calls the receiver member of the same name.
struct set_value_t {
  template <class R, class... Vs>
  constexpr void operator()(R &&r, Vs &&...vs) const noexcept {
    static_assert(
      noexcept(std::forward<R>(r).set_value(std::forward<Vs>(vs)...)),
      "set_value must be noexcept");
    std::forward<R>(r).set_value(std::forward<Vs>(vs)...);
  }
};
struct set_error_t {
  template <class R, class E>
  constexpr void operator()(R &&r, E &&e) const noexcept {
    static_assert(noexcept(std::forward<R>(r).set_error(std::forward<E>(e))),
      "set_error must be noexcept");
    std::forward<R>(r).set_error(std::forward<E>(e));
  }
};
struct set_stopped_t {
  template <class R> constexpr void operator()(R &&r) const noexcept {
    static_assert(noexcept(std::forward<R>(r).set_stopped()),
      "set_stopped must be noexcept");
    std::forward<R>(r).set_stopped();
  }
};
inline constexpr set_value_t set_value{};
inline constexpr set_error_t set_error{};
inline constexpr set_stopped_t set_stopped{};

struct start_t {
  template <class O> constexpr void operator()(O &op) const noexcept {
    static_assert(noexcept(op.start()), "start must be noexcept");
    op.start();
  }
};
inline constexpr start_t start{};

struct schedule_t {
  template <class Sch>
  constexpr auto operator()(Sch &&sch) const
    noexcept(noexcept(std::forward<Sch>(sch).schedule()))
      -> decltype(std::forward<Sch>(sch).schedule()) {
    return std::forward<Sch>(sch).schedule();
  }
};
inline constexpr schedule_t schedule{};

} // namespace sg::exec
