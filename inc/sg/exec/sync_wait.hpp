#pragma once

// sg::exec: sync_wait.

#include "../core/graph.hpp"
#include "into_variant.hpp"
#include "schedulers.hpp"

#include <exception>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
struct sync_wait_env {
  run_loop *loop;
  auto query(get_scheduler_t) const noexcept { return loop->get_scheduler(); }
  auto query(get_delegation_scheduler_t) const noexcept {
    return loop->get_scheduler();
  }
};

template <class Tuple> struct sync_wait_state {
  run_loop loop;
  std::optional<Tuple> values;
  std::exception_ptr error;
};

template <class E> std::exception_ptr as_exception_ptr(E &&e) noexcept {
  if constexpr (std::is_same_v<std::decay_t<E>, std::exception_ptr>)
    return std::forward<E>(e);
  else if constexpr (std::is_same_v<std::decay_t<E>, std::error_code>)
    return std::make_exception_ptr(std::system_error(e));
  else
    return std::make_exception_ptr(std::forward<E>(e));
}

template <class Tuple> struct sync_wait_receiver {
  using receiver_concept = receiver_t;
  sync_wait_state<Tuple> *state;
  template <class... As> void set_value(As &&...as) && noexcept {
    try {
      state->values.emplace(std::forward<As>(as)...);
    } catch (...) {
      state->error = std::current_exception();
    }
    state->loop.finish();
  }
  template <class E> void set_error(E &&e) && noexcept {
    state->error = as_exception_ptr(std::forward<E>(e));
    state->loop.finish();
  }
  void set_stopped() && noexcept { state->loop.finish(); }
  sync_wait_env get_env() const noexcept { return {&state->loop}; }
};
} // namespace detail

// Blocks until s completes, driving a local run_loop. Returns the values, or
// nullopt if s was stopped; rethrows an error.
struct sync_wait_t {
  template <sender_in<detail::sync_wait_env> S> auto operator()(S &&s) const {
    using values =
      value_types_of_t<S, detail::sync_wait_env, decayed_tuple, type_list>;
    static_assert(type_list_size<values>::value == 1,
      "sync_wait requires exactly one value completion");
    using tuple_type = typename type_list_at<0, values>::type;
    detail::sync_wait_state<tuple_type> state;
    auto op = exec::connect(
      std::forward<S>(s), detail::sync_wait_receiver<tuple_type>{&state});
    exec::start(op);
    state.loop.run();
    if (state.error)
      std::rethrow_exception(state.error);
    return std::move(state.values);
  }
};
struct sync_wait_with_variant_t {
  template <sender S> auto operator()(S &&s) const {
    return sync_wait_t{}(into_variant(std::forward<S>(s)));
  }
};
inline constexpr sync_wait_t sync_wait{};
inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};

namespace this_thread {
using exec::sync_wait;
using exec::sync_wait_with_variant;
} // namespace this_thread

} // namespace sg::exec
