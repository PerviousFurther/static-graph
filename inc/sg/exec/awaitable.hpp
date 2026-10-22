#pragma once

// sg::exec: Awaitables and connect.

#include "env.hpp"

#include <coroutine>
#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
// The operand of co_await after applying operator co_await, if any.
template <class A> decltype(auto) co_await_operand(A &&a) {
  if constexpr (requires { std::forward<A>(a).operator co_await(); })
    return std::forward<A>(a).operator co_await();
  else if constexpr (requires { operator co_await(std::forward<A>(a)); })
    return operator co_await(std::forward<A>(a));
  else
    return std::forward<A>(a);
}
// The awaiter of co_await a in a coroutine whose promise is P.
template <class A, class P> decltype(auto) get_awaiter(A &&a, P *p) {
  if constexpr (requires { p->await_transform(std::forward<A>(a)); })
    return co_await_operand(p->await_transform(std::forward<A>(a)));
  else
    return co_await_operand(std::forward<A>(a));
}

template <class W, class P>
concept awaiter = requires(W &w, std::coroutine_handle<P> h) {
  w.await_ready() ? 0 : 0;
  w.await_suspend(h);
  w.await_resume();
};
template <class A, class P>
concept is_awaitable = requires(A &&a, P *p) {
  { get_awaiter(std::forward<A>(a), p) } -> awaiter<P>;
};
template <class A, class P>
using await_result_t =
  decltype(std::declval<decltype(get_awaiter(
             std::declval<A>(), static_cast<P *>(nullptr)))>()
             .await_resume());

// A promise without await_transform, for checking plain awaitables.
struct plain_promise {};

// Awaits what defines as_awaitable(promise) through it, the rest as is.
template <class Promise> struct with_await_transform {
  template <class A> decltype(auto) await_transform(A &&a) {
    if constexpr (requires {
                    std::forward<A>(a).as_awaitable(
                      static_cast<Promise &>(*this));
                  })
      return std::forward<A>(a).as_awaitable(static_cast<Promise &>(*this));
    else
      return std::forward<A>(a);
  }
};

// The promise an awaitable is checked against when it is used as a sender in
// environment Env. Only used unevaluated.
template <class Env>
struct env_promise : with_await_transform<env_promise<Env>> {
  const Env &get_env() const noexcept;
  std::coroutine_handle<> unhandled_stopped() noexcept;
};

template <class R> class awaitable_operation;
// Connects an awaitable to r through a coroutine that awaits it and completes
// r with the result. Defined with the coroutine support at the end.
template <class A, class R> awaitable_operation<R> connect_awaitable(A a, R r);
} // namespace detail

struct connect_t {
  template <class S, class R>
  constexpr auto operator()(S &&s, R &&r) const
    noexcept(noexcept(std::forward<S>(s).connect(std::forward<R>(
      r)))) -> decltype(std::forward<S>(s).connect(std::forward<R>(r))) {
    return std::forward<S>(s).connect(std::forward<R>(r));
  }
  // An awaitable without connect: the operation is a coroutine awaiting it.
  template <class S, class R>
    requires(!requires(S &&s, R &&r) {
      std::forward<S>(s).connect(std::forward<R>(r));
    }) &&
    detail::is_awaitable<S, detail::env_promise<std::decay_t<env_of_t<R>>>>
  auto operator()(S &&s, R &&r) const {
    return detail::connect_awaitable(std::remove_cvref_t<S>(std::forward<S>(s)),
      std::remove_cvref_t<R>(std::forward<R>(r)));
  }
};
inline constexpr connect_t connect{};

} // namespace sg::exec
