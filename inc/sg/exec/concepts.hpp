#pragma once

// sg::exec: Concepts.

#include "../core/graph.hpp"
#include "completion_signatures.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
template <class S>
concept has_sender_concept =
  std::derived_from<typename std::remove_cvref_t<S>::sender_concept, sender_t>;
} // namespace detail

// Awaitables are senders too: connect awaits them in a coroutine.
template <class S>
concept sender = (detail::has_sender_concept<S> ||
                   detail::is_awaitable<S, detail::env_promise<env<>>>) &&
  std::move_constructible<std::remove_cvref_t<S>>;

template <class S, class Env = env<>>
concept sender_in =
  sender<S> && requires { typename completion_signatures_of_t<S, Env>; };

template <class R>
concept receiver =
  std::derived_from<typename std::remove_cvref_t<R>::receiver_concept,
    receiver_t> &&
  std::move_constructible<std::remove_cvref_t<R>>;

template <class O>
concept operation_state =
  std::destructible<O> && std::is_object_v<O> && requires(O &o) {
    { o.start() } noexcept;
  };

template <class Sch>
concept scheduler =
  requires(Sch &&sch) {
    { schedule(std::forward<Sch>(sch)) } -> sender;
  } && std::equality_comparable<std::remove_cvref_t<Sch>> &&
  std::copy_constructible<std::remove_cvref_t<Sch>>;

template <class S, class R>
concept sender_to = sender_in<S, env_of_t<R>> && receiver<R> &&
  requires(S &&s, R &&r) { connect(std::forward<S>(s), std::forward<R>(r)); };

namespace detail {
template <class R, class Sig> inline constexpr bool accepts_v = false;
template <class R, class... As>
inline constexpr bool accepts_v<R, set_value_t(As...)> =
  requires(R &&r, As &&...as) {
    { std::forward<R>(r).set_value(std::forward<As>(as)...) } noexcept;
  };
template <class R, class E>
inline constexpr bool accepts_v<R, set_error_t(E)> = requires(R &&r, E &&e) {
  { std::forward<R>(r).set_error(std::forward<E>(e)) } noexcept;
};
template <class R>
inline constexpr bool accepts_v<R, set_stopped_t()> = requires(R &&r) {
  { std::forward<R>(r).set_stopped() } noexcept;
};
template <class R, class Sigs> inline constexpr bool accepts_all_v = false;
template <class R, class... Ss>
inline constexpr bool accepts_all_v<R, type_list<Ss...>> =
  (accepts_v<R, Ss> && ...);
} // namespace detail

template <class R, class Sigs>
concept receiver_of =
  receiver<R> && detail::accepts_all_v<std::remove_cvref_t<R>, Sigs>;

template <class T>
concept stoppable_token =
  std::copyable<T> && std::equality_comparable<T> && requires(const T &t) {
    { t.stop_requested() } noexcept -> std::same_as<bool>;
    { t.stop_possible() } noexcept -> std::same_as<bool>;
  };
template <class T>
concept unstoppable_token =
  stoppable_token<T> && requires { requires(!T::stop_possible()); };

template <class S, class R>
using connect_result_t =
  decltype(connect(std::declval<S>(), std::declval<R>()));
template <class Sch>
using schedule_result_t = decltype(schedule(std::declval<Sch>()));

} // namespace sg::exec
