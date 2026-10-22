#pragma once

// sg::exec: Sources: just, just_error, just_stopped, read_env.

#include "layers.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
namespace desc {
template <class Tag, class... Vs> struct just {
  std::tuple<Vs...> values;
  static constexpr bool is_source = true;
  template <class, class> using completions = completion_signatures<Tag(Vs...)>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    std::tuple<Vs...> values;
    template <class... Rest>
    apply(just d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, values(std::move(d.values)) {}
    void start() noexcept {
      std::apply(
        [this](Vs &...vs) noexcept { this->pass(Tag{}, std::move(vs)...); },
        values);
    }
  };
};

template <class Q> struct read_env {
  [[no_unique_address]] Q query;
  static constexpr bool is_source = true;
  template <class, class Env>
  using completions = completion_signatures<set_value_t(
    std::decay_t<std::invoke_result_t<const Q &, const Env &>>)>;
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    [[no_unique_address]] Q query;
    template <class... Rest>
    apply(read_env d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, query(d.query) {}
    void start() noexcept {
      using value = std::decay_t<
        std::invoke_result_t<const Q &, decltype(this->downstream_env())>>;
      this->pass(set_value_t{}, value(query(this->downstream_env())));
    }
  };
};
} // namespace desc
} // namespace detail

struct just_t {
  template <class... Vs> auto operator()(Vs &&...vs) const {
    return basic_sender{detail::desc::just<set_value_t, std::decay_t<Vs>...>{
      {std::forward<Vs>(vs)...}}};
  }
};
struct just_error_t {
  template <class E> auto operator()(E &&e) const {
    return basic_sender{
      detail::desc::just<set_error_t, std::decay_t<E>>{{std::forward<E>(e)}}};
  }
};
struct just_stopped_t {
  auto operator()() const {
    return basic_sender{detail::desc::just<set_stopped_t>{{}}};
  }
};
struct read_env_t {
  template <class Q> auto operator()(Q q) const {
    return basic_sender{detail::desc::read_env<Q>{q}};
  }
};
inline constexpr just_t just{};
inline constexpr just_error_t just_error{};
inline constexpr just_stopped_t just_stopped{};
inline constexpr read_env_t read_env{};

} // namespace sg::exec
