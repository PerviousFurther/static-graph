#pragma once

// sg::exec: write_env.

#include "../core/graph.hpp"
#include "layers.hpp"

#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
namespace desc {
// A graph state, not a step: it holds E, and the layers before it see E in
// front of the environment below. Completions pass through it by
// inheritance.
template <class E> struct write_env {
  E environment;
  static constexpr bool is_source = false;
  template <class In, class> using completions = In;
  template <class Env> using upstream_env = env<E, Env>;
  template <class T, class In>
  struct apply : sg::state::apply<T, apply<T, In>> {
    using base = sg::state::apply<T, apply<T, In>>;
    using outer_env = std::decay_t<env_of_t<const T &>>;
    E environment;
    template <class... Rest>
    apply(write_env d, Rest &&...rest)
        : base{sg::state{}, std::forward<Rest>(rest)...},
          environment(std::move(d.environment)) {}
    env<E, outer_env> get_env() const noexcept {
      return {environment, static_cast<const T &>(*this).get_env()};
    }
  };
};
} // namespace desc
} // namespace detail

struct write_env_t : detail::adaptor<write_env_t> {
  using adaptor::operator();
  template <sender S, class E> auto operator()(S &&s, E e) const {
    return detail::append(
      std::forward<S>(s), detail::desc::write_env<E>{std::move(e)});
  }
};
inline constexpr write_env_t write_env{};

} // namespace sg::exec
