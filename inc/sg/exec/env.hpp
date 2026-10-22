#pragma once

// sg::exec: Environments and queries.

#include "tags.hpp"
#include "stop_token.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sg::exec {

template <class Q, class V> struct prop {
  [[no_unique_address]] Q query_;
  V value_;
  constexpr const V &query(Q) const noexcept { return value_; }
};
template <class Q, class V> prop(Q, V) -> prop<Q, V>;

namespace detail {
template <class E, class Q>
concept has_query = requires(const E &e, Q q) { e.query(q); };
} // namespace detail

// Queries are answered by the first environment that supports them.
template <class... Es> struct env {
  std::tuple<Es...> envs;
  constexpr env(Es... es) : envs(std::move(es)...) {}

private:
  template <class Q> static constexpr std::size_t index_of() noexcept {
    constexpr bool found[] = {detail::has_query<Es, Q>..., true};
    std::size_t i = 0;
    while (!found[i])
      ++i;
    return i;
  }

public:
  template <class Q>
    requires(index_of<Q>() < sizeof...(Es))
  constexpr decltype(auto) query(Q q) const noexcept {
    return std::get<index_of<Q>()>(envs).query(q);
  }
};
template <class... Es> env(Es...) -> env<Es...>;

struct get_env_t {
  template <class T>
  constexpr decltype(auto) operator()(const T &t) const noexcept {
    if constexpr (requires { t.get_env(); })
      return t.get_env();
    else
      return env<>{};
  }
};
inline constexpr get_env_t get_env{};
template <class T> using env_of_t = decltype(get_env(std::declval<T>()));

struct get_stop_token_t {
  template <class E> constexpr auto operator()(const E &e) const noexcept {
    if constexpr (detail::has_query<E, get_stop_token_t>)
      return e.query(*this);
    else
      return never_stop_token{};
  }
};
inline constexpr get_stop_token_t get_stop_token{};
template <class T>
using stop_token_of_t =
  std::decay_t<decltype(get_stop_token(std::declval<const T &>()))>;

#define SG_EXEC_QUERY(name)                                                    \
  struct name##_t {                                                            \
    template <class E>                                                         \
    constexpr auto operator()(const E &e) const noexcept                       \
      -> decltype(e.query(std::declval<const name##_t &>())) {                 \
      return e.query(*this);                                                   \
    }                                                                          \
  };                                                                           \
  inline constexpr name##_t name {}
SG_EXEC_QUERY(get_scheduler);
SG_EXEC_QUERY(get_delegation_scheduler);
SG_EXEC_QUERY(get_allocator);
#undef SG_EXEC_QUERY

template <class Tag> struct get_completion_scheduler_t {
  template <class E>
  constexpr auto operator()(const E &e) const noexcept
    -> decltype(e.query(std::declval<const get_completion_scheduler_t &>())) {
    return e.query(*this);
  }
};
template <class Tag>
inline constexpr get_completion_scheduler_t<Tag> get_completion_scheduler{};

enum class forward_progress_guarantee { concurrent, parallel, weakly_parallel };

struct get_forward_progress_guarantee_t {
  template <class Sch>
  constexpr forward_progress_guarantee operator()(
    const Sch &sch) const noexcept {
    if constexpr (detail::has_query<Sch, get_forward_progress_guarantee_t>)
      return sch.query(*this);
    else
      return forward_progress_guarantee::weakly_parallel;
  }
};
inline constexpr get_forward_progress_guarantee_t
  get_forward_progress_guarantee{};

// The most tasks a scheduler's resource runs at once: a fixed property, not
// how many are idle. Not a standard query; parallel bulk splits its work by
// it.
struct get_max_concurrency_t {
  template <class Sch>
    requires detail::has_query<Sch, get_max_concurrency_t>
  constexpr std::size_t operator()(const Sch &sch) const noexcept {
    return sch.query(*this);
  }
};
inline constexpr get_max_concurrency_t get_max_concurrency{};

namespace detail {
// Attributes of a sender whose value and stopped completions run on sch.
template <class Sch> struct sched_attrs {
  Sch sch;
  template <class Tag>
    requires(
      std::is_same_v<Tag, set_value_t> || std::is_same_v<Tag, set_stopped_t>)
  constexpr Sch query(get_completion_scheduler_t<Tag>) const noexcept {
    return sch;
  }
};
} // namespace detail

} // namespace sg::exec
