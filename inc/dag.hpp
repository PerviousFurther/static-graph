#pragma once

// Executable DAG on top of graph.hpp: a compile-time topological order,
// an owning state, and nesting of executable graphs as nodes.

#include "graph.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace sg {

// Optional owning-state example. Use an explicit std::reference_wrapper<V>
// as V when borrowing is intended; its referent must outlive graph execution.
template <typename V> struct value_state {
  V value;

  template <typename T> struct apply : state::apply<T, apply<T>> {
    using base = typename state::template apply<T, apply<T>>;
    V value;

    template <typename... Args>
    constexpr apply(value_state s, Args &&...args)
        : base{state{}, std::forward<Args>(args)...},
          value(std::move(s.value)) {}
  };
};
template <typename V> value_state(V &&) -> value_state<std::remove_cvref_t<V>>;

namespace detail {

// Composition deliberately keeps child node/state indices local to the child.
template <typename G, typename T>
struct graph_node : node::apply<T, graph_node<G, T>> {
  using base = typename node::apply<T, graph_node<G, T>>;
  G value;

  template <typename... Args>
  constexpr graph_node(G g, Args &&...args)
      : base{node{}, std::forward<Args>(args)...}, value(std::move(g)) {}

  template <typename... States> constexpr void execute(States &...states) {
    value.execute(states...);
  }
  template <typename... States>
  constexpr void execute(States &...states) const {
    value.execute(states...);
  }
  template <std::size_t index> constexpr auto get_node() noexcept {
    return value.template get_node<index>();
  }
  template <std::size_t index> constexpr auto get_node() const noexcept {
    return value.template get_node<index>();
  }
  template <std::size_t node_index, std::size_t state_index>
  constexpr auto get_state() noexcept {
    return value.template get_state<node_index, state_index>();
  }
  template <std::size_t node_index, std::size_t state_index>
  constexpr auto get_state() const noexcept {
    return value.template get_state<node_index, state_index>();
  }

  template <std::size_t N, std::size_t C, std::size_t S>
  constexpr auto get_connection_state() noexcept {
    return value.template get_connection_state<N, C, S>();
  }
  template <std::size_t N, std::size_t C, std::size_t S>
  constexpr auto get_connection_state() const noexcept {
    return value.template get_connection_state<N, C, S>();
  }
  template <std::size_t N>
  static constexpr auto get_connection_count() noexcept {
    return G::template get_connection_count<N>();
  }
  template <std::size_t N, std::size_t C>
  static constexpr auto get_connection_indices() noexcept {
    return G::template get_connection_indices<N, C>();
  }
  template <std::size_t N, std::size_t C>
  static constexpr auto get_connection_state_count() noexcept {
    return G::template get_connection_state_count<N, C>();
  }

protected:
  // Keep the outer graph's indexed lookup overloads visible.
  using base::get_connection_count;
  using base::get_connection_indices;
  using base::get_connection_state;
  using base::get_connection_state_count;
  using base::get_node;
  using base::get_state;
};

struct edge {
  std::size_t from;
  std::size_t to;
};

template <std::size_t N> struct plan_result {
  std::array<std::size_t, N> order{};
  bool valid_indices = true;
  bool acyclic = true;
};

// Compile-time Kahn traversal: always select the earliest declared READY node.
// Time O(N^2 + E), temporary storage O(N + E). Duplicate edges are harmless:
// each counted incoming edge is removed exactly once.
template <std::size_t N, typename... Groups>
consteval auto make_plan(type_list<Groups...>) noexcept {
  constexpr auto edge_count = ((Groups::size() - 1) + ... + std::size_t(0));
  std::array<edge, edge_count> edges{};
  std::size_t offset = 0;
  [[maybe_unused]] auto add_group = [&]<std::size_t To, std::size_t... From>(
                                      std::index_sequence<To, From...>) {
    ((edges[offset++] = edge{From, To}), ...);
  };
  (add_group(Groups{}), ...);

  plan_result<N> result;
  for (const auto &e : edges) {
    if (e.from >= N || e.to >= N) {
      result.valid_indices = false;
      return result;
    }
  }

  std::array<std::size_t, N> indegree{};
  std::array<std::size_t, N> head{};
  std::array<std::size_t, edge_count> next{};
  std::array<bool, N> done{};
  head.fill(edge_count);
  for (std::size_t i = 0; i < edge_count; ++i) {
    const auto &e = edges[i];
    ++indegree[e.to];
    next[i] = head[e.from];
    head[e.from] = i;
  }

  for (std::size_t k = 0; k < N; ++k) {
    std::size_t chosen = N;
    for (std::size_t i = 0; i < N; ++i) {
      if (!done[i] && indegree[i] == 0) {
        chosen = i;
        break;
      }
    }
    if (chosen == N) {
      result.acyclic = false;
      return result;
    }
    result.order[k] = chosen;
    done[chosen] = true;
    for (auto i = head[chosen]; i != edge_count; i = next[i])
      --indegree[edges[i].to];
  }
  return result;
}

template <std::size_t I, typename G, std::size_t... Ss, typename... States>
constexpr void execute_node(
  G &g, std::index_sequence<Ss...>, States &...states) {
  g.template get_node<I>()->execute(
    states..., *g.template get_state<I, Ss>()...);
}

template <typename G, std::size_t... Is, typename... States>
constexpr void execute_graph(
  G &g, std::index_sequence<Is...>, States &...states) {
  (static_cast<void>(execute_node<Is>(g,
     std::make_index_sequence<G::template get_state_count<Is>()>{}, states...)),
    ...);
}

} // namespace detail

// The only public execution entry point. Plain graph remains description/query
// only. Use dag when composing an executable child node.
template <class... Ts> struct dag : graph<Ts...> {
  using base = graph<Ts...>;
  using base::base;

  template <class T> using apply = detail::graph_node<dag, T>;

private:
  static constexpr auto plan =
    detail::make_plan<base::node_count>(base::get_connection());
  static_assert(plan.valid_indices, "dag: connection index out of range");
  static_assert(plan.acyclic, "dag: cyclic dependency");

  template <std::size_t... Is>
  static constexpr auto make_order(
    std::index_sequence<Is...>) -> std::index_sequence<plan.order[Is]...> {
    return {};
  }

public:
  using order_type =
    decltype(make_order(std::make_index_sequence<base::node_count>{}));
  static consteval order_type get_order() noexcept { return {}; }

  template <typename... States> constexpr void execute(States &...states) {
    detail::execute_graph(static_cast<base &>(*this), get_order(), states...);
  }
  template <typename... States>
  constexpr void execute(States &...states) const {
    detail::execute_graph(
      static_cast<const base &>(*this), get_order(), states...);
  }
};
template <class... Ts> dag(Ts &&...) -> dag<std::remove_cvref_t<Ts>...>;

} // namespace sg
