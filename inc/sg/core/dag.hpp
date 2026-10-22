#pragma once

// Executable DAG on top of graph.hpp: a compile-time topological order,
// an owning state, and nesting of executable graphs as nodes.

#include "graph.hpp"
#include "protocol.hpp"

#include <array>
#include <tuple>
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
// Ids node N spans: a node that contains nodes declares id_count, a leaf
// spans one.
template <typename N> consteval std::size_t id_count_of() noexcept {
  if constexpr (requires { N::id_count; })
    return N::id_count;
  else
    return 1;
}
template <typename G, std::size_t I>
using node_type_t =
  std::remove_pointer_t<decltype(std::declval<G &>().template get_node<I>())>;

// Ids spanned by the nodes of G declared before I.
template <typename G, std::size_t I> consteval std::size_t id_offset() noexcept {
  return []<std::size_t... Js>(std::index_sequence<Js...>) {
    return (id_count_of<node_type_t<G, Js>>() + ... + std::size_t(0));
  }(std::make_index_sequence<I>{});
}
} // namespace detail

// Node I of graph G, as enumerate visits it and on_event receives it. It
// exposes the node's states and connection states, which are layers above the
// node and so unreachable from the node itself.
//
// node_index is local to G. node_id numbers the nodes of the outermost graph
// with nested graphs flattened, in declaration order: [0, id_count) of that
// graph. A nested graph's position spans its nodes' ids, from Base. One graph
// object per type is assumed, so the id carries no address.
template <std::size_t I, typename G, std::size_t Base = 0> struct node_context {
  G &graph;

  static constexpr std::size_t node_index = I;
  static constexpr std::size_t node_id = Base + detail::id_offset<G, I>();
  static constexpr std::size_t connection_count =
    G::template get_connection_count<I>();

  constexpr auto &node() const noexcept {
    return *graph.template get_node<I>();
  }

  // std::tuple of references to the node's states, in declaration order.
  constexpr auto states() const noexcept {
    return tie_states(
      std::make_index_sequence<G::template get_state_count<I>()>{});
  }

  // std::tuple of references to the states of connection C.
  template <std::size_t C> constexpr auto connection_states() const noexcept {
    return tie_connection_states<C>(std::make_index_sequence<
      G::template get_connection_state_count<I, C>()>{});
  }

  // Dispatches a runtime connection index to
  // f(std::integral_constant<std::size_t, C>, connection_states<C>()).
  // Returns false, without calling f, when active is out of range.
  template <typename F>
  constexpr bool visit_connection(std::size_t active, F &&f) const {
    return visit_connection_impl(
      active, f, std::make_index_sequence<connection_count>{});
  }

private:
  template <std::size_t... Ss>
  constexpr auto tie_states(std::index_sequence<Ss...>) const noexcept {
    return std::tie(*graph.template get_state<I, Ss>()...);
  }
  template <std::size_t C, std::size_t... Ss>
  constexpr auto tie_connection_states(
    std::index_sequence<Ss...>) const noexcept {
    return std::tie(*graph.template get_connection_state<I, C, Ss>()...);
  }
  template <typename F, std::size_t... Cs>
  constexpr bool visit_connection_impl(
    std::size_t active, F &f, std::index_sequence<Cs...>) const {
    return ((active == Cs &&
              (f(std::integral_constant<std::size_t, Cs>{},
                 connection_states<Cs>()),
                true)) ||
      ...);
  }
};

namespace detail {
// sg::execute on the node of ctx with args..., outer states..., node states.
template <typename Args, typename Ctx, typename... Outer>
constexpr void invoke_execute(Args args, const Ctx &ctx, const Outer &...outer) {
  std::apply(
    [&](auto &...xs) {
      sg::execute(ctx.node(), std::in_place_index<Ctx::node_index>, xs...);
    },
    std::tuple_cat(args, outer.states()..., ctx.states()));
}

// Composition deliberately keeps child node/state indices local to the child.
template <typename G, typename T>
struct graph_node : node::apply<T, graph_node<G, T>> {
  using base = typename node::apply<T, graph_node<G, T>>;
  G value;

  template <typename... Args>
  constexpr graph_node(G g, Args &&...args)
      : base{node{}, std::forward<Args>(args)...}, value(std::move(g)) {}

  static constexpr std::size_t id_count = G::id_count;

  // The child's nodes take this position's place in the parent.
  template <typename F, typename... Outer>
  constexpr void enumerate(std::in_place_index_t<base::node_index>, F &&f,
    Outer &...outer) {
    value.enumerate(f, outer...);
  }
  template <typename F, typename... Outer>
  constexpr void enumerate(std::in_place_index_t<base::node_index>, F &&f,
    Outer &...outer) const {
    value.enumerate(f, outer...);
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

template <typename... Groups>
consteval auto make_edges(type_list<Groups...>) noexcept {
  constexpr auto edge_count = ((Groups::size() - 1) + ... + std::size_t(0));
  std::array<edge, edge_count> edges{};
  std::size_t offset = 0;
  [[maybe_unused]] auto add_group = [&]<std::size_t To, std::size_t... From>(
                                      std::index_sequence<To, From...>) {
    ((edges[offset++] = edge{From, To}), ...);
  };
  (add_group(Groups{}), ...);
  return edges;
}

// Compile-time Kahn traversal: always select the earliest declared READY node.
// Time O(N^2 + E), temporary storage O(N + E). Duplicate edges are harmless:
// each counted incoming edge is removed exactly once.
template <std::size_t N, std::size_t E>
consteval auto make_plan(const std::array<edge, E> &edges) noexcept {
  plan_result<N> result;
  for (const auto &e : edges) {
    if (e.from >= N || e.to >= N) {
      result.valid_indices = false;
      return result;
    }
  }

  std::array<std::size_t, N> indegree{};
  std::array<std::size_t, N> head{};
  std::array<std::size_t, E> next{};
  std::array<bool, N> done{};
  head.fill(E);
  for (std::size_t i = 0; i < E; ++i) {
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
    for (auto i = head[chosen]; i != E; i = next[i])
      --indegree[edges[i].to];
  }
  return result;
}

// reach[a][b]: b is a itself or depends on a, directly or transitively.
// Walks the order backwards so every successor is complete before use.
template <std::size_t N, std::size_t E>
consteval auto make_reach(
  const std::array<edge, E> &edges, const plan_result<N> &plan) noexcept {
  std::array<std::array<bool, N>, N> reach{};
  if (!plan.valid_indices || !plan.acyclic)
    return reach;
  for (std::size_t k = N; k-- > 0;) {
    const auto from = plan.order[k];
    reach[from][from] = true;
    for (const auto &e : edges)
      if (e.from == from)
        for (std::size_t i = 0; i < N; ++i)
          reach[from][i] = reach[from][i] || reach[e.to][i];
  }
  return reach;
}

} // namespace detail

// The only public execution entry point. Plain graph remains description/query
// only. Use dag when composing an executable child node.
template <class... Ts> struct dag : graph<Ts...> {
  using base = graph<Ts...>;
  using base::base;

  template <class T> using apply = detail::graph_node<dag, T>;

private:
  static constexpr auto edges = detail::make_edges(base::get_connection());
  static constexpr auto plan = detail::make_plan<base::node_count>(edges);
  static constexpr auto reach = detail::make_reach(edges, plan);
  static_assert(plan.valid_indices, "dag: connection index out of range");
  static_assert(plan.acyclic, "dag: cyclic dependency");

  template <std::size_t... Is>
  static constexpr auto make_order(
    std::index_sequence<Is...>) -> std::index_sequence<plan.order[Is]...> {
    return {};
  }

public:
  // Ids of the nodes, nested graphs flattened; see node_context::node_id.
  static constexpr std::size_t id_count =
    detail::id_offset<base, base::node_count>();

  using order_type =
    decltype(make_order(std::make_index_sequence<base::node_count>{}));
  static consteval order_type get_order() noexcept { return {}; }

  // True when To is From itself or depends on From, directly or transitively.
  static constexpr bool reaches(std::size_t from, std::size_t to) noexcept {
    return reach[from][to];
  }

  // Visits the nodes in execution order; see sg::enumerate.
  template <typename F, typename... Outer>
  constexpr void enumerate(F &&f, Outer &...outer) {
    visit<all>(static_cast<base &>(*this), f, order_type{}, outer...);
  }
  template <typename F, typename... Outer>
  constexpr void enumerate(F &&f, Outer &...outer) const {
    visit<all>(static_cast<const base &>(*this), f, order_type{}, outer...);
  }

  // Runs every node once, in execution order, as node.execute(
  // std::in_place_index<I>, args..., node_states...). Nodes of a nested dag
  // also receive the states of the dag's position, before their own. Nodes
  // without execute are skipped.
  template <typename... Args> constexpr void execute(Args &...args) {
    enumerate([&](const auto &ctx, const auto &...outer) {
      detail::invoke_execute(std::tie(args...), ctx, outer...);
    });
  }
  template <typename... Args> constexpr void execute(Args &...args) const {
    enumerate([&](const auto &ctx, const auto &...outer) {
      detail::invoke_execute(std::tie(args...), ctx, outer...);
    });
  }

  // Delivers tag to every node, in execution order, as
  // sg::on_event(node, std::in_place_index<I>, tag, ctx, args...).
  template <typename Tag, typename... Args>
  constexpr void emit(const Tag &tag, Args &...args) {
    visit<all>(static_cast<base &>(*this), emitter(tag, args...), order_type{});
  }

  // Like emit, restricted to From and the nodes that depend on it. A nested
  // dag that depends on From receives the event on all its nodes.
  template <std::size_t From, typename Tag, typename... Args>
  constexpr void emit_from(const Tag &tag, Args &...args) {
    static_assert(From < base::node_count, "dag: node index out of range");
    visit<From>(static_cast<base &>(*this), emitter(tag, args...), order_type{});
  }

private:
  static constexpr std::size_t all = std::size_t(-1);

  // First id of this graph: the id of its position in the enclosing graph.
  template <typename... Outer> static consteval std::size_t base_id() noexcept {
    if constexpr (sizeof...(Outer) == 0)
      return 0;
    else
      return std::array{Outer::node_id...}.back();
  }

  template <typename Tag, typename... Args>
  static constexpr auto emitter(const Tag &tag, Args &...args) {
    return [t = &tag, ... a = &args]<typename Ctx>(Ctx &ctx, auto &...) {
      sg::on_event(
        ctx.node(), std::in_place_index<Ctx::node_index>, *t, ctx, *a...);
    };
  }

  // Visits the nodes in Is that From reaches, or all of them.
  template <std::size_t From, typename B, typename F, std::size_t... Is,
    typename... Outer>
  static constexpr void visit(
    B &b, F &&f, std::index_sequence<Is...>, Outer &...outer) {
    (visit_node<Is, From == all || reach[From][Is]>(b, f, outer...), ...);
  }
  template <std::size_t I, bool Enabled, typename B, typename F,
    typename... Outer>
  static constexpr void visit_node(B &b, F &f, Outer &...outer) {
    if constexpr (Enabled) {
      node_context<I, B, base_id<Outer...>()> ctx{b};
      auto &node = ctx.node();
      if constexpr (requires {
                      sg::enumerate(node, std::in_place_index<I>, f, outer...,
                        ctx);
                    })
        sg::enumerate(node, std::in_place_index<I>, f, outer..., ctx);
      else
        f(ctx, outer...);
    }
  }
};
template <class... Ts> dag(Ts &&...) -> dag<std::remove_cvref_t<Ts>...>;

} // namespace sg
