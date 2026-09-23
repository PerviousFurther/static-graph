#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sg {
template <typename... Ts> struct type_list {};

template <::std::size_t index, typename T> struct type_list_at {};
template <::std::size_t index> struct type_list_at<index, type_list<>> {};
template <::std::size_t index, typename T, typename... Ts>
  requires(index != 0u)
struct type_list_at<index, type_list<T, Ts...>>
    : type_list_at<index - 1u, type_list<Ts...>> {};
template <typename T, typename... Ts>
struct type_list_at<0u, type_list<T, Ts...>> {
  using type = T;
};

// Derived is optional: node{} remains a no-op node.
// A custom node should use base node::apply<T, apply<T>> so get_node returns
// the concrete node layer.
struct node {
  template <typename T, typename Derived = void> struct apply : T {
    using node_tag = void;
    using self_type =
        std::conditional_t<std::is_void_v<Derived>, apply, Derived>;

    template <typename... Args>
    constexpr apply(node, Args &&...args)
        : T{std::forward<Args>(args)...} {}
    template <typename... Args>
    constexpr apply(Args &&...args)
        : T{std::forward<Args>(args)...} {}

    template <typename... Args>
    constexpr void execute(Args &&...) const noexcept {}

  protected:
    using T::get_node;

    static constexpr std::size_t node_index = []() constexpr {
      if constexpr (requires { T::node_index; })
        return T::node_index + 1;
      else
        return std::size_t{0};
    }();

    constexpr self_type *get_node(std::in_place_index_t<node_index>) noexcept {
      return static_cast<self_type *>(this); // MSVC have bug here when in consteval.
    }
    constexpr const self_type *
    get_node(std::in_place_index_t<node_index>) const noexcept {
      return static_cast<const self_type *>(this); // MSVC have bug here when in consteval.
    }
  };
};

// Attached to node N, connection<I> means I must finish before N starts.
// Forward references are allowed; exec_graph checks the complete graph.
template <std::size_t index> struct connection {
  template <typename T> struct apply : T {
    template <typename... Args>
    constexpr apply(connection, Args &&...args)
      : T{std::forward<Args>(args)...} {}

  protected:
    static_assert(requires { T::node_index; }, "connection must follow a node");
    static constexpr auto last_connection_node_index = T::node_index;
    static constexpr bool last_is_same_connection = []() constexpr {
      if constexpr (requires { T::connection_index; })
        return T::last_connection_node_index == last_connection_node_index;
      else
        return false;
    }();
    static constexpr std::size_t connection_index = []() constexpr {
      if constexpr (last_is_same_connection)
        return T::connection_index + 1;
      else
        return std::size_t{0};
    }();

    static constexpr auto get_connection() noexcept {
      return get_connection_impl(T::get_connection());
    }

  private:
    template <typename... Qs>
    static constexpr auto get_connection_impl(type_list<Qs...>) noexcept
      requires(!last_is_same_connection) {
      return type_list<std::index_sequence<last_connection_node_index, index>, Qs...>{};
    }

    template <std::size_t... ids, typename... Qs>
    static constexpr auto
    get_connection_impl(type_list<std::index_sequence<ids...>, Qs...>) noexcept
      requires(last_is_same_connection) {
      return type_list<std::index_sequence<ids..., index>, Qs...>{};
    }
  };
};

struct state {
  template <typename T, typename Derived = void> struct apply : T {
    using self_type =
        std::conditional_t<std::is_void_v<Derived>, apply, Derived>;

    template <typename... Args>
    constexpr apply(state, Args &&...args)
      : T{std::forward<Args>(args)...} {}
    template <typename...Args>
    constexpr apply(Args &&...args) 
      : T{std::forward<Args>(args)...} {}

  protected:
    using T::get_state;
    using T::get_state_count;

    static constexpr auto last_state_node_index = T::node_index;
    static constexpr bool last_is_same_state = []() constexpr {
      if constexpr (requires { T::state_index; })
        return T::last_state_node_index == last_state_node_index;
      else
        return false;
    }();
    static constexpr std::size_t state_index = []() constexpr {
      if constexpr (last_is_same_state)
        return T::state_index + 1;
      else
        return std::size_t{0};
    }();

    static constexpr std::size_t
    get_state_count(std::in_place_index_t<last_state_node_index>) noexcept {
      return state_index + 1;
    }

    constexpr self_type *
    get_state(std::in_place_index_t<last_state_node_index>,
      std::in_place_index_t<state_index>) noexcept {
      return static_cast<self_type *>(
          this); // MSVC have bug here when in consteval.
    }
    constexpr self_type const *
    get_state(std::in_place_index_t<last_state_node_index>, 
      std::in_place_index_t<state_index>) const noexcept {
      return static_cast<self_type const *>(
          this); // MSVC have bug here when in consteval.
    }
  };
};

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

template <typename... Ts> struct adopt;
template <> struct adopt<> {
  constexpr adopt() = default;

  void get_node(...) const = delete;
  void get_state(...) const = delete;
  static constexpr auto get_connection() noexcept { return type_list<>(); }

  template <std::size_t N>
  static constexpr std::size_t
  get_state_count(std::in_place_index_t<N>) noexcept {
    return 0;
  }
};
template <typename T, typename... Ts>
struct adopt<T, Ts...> : T::template apply<adopt<Ts...>> {
  using base = typename T::template apply<adopt<Ts...>>;

  template <typename... Args>
  constexpr adopt(Args &&...args)
      : base{std::forward<Args>(args)...} {}
};

template <std::size_t N, std::size_t... Is>
constexpr std::index_sequence<(N - 1 - Is)...>
reverse_indices(std::index_sequence<Is...>) {
  return {};
};
template <std::size_t N>
using make_reverse_index_sequence =
  decltype(reverse_indices<N>(std::make_index_sequence<N>{}));

template <typename Tuple, typename Indices> struct reverse_adopt_impl;
template <typename Tuple, std::size_t... Is>
struct reverse_adopt_impl<Tuple, std::index_sequence<Is...>> {
  using type = adopt<std::tuple_element_t<Is, Tuple>...>;
};
template <typename... Ts>
using reverse_adopt = typename reverse_adopt_impl<
    ::std::tuple<Ts...>, make_reverse_index_sequence<sizeof...(Ts)>>::type;

// Composition deliberately keeps child node/state indices local to the child.
template <typename G, typename T>
struct graph_node : node::apply<T, graph_node<G, T>> {
  using base = typename node::apply<T, graph_node<G, T>>;
  G value;

  template <typename... Args>
  constexpr graph_node(G g, Args &&...args)
      : base{node{}, std::forward<Args>(args)...}, value(std::move(g)) {}

  template <typename... States>
  constexpr void execute(States &...states) 
  {
    value.execute(states...);
  }
  template <typename... States>
  constexpr void execute(States &...states) const
  {
    value.execute(states...);
  }
  template <std::size_t index>
  constexpr auto get_node() noexcept {
    return value.template get_node<index>();
  }
  template <std::size_t index>
  constexpr auto get_node() const noexcept {
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

protected:
  // Keep the outer graph's indexed lookup overloads visible.
  using base::get_node;
  using base::get_state;
};

} // namespace detail

template <typename... Ts> struct graph : detail::reverse_adopt<Ts...> {
  using base = detail::reverse_adopt<Ts...>;

private:
  template <std::size_t... Is, typename Tuple>
  constexpr graph(std::index_sequence<Is...>, Tuple &&tuple)
      : base{std::get<Is>(std::forward<Tuple>(tuple))...} {}

public:
  static constexpr std::size_t node_count = []() constexpr {
    if constexpr (requires { base::node_index; })
      return base::node_index + 1;
    else
      return std::size_t(0);
  }();

  template <typename... Args>
    requires(sizeof...(Args) == sizeof...(Ts) &&
             (std::is_same_v<std::remove_cvref_t<Args>, Ts> && ...))
  constexpr graph(Args &&...args)
    : graph(detail::make_reverse_index_sequence<sizeof...(Args)>{}
      , std::forward_as_tuple(std::forward<Args>(args)...)) {}

  template <std::size_t I>
  constexpr auto get_node() noexcept {
    static_assert(requires { base::get_node(std::in_place_index<I>); },
                  "graph: node index out of range");
    return base::get_node(std::in_place_index<I>);
  }
  template <std::size_t I>
  constexpr auto get_node() const noexcept {
    static_assert(requires { base::get_node(std::in_place_index<I>); },
                  "graph: node index out of range");
    return base::get_node(std::in_place_index<I>);
  }

  template <std::size_t N, std::size_t S>
  constexpr auto get_state() noexcept {
    static_assert(requires { base::get_state(std::in_place_index<N>, std::in_place_index<S>); },
                  "graph: state index out of range");
    return base::get_state(std::in_place_index<N>, std::in_place_index<S>);
  }
  template <std::size_t N, std::size_t S>
  constexpr auto get_state() const noexcept {
    static_assert(requires { base::get_state(std::in_place_index<N>, std::in_place_index<S>); },
                  "graph: state index out of range");
    return base::get_state(std::in_place_index<N>, std::in_place_index<S>);
  }

  template <std::size_t N>
  static constexpr std::size_t get_state_count() noexcept {
    static_assert(N < node_count, "graph: node index out of range");
    return base::get_state_count(std::in_place_index<N>);
  }

  static constexpr auto get_connection() noexcept {
    return base::get_connection();
  }

private:
  // Do not accidentally expose execute inherited from the last node layer.
  void execute(...) const = delete;
};
template <typename...Ts> graph(Ts&&...) -> graph<std::remove_cvref_t<Ts>...>;

namespace detail {

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
  [[maybe_unused]] auto add_group 
    = [&]<std::size_t To, std::size_t... From>(std::index_sequence<To, From...>) {
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
constexpr void execute_node(G &g, std::index_sequence<Ss...>, States &...states) {
  g.template get_node<I>()
    ->execute(states..., *g.template get_state<I, Ss>()...);
}

template <typename G, std::size_t... Is, typename... States>
constexpr void execute_graph(G &g, std::index_sequence<Is...>, States &...states) {
  (static_cast<void>(execute_node<Is>(g, std::make_index_sequence<G::template get_state_count<Is>()>{}, states...)), ...);
}

} // namespace detail

// The only public execution entry point. Plain graph remains description/query
// only. Use exec_graph when composing an executable child node.
template <class... Ts> struct exec_graph : graph<Ts...> {
  using base = graph<Ts...>;
  using base::base;

  template <class T> using apply = detail::graph_node<exec_graph, T>;

private:
  static constexpr auto plan =
      detail::make_plan<base::node_count>(base::get_connection());
  static_assert(plan.valid_indices,
                "exec_graph: connection index out of range");
  static_assert(plan.acyclic, "exec_graph: cyclic dependency");

  template <std::size_t... Is>
  static constexpr auto make_order(std::index_sequence<Is...>)
      -> std::index_sequence<plan.order[Is]...> { return {}; }

public:
  using order_type = decltype(make_order(std::make_index_sequence<base::node_count>{}));
  static consteval order_type get_order() noexcept { return {}; }

  template <typename... States>
  constexpr void execute(States &...states) {
    detail::execute_graph(static_cast<base &>(*this), get_order(), states...);
  }
  template <typename... States>
  constexpr void execute(States &...states) const {
    detail::execute_graph(static_cast<const base &>(*this), get_order(), states...);
  }
};
template <class... Ts>
exec_graph(Ts &&...) -> exec_graph<std::remove_cvref_t<Ts>...>;

}
