#pragma once

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

template <typename List> struct type_list_size;
template <typename... Ts>
struct type_list_size<type_list<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <typename... Lists> struct type_list_cat {
  using type = type_list<>;
};
template <typename... Ts> struct type_list_cat<type_list<Ts...>> {
  using type = type_list<Ts...>;
};
template <typename... As, typename... Bs, typename... Rest>
struct type_list_cat<type_list<As...>, type_list<Bs...>, Rest...>
    : type_list_cat<type_list<As..., Bs...>, Rest...> {};

// Keeps the first occurrence of each type.
template <typename List, typename Out = type_list<>> struct type_list_unique {
  using type = Out;
};
template <typename T, typename... Ts, typename... Os>
struct type_list_unique<type_list<T, Ts...>, type_list<Os...>>
    : type_list_unique<type_list<Ts...>,
        std::conditional_t<(std::is_same_v<T, Os> || ...), type_list<Os...>,
          type_list<Os..., T>>> {};

template <template <typename...> class F, typename List> struct type_list_apply;
template <template <typename...> class F, typename... Ts>
struct type_list_apply<F, type_list<Ts...>> {
  using type = F<Ts...>;
};

// A descriptor whose apply<As...> is F<As..., Bs...>.
template <template <typename...> class F, typename... Bs>
struct type_bind_back {
  template <typename... As> using apply = F<As..., Bs...>;
};

// Applies descriptors in order: apply_each_t<T, A, B> is
// B::apply<A::apply<T>>.
template <typename T, typename... Ds> struct apply_each {
  using type = T;
};
template <typename T, typename D, typename... Ds>
struct apply_each<T, D, Ds...>
    : apply_each<typename D::template apply<T>, Ds...> {};
template <typename T, typename... Ds>
using apply_each_t = typename apply_each<T, Ds...>::type;

// Derived is optional: node{} remains a no-op node.
// A custom node should use base node::apply<T, apply<T>> so get_node returns
// the concrete node layer.
struct node {
  template <typename T, typename Derived = void> struct apply : T {
    using node_tag = void;
    using self_type =
      std::conditional_t<std::is_void_v<Derived>, apply, Derived>;

    template <typename... Args>
    constexpr apply(node, Args &&...args) : T{std::forward<Args>(args)...} {}
    template <typename... Args>
    constexpr apply(Args &&...args) : T{std::forward<Args>(args)...} {}

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
      return static_cast<self_type*>(this); // MSVC have bug here when in consteval.
    }
    constexpr const self_type *get_node(
      std::in_place_index_t<node_index>) const noexcept {
      return static_cast<const self_type*>(this); // MSVC have bug here when in consteval.
    }
  };
};

// Attached to node N, connection<Is...> means every Is must finish before N
// starts. An empty pack adds no dependencies.
// Forward references are allowed; dag checks the complete graph.
template <std::size_t... indices> struct connection {
  template <typename T> struct apply : T {
    template <typename... Args>
    constexpr apply(connection, Args &&...args)
        : T{std::forward<Args>(args)...} {}

  protected:
    using T::get_connection_count;
    using T::get_connection_indices;

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

    static constexpr std::size_t get_connection_count(
      std::in_place_index_t<last_connection_node_index>) noexcept {
      return connection_index + 1;
    }

    static constexpr auto get_connection_indices(
      std::in_place_index_t<last_connection_node_index>,
      std::in_place_index_t<connection_index>) noexcept {
      return std::index_sequence<indices...>{};
    }

  private:
    template <typename... Qs>
    static constexpr auto get_connection_impl(type_list<Qs...>) noexcept
      requires(!last_is_same_connection)
    {
      return type_list<
        std::index_sequence<last_connection_node_index, indices...>, Qs...>{};
    }

    template <std::size_t... ids, typename... Qs>
    static constexpr auto get_connection_impl(
      type_list<std::index_sequence<ids...>, Qs...>) noexcept
      requires(last_is_same_connection)
    {
      return type_list<std::index_sequence<ids..., indices...>, Qs...>{};
    }
  };
};

namespace detail {
template <typename T, typename Derived> struct node_state;
template <typename T, typename Derived> struct connection_state;

// Whether the layers in T have already declared a connection on their node.
// The indices are protected, so the check reads them from a derived class.
template <typename T> struct has_connection : T {
  static constexpr bool value = [] {
    if constexpr (requires {
                    T::node_index;
                    T::last_connection_node_index;
                  })
      return T::last_connection_node_index == T::node_index;
    else
      return false;
  }();
};
} // namespace detail

// A state belongs to the current node until the node's first connection, and
// to the node's last connection after it. Custom states derive from
// state::apply<T, apply<T>>.
struct state {
  template <typename T, typename Derived = void>
  using apply = std::conditional_t<detail::has_connection<T>::value,
    detail::connection_state<T, Derived>, detail::node_state<T, Derived>>;
};

namespace detail {
// A state before the current node's first connection belongs to the node and
// is passed to its execute.
template <typename T, typename Derived> struct node_state : T {
  using self_type =
    std::conditional_t<std::is_void_v<Derived>, node_state, Derived>;

  template <typename... Args>
  constexpr node_state(state, Args &&...args)
      : T{std::forward<Args>(args)...} {}
  template <typename... Args>
  constexpr node_state(Args &&...args) : T{std::forward<Args>(args)...} {}

protected:
  static_assert(requires { T::node_index; }, "state must follow a node");

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

  static constexpr std::size_t get_state_count(
    std::in_place_index_t<last_state_node_index>) noexcept {
    return state_index + 1;
  }

  constexpr self_type *get_state(std::in_place_index_t<last_state_node_index>,
    std::in_place_index_t<state_index>) noexcept {
    return static_cast<self_type *>(
      this); // MSVC have bug here when in consteval.
  }
  constexpr self_type const *get_state(
    std::in_place_index_t<last_state_node_index>,
    std::in_place_index_t<state_index>) const noexcept {
    return static_cast<self_type const *>(
      this); // MSVC have bug here when in consteval.
  }
};

// A state after a connection belongs to the current node's last connection.
// It supplies indexing and lookup only; behavior belongs to custom states.
template <typename T, typename Derived> struct connection_state : T {
  using self_type =
    std::conditional_t<std::is_void_v<Derived>, connection_state, Derived>;

  template <typename... Args>
  constexpr connection_state(state, Args &&...args)
      : T{std::forward<Args>(args)...} {}
  template <typename... Args>
  constexpr connection_state(Args &&...args) : T{std::forward<Args>(args)...} {}

protected:
  using T::get_connection_state;
  using T::get_connection_state_count;

  static constexpr auto last_connection_state_node_index = T::node_index;
  static constexpr auto last_connection_state_connection_index =
    T::connection_index;
  static constexpr std::size_t connection_state_index = [] {
    if constexpr (requires { T::connection_state_index; }) {
      if constexpr (T::last_connection_state_node_index == T::node_index &&
        T::last_connection_state_connection_index == T::connection_index)
        return T::connection_state_index + 1;
    }
    return std::size_t{0};
  }();

  static constexpr std::size_t get_connection_state_count(
    std::in_place_index_t<last_connection_state_node_index>,
    std::in_place_index_t<last_connection_state_connection_index>) noexcept {
    return connection_state_index + 1;
  }

  constexpr self_type *get_connection_state(
    std::in_place_index_t<last_connection_state_node_index>,
    std::in_place_index_t<last_connection_state_connection_index>,
    std::in_place_index_t<connection_state_index>) noexcept {
    return static_cast<self_type *>(this);
  }
  constexpr const self_type *get_connection_state(
    std::in_place_index_t<last_connection_state_node_index>,
    std::in_place_index_t<last_connection_state_connection_index>,
    std::in_place_index_t<connection_state_index>) const noexcept {
    return static_cast<const self_type *>(this);
  }
};

// Bottom of every graph: lookups that reach it are out of range.
struct root {
  constexpr root() = default;

  void get_node(...) const = delete;
  void get_state(...) const = delete;
  void get_connection_state(...) const = delete;
  static constexpr auto get_connection() noexcept { return type_list<>(); }
  static void get_connection_indices(...) = delete;

  template <std::size_t N>
  static constexpr std::size_t get_connection_count(
    std::in_place_index_t<N>) noexcept {
    return 0;
  }

  template <std::size_t N, std::size_t C>
  static constexpr std::size_t get_connection_state_count(
    std::in_place_index_t<N>, std::in_place_index_t<C>) noexcept {
    return 0;
  }

  template <std::size_t N>
  static constexpr std::size_t get_state_count(
    std::in_place_index_t<N>) noexcept {
    return 0;
  }
};
template <std::size_t N, std::size_t... Is>
constexpr std::index_sequence<(N - 1 - Is)...> reverse_indices(
  std::index_sequence<Is...>) {
  return {};
};
template <std::size_t N>
using make_reverse_index_sequence =
  decltype(reverse_indices<N>(std::make_index_sequence<N>{}));

} // namespace detail

// The first descriptor is applied first, so it is the innermost base; the
// constructor receives the outermost descriptor's argument first.
template <typename... Ts> struct graph : apply_each_t<detail::root, Ts...> {
  using base = apply_each_t<detail::root, Ts...>;

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
      : graph(detail::make_reverse_index_sequence<sizeof...(Args)>{},
          std::forward_as_tuple(std::forward<Args>(args)...)) {}

  template <std::size_t I> constexpr auto get_node() noexcept {
    static_assert(
      requires { base::get_node(std::in_place_index<I>); },
      "graph: node index out of range");
    return base::get_node(std::in_place_index<I>);
  }
  template <std::size_t I> constexpr auto get_node() const noexcept {
    static_assert(
      requires { base::get_node(std::in_place_index<I>); },
      "graph: node index out of range");
    return base::get_node(std::in_place_index<I>);
  }

  template <std::size_t N, std::size_t S> constexpr auto get_state() noexcept {
    static_assert(
      requires {
        base::get_state(std::in_place_index<N>, std::in_place_index<S>);
      }, "graph: state index out of range");
    return base::get_state(std::in_place_index<N>, std::in_place_index<S>);
  }
  template <std::size_t N, std::size_t S>
  constexpr auto get_state() const noexcept {
    static_assert(
      requires {
        base::get_state(std::in_place_index<N>, std::in_place_index<S>);
      }, "graph: state index out of range");
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

  template <std::size_t N>
  static constexpr std::size_t get_connection_count() noexcept {
    static_assert(N < node_count, "graph: node index out of range");
    return base::get_connection_count(std::in_place_index<N>);
  }

  template <std::size_t N, std::size_t C>
  static constexpr auto get_connection_indices() noexcept {
    static_assert(
      C < get_connection_count<N>(), "graph: connection index out of range");
    return base::get_connection_indices(
      std::in_place_index<N>, std::in_place_index<C>);
  }

  template <std::size_t N, std::size_t C>
  static constexpr std::size_t get_connection_state_count() noexcept {
    static_assert(
      C < get_connection_count<N>(), "graph: connection index out of range");
    return base::get_connection_state_count(
      std::in_place_index<N>, std::in_place_index<C>);
  }

  template <std::size_t N, std::size_t C, std::size_t S>
  constexpr auto get_connection_state() noexcept {
    static_assert(S < get_connection_state_count<N, C>(),
      "graph: connection state index out of range");
    return base::get_connection_state(
      std::in_place_index<N>, std::in_place_index<C>, std::in_place_index<S>);
  }
  template <std::size_t N, std::size_t C, std::size_t S>
  constexpr auto get_connection_state() const noexcept {
    static_assert(S < get_connection_state_count<N, C>(),
      "graph: connection state index out of range");
    return base::get_connection_state(
      std::in_place_index<N>, std::in_place_index<C>, std::in_place_index<S>);
  }

private:
  // Do not accidentally expose execute inherited from the last node layer.
  void execute(...) const = delete;
};
template <typename... Ts> graph(Ts &&...) -> graph<std::remove_cvref_t<Ts>...>;

} // namespace sg
