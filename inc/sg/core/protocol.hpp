#pragma once

// Protocols a graph sends to its nodes, independent of any graph type.
//
// A protocol is a CPO. Called on node I, it passes std::in_place_index<I>
// first and tries, in order:
//  1. the node's member of the same name: node.name(index, xs...);
//  2. the node's generic handler: node.handle(index, tag, xs...), where tag
//     is the CPO object itself, or the event tag for on_event;
//  3. the protocol's default.
// A new protocol is a new CPO; nodes that ignore it need nothing.

#include <cstddef>
#include <type_traits>
#include <utility>

namespace sg {
namespace detail {
template <typename T> struct is_in_place_index : std::false_type {};
template <std::size_t I>
struct is_in_place_index<std::in_place_index_t<I>> : std::true_type {};
} // namespace detail

// The CPO types are public: handle receives them as tags.

struct execute_t {
  // Default: a node without execute does nothing.
  template <typename Node, std::size_t I, typename... Args>
  constexpr void operator()(
    Node &node, std::in_place_index_t<I> index, Args &...args) const {
    if constexpr (requires { node.execute(index, args...); })
      node.execute(index, args...);
    else if constexpr (requires { node.handle(index, *this, args...); })
      node.handle(index, *this, args...);
  }
};

struct on_event_t {
  // Default: a node without a matching handler ignores the event.
  template <typename Node, std::size_t I, typename Tag, typename... Args>
  constexpr void operator()(Node &node, std::in_place_index_t<I> index,
    const Tag &tag, Args &...args) const {
    if constexpr (requires { node.on_event(index, tag, args...); })
      node.on_event(index, tag, args...);
    else if constexpr (requires { node.handle(index, tag, args...); })
      node.handle(index, tag, args...);
  }
};

struct enumerate_t {
  // A graph: enumerate(g, f, outer...).
  template <typename G, typename F, typename... Outer>
    requires(!detail::is_in_place_index<std::remove_cvref_t<F>>::value) &&
    requires(G &g, F &f, Outer &...outer) { g.enumerate(f, outer...); }
  constexpr void operator()(G &g, F &&f, Outer &...outer) const {
    g.enumerate(f, outer...);
  }

  // Node I that contains nodes, such as a nested graph. No default: a node
  // without one is a leaf, and this overload does not exist for it.
  template <typename Node, std::size_t I, typename F, typename... Outer>
    requires requires(Node &node, F &f, Outer &...outer) {
      node.enumerate(std::in_place_index<I>, f, outer...);
    } || requires(Node &node, F &f, const enumerate_t &self, Outer &...outer) {
      node.handle(std::in_place_index<I>, self, f, outer...);
    }
  constexpr void operator()(Node &node, std::in_place_index_t<I> index, F &&f,
    Outer &...outer) const {
    if constexpr (requires { node.enumerate(index, f, outer...); })
      node.enumerate(index, f, outer...);
    else
      node.handle(index, *this, f, outer...);
  }
};

// execute(node, std::in_place_index<I>, args...): runs node I.
inline constexpr execute_t execute{};

// on_event(node, std::in_place_index<I>, tag, args...): delivers tag to
// node I.
inline constexpr on_event_t on_event{};

// enumerate(g, f, outer...) visits the nodes of graph g in execution order as
// f(ctx, outer...), where ctx describes the node. A node that contains nodes
// is entered instead of visited, through enumerate(node, index, f, outer...,
// ctx): outer contexts run from the outermost graph inwards.
inline constexpr enumerate_t enumerate{};

} // namespace sg
