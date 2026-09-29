#include "../inc/graph.hpp"

#include <cassert>

template <std::size_t Id> struct record_node {
  template <typename T> struct apply : sg::node::apply<T, apply<T>> {
    using base = typename sg::node::template apply<T, apply<T>>;

    template <typename... Args>
    constexpr apply(record_node, Args &&...args)
        : base{std::forward<Args>(args)...} {}

    constexpr void execute(std::array<std::size_t, 3> &trace,
                           std::size_t &count) const {
      trace[count++] = Id;
    }
  };
};

constexpr bool test_connections() {
  sg::exec_graph g{record_node<0>{}, sg::connection<1, 2>{},
                   record_node<1>{}, sg::connection<>{}, record_node<2>{}};
  static_assert(std::is_same_v<decltype(g.get_order()),
                               std::index_sequence<1, 2, 0>>);
  std::array<std::size_t, 3> trace{};
  std::size_t count = 0;
  g.execute(trace, count);
  if (count != 3 || trace != std::array<std::size_t, 3>{1, 2, 0})
    return false;
  count = 0;
  std::as_const(g).execute(trace, count);
  return count == 3 && trace == std::array<std::size_t, 3>{1, 2, 0};
}
static_assert(test_connections());

using combined = decltype(sg::exec_graph{
    sg::node{}, sg::connection<>{}, sg::connection<1>{},
    sg::value_state{0}, sg::connection<1, 2>{}, sg::connection<>{},
    sg::node{}, sg::node{}});
static_assert(std::is_same_v<combined::order_type,
                             std::index_sequence<1, 2, 0>>);
static_assert(combined::get_state_count<0>() == 1);

using empty = decltype(sg::exec_graph{sg::node{}, sg::connection<>{}});
static_assert(std::is_same_v<empty::order_type, std::index_sequence<0>>);

// Exercise validation with the same metadata consumed by exec_graph.
using invalid = decltype(sg::graph{sg::node{}, sg::connection<0, 2>{}});
static_assert(!sg::detail::make_plan<1>(invalid::get_connection()).valid_indices);
using cyclic = decltype(sg::graph{
    sg::node{}, sg::connection<1, 2>{},
    sg::node{}, sg::connection<0>{}, sg::node{}});
static_assert(!sg::detail::make_plan<3>(cyclic::get_connection()).acyclic);

int main() { assert(test_connections()); }
