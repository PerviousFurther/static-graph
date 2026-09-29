// Standalone example: compile with -std=c++20.

#include "../inc/graph.hpp"

// main.cpp

#include <cassert>
#include <iostream>

struct trace {
  std::array<std::size_t, 32> ids{};
  std::size_t record_count = 0;

  constexpr void push(std::size_t id) {
    if (record_count == ids.size())
      throw "trace capacity exceeded";
    ids[record_count++] = id;
  }
};

// User-defined node; apply owns the node payload and receives attached states.
template <int Id> struct record_node {
  template <typename T> struct apply : sg::node::apply<T, apply<T>> {
    using base = sg::node::apply<T, apply<T>>;

    template <class... Args>
    constexpr apply(record_node, Args &&...args)
        : base{std::forward<Args>(args)...} {}

    template <class... States>
    constexpr void execute(trace &trace, States &...states) {
      trace.push(Id);
      // Each supplied state is the concrete state::apply-derived object.
      (++states.value, ...);
    }

    static constexpr auto have_record = 0;
  };
};

// MSVC have constexpr bug with CRTP.
// you can test for clang or gcc.
// constexpr bool test_constexpr_execution() {
//   exec_graph g{
//     record_node<0>{}, value_state{10}, connection<2>{},
//     record_node<1>{}, value_state{20},
//     record_node<2>{}, value_state{30}
//   };
//   static_assert(std::is_same_v<decltype(g.get_order()),
//   std::index_sequence<1, 2, 0>>); g.execute(); return g.record_count == 3u;
// }
// static_assert(test_constexpr_execution());

int main() {
  using namespace sg;

  // Child and parent own separate local logs; share one trace for total order.
  trace log;
  exec_graph child{record_node<10>{}, value_state{0}, connection<1>{},
    record_node<11>{}, value_state{0}};
  exec_graph g{record_node<0>{}, value_state{0}, std::move(child),
    value_state{100}, connection<0>{}, record_node<2>{}, value_state{0},
    connection<1>{}};
  g.execute(log);
  for (std::size_t i = 0; i < log.record_count; ++i)
    std::cout << (i == 0 ? "" : " ") << log.ids[i];
  std::cout << '\n'; // 0 11 10 2

  // Boundary state is shared by the two child leaves; local states stay local.
  assert((g.get_state<1, 0>()->value == 102));
  assert((g.get_node<1>()->get_state<0, 0>()->value == 1));
  assert((g.get_node<1>()->get_state<1, 0>()->value == 1));
  assert((g.get_state<0, 0>()->value == 1));
  assert((g.get_state<2, 0>()->value == 1));
  assert(log.record_count == 4);
  assert((log.ids[0] == 0 && log.ids[1] == 11 && log.ids[2] == 10 &&
    log.ids[3] == 2));
  return 0;
}
