// Standalone example: compile with -std=c++20.

#include <sg/core/dag.hpp>

#include <array>
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
    constexpr void execute(
      std::in_place_index_t<base::node_index>, trace &trace, States &...states) {
      trace.push(Id);
      // Each supplied state is the concrete state::apply-derived object.
      (++states.value, ...);
    }
  };
};

constexpr auto make_graph() {
  using namespace sg;
  dag child{record_node<10>{}, value_state{0}, connection<1>{},
    record_node<11>{}, value_state{0}};
  return dag{record_node<0>{}, value_state{0}, std::move(child),
    value_state{100}, connection<0>{}, record_node<2>{}, value_state{0},
    connection<1>{}};
}

// Child and parent own separate local states; one trace records the order.
constexpr bool check(trace &log) {
  auto g = make_graph();
  g.execute(log);
  // Boundary state is shared by the two child leaves; local states stay local.
  return g.get_state<1, 0>()->value == 102 &&
    g.get_node<1>()->get_state<0, 0>()->value == 1 &&
    g.get_node<1>()->get_state<1, 0>()->value == 1 &&
    g.get_state<0, 0>()->value == 1 && g.get_state<2, 0>()->value == 1 &&
    log.record_count == 4 && log.ids[0] == 0 && log.ids[1] == 11 &&
    log.ids[2] == 10 && log.ids[3] == 2;
}

// MSVC rejects constexpr execution through the CRTP chain.
#ifndef _MSC_VER
static_assert([] {
  trace log;
  return check(log);
}());
#endif

int main() {
  trace log;
  [[maybe_unused]] bool ok = check(log);
  assert(ok);
  for (std::size_t i = 0; i < log.record_count; ++i)
    std::cout << (i == 0 ? "" : " ") << log.ids[i];
  std::cout << '\n'; // 0 11 10 2
}
