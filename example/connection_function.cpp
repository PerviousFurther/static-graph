// A connection state can expose a function that configures its owning node.
#include "../inc/graph.hpp"

#include <cassert>

struct configured_node {
  template <typename T> struct apply : sg::node::apply<T, apply<T>> {
    using base = typename sg::node::template apply<T, apply<T>>;
    int value = 0;

    template <typename... Args>
    constexpr apply(configured_node, Args &&...args)
        : base{sg::node{}, std::forward<Args>(args)...} {}

    constexpr void set_value(int next) { value = next; }
  };
};

struct forwarding_state {
  template <typename T>
  struct apply : sg::connection_state::apply<T, apply<T>> {
    using base = typename sg::connection_state::template apply<T, apply<T>>;

    template <typename... Args>
    constexpr apply(forwarding_state, Args &&...args)
        : base{sg::connection_state{}, std::forward<Args>(args)...} {}

    constexpr void configure(int value) {
      // These indices are supplied by connection_state's descriptor layer.
      this->get_node(std::in_place_index<base::last_connection_state_node_index>)
          ->set_value(value);
    }
  };
};

constexpr bool test_function_arguments() {
  sg::graph g{configured_node{}, sg::connection<>{}, forwarding_state{},
              configured_node{}, sg::connection<0>{}, forwarding_state{}};
  g.get_connection_state<0, 0, 0>()->configure(42);
  g.get_connection_state<1, 0, 0>()->configure(7);
  return g.get_node<0>()->value == 42 && g.get_node<1>()->value == 7;
}
static_assert(test_function_arguments());

int main() { assert(test_function_arguments()); }
