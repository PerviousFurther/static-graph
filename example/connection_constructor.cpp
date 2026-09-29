// A connection state can inject constructor arguments into earlier layers.
// This example's injecting descriptor goes immediately after its connection,
// which itself goes immediately after a parameter_node.
#include "../inc/graph.hpp"

#include <cassert>

struct parameter_node {
  template <typename T> struct apply : sg::node::apply<T, apply<T>> {
    using base = typename sg::node::template apply<T, apply<T>>;
    int value;

    template <typename... Args>
    constexpr apply(int initial, parameter_node, Args &&...args)
        : base{sg::node{}, std::forward<Args>(args)...}, value(initial) {}
  };
};

struct constructor_state {
  int initial;

  template <typename T>
  struct apply : sg::connection_state::apply<T, apply<T>> {
    using base = typename sg::connection_state::template apply<T, apply<T>>;

    template <std::size_t... Is, typename... Args>
    constexpr apply(constructor_state s, sg::connection<Is...> c, Args &&...args)
        : base{sg::connection_state{}, c, s.initial,
               std::forward<Args>(args)...} {}
  };
};

constexpr bool test_constructor_arguments() {
  sg::graph g{parameter_node{}, sg::connection<>{}, constructor_state{42},
              parameter_node{}, sg::connection<0>{}, constructor_state{7}};
  return g.get_node<0>()->value == 42 && g.get_node<1>()->value == 7;
}
static_assert(test_constructor_arguments());

int main() { assert(test_constructor_arguments()); }
