// Activation is an application policy, independent of dag's scheduler.
// Here an external selector visits the states of one connection; each state
// owns its payload and decides how to react.
#include <sg/core/dag.hpp>

#include <cassert>

struct activation_state {
  int initial;

  template <typename T> struct apply : sg::state::apply<T, apply<T>> {
    using base = typename sg::state::template apply<T, apply<T>>;
    int value;
    unsigned activations = 0;

    template <typename... Args>
    constexpr apply(activation_state s, Args &&...args)
        : base{sg::state{}, std::forward<Args>(args)...}, value(s.initial) {}

    constexpr void activate(int &output) {
      output += value;
      ++activations;
    }
  };
};

// A runtime index dispatches to compile-time state lookups. No match means
// no side effects; only the selected connection's states are activated.
template <std::size_t N, typename G>
constexpr bool set_current_connection(G &g, std::size_t current, int &output) {
  return sg::node_context<N, G>{g}.visit_connection(
    current, [&](auto, auto states) {
      std::apply([&](auto &...s) { (s.activate(output), ...); }, states);
    });
}

constexpr bool test_activation() {
  sg::graph g{sg::node{}, sg::connection<>{}, activation_state{2},
    activation_state{3}, sg::connection<>{}, activation_state{10}};
  int output = 0;
  if (!set_current_connection<0>(g, 1, output) || output != 10)
    return false;
  if (g.get_connection_state<0, 0, 0>()->activations != 0)
    return false;
  g.get_connection_state<0, 0, 0>()->value = 7;
  if (!set_current_connection<0>(g, 0, output) || output != 20)
    return false;
  if (set_current_connection<0>(g, 2, output) || output != 20)
    return false;
  return g.get_connection_state<0, 0, 0>()->activations == 1 &&
    g.get_connection_state<0, 0, 1>()->activations == 1 &&
    g.get_connection_state<0, 1, 0>()->activations == 1;
}
static_assert(test_activation());

int main() { assert(test_activation()); }
