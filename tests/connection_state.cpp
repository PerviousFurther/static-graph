#include "../inc/graph.hpp"

#include <cassert>
#include <memory>

template <typename V> struct payload_state {
  V value;

  template <typename T>
  struct apply : sg::connection_state::apply<T, apply<T>> {
    using base = typename sg::connection_state::template apply<T, apply<T>>;
    using indices = std::index_sequence<
        base::last_connection_state_node_index,
        base::last_connection_state_connection_index, base::connection_state_index>;
    V value;

    template <typename... Args>
    constexpr apply(payload_state s, Args &&...args)
        : base{sg::connection_state{}, std::forward<Args>(args)...},
          value(std::move(s.value)) {}
  };
};
template <typename V> payload_state(V) -> payload_state<V>;

constexpr bool test_indices_and_lookup() {
  sg::exec_graph g{
      sg::node{}, sg::value_state{1},
      sg::connection<1, 2>{}, payload_state{10}, sg::value_state{2}, payload_state{20},
      sg::connection<2>{}, payload_state{30}, sg::connection<>{},
      sg::node{}, sg::connection<>{}, payload_state{40},
      sg::node{}, sg::connection<>{}, sg::connection_state{},
      sg::node{}};
  using G = decltype(g);
  static_assert(G::get_connection_count<0>() == 3);
  static_assert(G::get_connection_count<1>() == 1);
  static_assert(G::get_connection_count<3>() == 0);
  static_assert(G::get_connection_state_count<0, 0>() == 2);
  static_assert(G::get_connection_state_count<0, 1>() == 1);
  static_assert(G::get_connection_state_count<0, 2>() == 0);
  static_assert(G::get_connection_state_count<1, 0>() == 1);
  static_assert(G::get_connection_state_count<2, 0>() == 1);
  static_assert(G::get_state_count<0>() == 2);
  static_assert(std::is_same_v<decltype(G::get_connection_indices<0, 0>()),
                               std::index_sequence<1, 2>>);
  static_assert(std::is_same_v<decltype(G::get_connection_indices<0, 1>()),
                               std::index_sequence<2>>);
  static_assert(std::is_same_v<decltype(G::get_connection_indices<0, 2>()),
                               std::index_sequence<>>);
  using S = std::remove_pointer_t<decltype(g.get_connection_state<0, 0, 1>())>;
  static_assert(std::is_same_v<typename S::indices, std::index_sequence<0, 0, 1>>);
  using Next = std::remove_pointer_t<decltype(g.get_connection_state<0, 1, 0>())>;
  static_assert(std::is_same_v<typename Next::indices, std::index_sequence<0, 1, 0>>);
  using Other = std::remove_pointer_t<decltype(g.get_connection_state<1, 0, 0>())>;
  static_assert(std::is_same_v<typename Other::indices, std::index_sequence<1, 0, 0>>);
  static_assert(std::is_same_v<
      decltype(std::as_const(g).get_connection_state<0, 0, 1>()), const S *>);

  // Connection states are neither node execution arguments nor automatic hooks.
  g.get_connection_state<0, 0, 1>()->value = 25;
  g.execute();
  std::as_const(g).execute();
  return g.get_state<0, 0>()->value == 1 && g.get_state<0, 1>()->value == 2 &&
         g.get_connection_state<0, 0, 0>()->value == 10 &&
         std::as_const(g).get_connection_state<0, 0, 1>()->value == 25 &&
         g.get_connection_state<0, 1, 0>()->value == 30 &&
         g.get_connection_state<1, 0, 0>()->value == 40 &&
         g.get_connection_state<2, 0, 0>() != nullptr;
}
static_assert(test_indices_and_lookup());

constexpr bool test_nested_graph() {
  sg::exec_graph child{sg::node{}, sg::connection<>{}, payload_state{4}};
  sg::exec_graph parent{
      sg::node{}, sg::connection<>{}, payload_state{1},
      std::move(child), sg::connection<0>{}, payload_state{2},
      sg::node{}, sg::connection<1>{}, payload_state{3}};
  auto *nested = parent.get_node<1>();
  static_assert(nested->template get_connection_count<0>() == 1);
  static_assert(nested->template get_connection_state_count<0, 0>() == 1);
  static_assert(std::is_same_v<decltype(nested->template get_connection_indices<0, 0>()),
                               std::index_sequence<>>);
  nested->template get_connection_state<0, 0, 0>()->value = 5;
  parent.execute();
  std::as_const(parent).execute();
  return parent.get_connection_state<0, 0, 0>()->value == 1 &&
         parent.get_connection_state<1, 0, 0>()->value == 2 &&
         parent.get_connection_state<2, 0, 0>()->value == 3 &&
         std::as_const(parent).get_node<1>()
             ->template get_connection_state<0, 0, 0>()->value == 5;
}
static_assert(test_nested_graph());

int main() {
  assert(test_indices_and_lookup());
  assert(test_nested_graph());
  sg::graph g{sg::node{}, sg::connection<>{}, payload_state{std::make_unique<int>(42)}};
  auto moved = std::move(g);
  assert((*moved.get_connection_state<0, 0, 0>()->value == 42));
}
