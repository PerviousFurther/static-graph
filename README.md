# static graph

static graph is a header-only C++20 library for describing and running small dependency graphs. Nodes, their local states, and dependencies are listed in one declaration. `sg::exec_graph` computes a valid execution order at compile time and runs each node in that order.

## Getting started

Add `inc/graph.hpp` to your project and compile with C++20. There are no library binaries to link.

```cpp
#include "graph.hpp"

#include <cassert>
#include <utility>
#include <vector>

template <int Id>
struct task {
  template <class T>
  struct apply : sg::node::apply<T, apply<T>> {
    using base = sg::node::apply<T, apply<T>>;

    template <class... Args>
    constexpr apply(task, Args&&... args)
        : base{std::forward<Args>(args)...} {}

    template <class State>
    void execute(std::vector<int>& order, State& local) {
      order.push_back(Id);
      ++local.value;
    }
  };
};

int main() {
  sg::exec_graph g{
      task<0>{}, sg::value_state{0}, sg::connection<1>{},
      task<1>{}, sg::value_state{0}
  };

  std::vector<int> order;
  g.execute(order);

  assert((order == std::vector<int>{1, 0}));
  assert((g.get_state<0, 0>()->value == 1));
  assert((g.get_state<1, 0>()->value == 1));
}
```

From the `graph` directory, save this as `quickstart.cpp` and compile it with either:

```sh
g++ -std=c++20 -I inc quickstart.cpp -o quickstart
# or
clang++ -std=c++20 -I inc quickstart.cpp -o quickstart
```

With the MSVC developer command prompt:

```bat
cl /std:c++20 /EHsc /I inc quickstart.cpp
```

## Declaring a graph

`sg::node{}` is a node that does nothing. To define behavior, provide a type with a nested `apply<T>` that derives from `sg::node::apply<T, apply<T>>`, as in the example above. The constructor receives the node descriptor first, then forwards the remaining graph construction arguments to its base. Define `execute(...)` on the `apply` type to run the node.

Place states and connections **after the node they belong to**. `sg::value_state{value}` creates an owned state with a public `value` member. A node can have several states. A custom state descriptor can similarly provide `apply<T>` derived from `sg::state::apply<T, apply<T>>`.

`sg::connection<I>{}` says that node `I` must finish before the current node starts. Node indices are zero-based and count only nodes, not states or connections. Dependencies can refer to nodes declared later. The compiler rejects dependencies with an out-of-range index and cycles.

`sg::graph` holds the declaration and supports queries; `sg::exec_graph` adds execution. Both types can be deduced from their constructor arguments. When several nodes are ready, `exec_graph` picks the earliest declared one. The order is fixed by the graph's types and can be inspected with `decltype(g.get_order())`, an `std::index_sequence`.

## Execution and state

Call `g.execute(globals...)` with any shared state objects your nodes need. Each node receives those objects by reference, followed by its own attached states in declaration order:

```cpp
node.execute(globals..., node_state_0, node_state_1);
```

The arguments must match each node's `execute` signature. Call `g.execute()` when no shared arguments are needed. Execution is synchronous; a call runs every node once in dependency order. Subsequent calls run the graph again using its current state.

Use `g.get_node<I>()` to obtain a node pointer and `g.get_state<I, S>()` to obtain its `S`th state pointer. `g.get_state_count<I>()` reports the number of states attached to node `I`; `g.node_count` reports the number of nodes. Out-of-range queries are compile-time errors.

An `sg::exec_graph` can itself appear in another `exec_graph` declaration. It occupies one node position in the parent. The parent's shared arguments and the states attached to that position are passed into the child graph; each child node then receives its own local states. Child node indices start at zero within the child. To query a child, use `parent.get_node<I>()->get_node<J>()` or `parent.get_node<I>()->get_state<J, S>()`.

See [`example/main.cpp`](example/main.cpp) for a nested graph with a shared trace, dependencies, and parent and child states.
