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

`sg::connection<Is...>{}` says that every listed node must finish before the current node starts. For example, `sg::connection<1, 2>{}` waits for both nodes 1 and 2. Multiple connection descriptors on a node combine their dependencies; `sg::connection<>{}` adds none. The single-index form `sg::connection<I>{}` remains supported. Node indices are zero-based and count only nodes, not states or connections. Dependencies can refer to nodes declared later. The compiler rejects dependencies with an out-of-range index and cycles.

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

## Connection state descriptors

`sg::connection_state` is a structural descriptor, like `sg::node` and
`sg::state`. It adds static indices and lookup to the inheritance chain. Custom
descriptors derive their `apply<T>` from
`sg::connection_state::apply<T, apply<T>>` and decide what data and behavior to
provide. There is no required payload, callback interface, or lifecycle enum.

Place each connection state after its connection, before the next connection
or node. Multiple states may belong to one connection; ordinary node states
may be interleaved without changing the attachment. Each `connection<Is...>`
descriptor is one connection, regardless of the number of dependencies in its
pack. Use `connection<>{}` when a connection needs state but no dependencies.

```cpp
sg::graph g{
    sg::node{}, sg::connection<1, 2>{}, custom_state{}, custom_state{},
                sg::connection<2>{}, custom_state{},
    sg::node{},
    sg::node{}
};
auto* s = g.get_connection_state<0, 0, 1>(); // node 0, connection 0, state 1
```

All indices are zero-based. A custom `apply` can access these protected static
attributes from its `connection_state::apply` base:

| Attribute | Meaning |
| --- | --- |
| `last_connection_state_node_index` | Owning node index |
| `last_connection_state_connection_index` | Connection index within that node |
| `connection_state_index` | State index within that connection |

`g.get_connection_count<N>()` counts connection descriptors on node `N`.
`g.get_connection_indices<N, C>()` returns that connection's dependency indices
as an `std::index_sequence`. `g.get_connection_state_count<N, C>()` counts its
states, and `g.get_connection_state<N, C, S>()` returns a concrete state pointer
(const for a const graph). Invalid queries and states without a preceding
connection on the same node are compile-time errors. Nested executable graphs
keep their own indices; query them through `parent.get_node<I>()`.

Connection states do not change `exec_graph::execute`: execution still runs
every node in dependency order and passes only shared and node-local states to
nodes. Activation, gating, initialization, and lifecycle behavior are policies
that custom descriptors or external functions can implement.

Standalone C++20 examples:

- [`connection_constructor.cpp`](example/connection_constructor.cpp) injects
  constructor arguments through the base layers into the owning node.
- [`connection_function.cpp`](example/connection_function.cpp) forwards a
  function argument from a connection state to its owning node.
- [`connection_activation.cpp`](example/connection_activation.cpp) owns mutable
  state and dispatches an externally selected connection to its states.

For example:

```sh
g++ -std=c++20 example/connection_activation.cpp -o connection_activation
./connection_activation
```

## Code style

Use the repository's `.clang-format`: LLVM-based, two-space indentation and
continuation indentation, with `DontAlign` for bracket continuations and
operands. Consecutive assignments, declarations, and trailing comments are not
column-aligned. Formatting is verified with clang-format 18.1.8.

```sh
clang-format -i inc/graph.hpp example/*.cpp tests/*.cpp
clang-format --dry-run --Werror inc/graph.hpp example/*.cpp tests/*.cpp
```
