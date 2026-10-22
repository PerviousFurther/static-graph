// Mixins stacked by apply_each: each mixin is applied once per apply_each,
// after the mixins it inherits.
// Build: g++ -std=c++20 -I inc tests/core/mixin.cpp -o mixin
#include <sg/core/dag.hpp>
#include <cassert>
#include <cstdio>
#include <string_view>

struct init_t {};
struct expire_t {};
int pass_base_layers = 0;

struct pass_base {
  template <typename T> struct apply : T {
    template <class... A>
    apply(A &&...a) : T{std::forward<A>(a)...} { ++pass_base_layers; }
    int inits = 0;
    template <typename Ctx>
    void on_event(std::in_place_index_t<T::node_index>, init_t, Ctx &) {
      ++inits;
    }
  };
};
// `inherit` is an alias, not a base or member: sized_pass{} stays an
// aggregate.
struct sized_pass {
  using inherit = sg::type_list<pass_base>;
  template <typename T> struct apply : T {
    using T::T;
    using T::on_event;
    int expired = 0;
    template <typename Ctx>
    void on_event(std::in_place_index_t<T::node_index>, expire_t, Ctx &) {
      ++expired;
    }
  };
};
struct named_pass {
  using inherit = sg::type_list<pass_base>;
  template <typename T> struct apply : T {
    using T::T;
    std::string_view label() { return this->get_this_node()->name; }
  };
};
struct framed {
  using inherit = sg::type_list<sized_pass, named_pass>;
  template <typename T> struct apply : T {
    using T::T;
    int frames = 0;
  };
};
// Fails to compile when applied twice: shows a skipped apply is never
// instantiated.
struct once {
  struct mark {};
  template <typename T> struct apply : T, mark {
    static_assert(!std::is_base_of_v<mark, T>, "applied twice");
  };
};

struct tag_state {
  int value;
  template <typename T> struct apply : sg::state::apply<T, apply<T>> {
    using base = typename sg::state::template apply<T, apply<T>>;
    int value;
    template <class... A>
    apply(tag_state s, A &&...a)
        : base{sg::state{}, std::forward<A>(a)...}, value(s.value) {}
  };
};

template <int Id, typename... Ms> struct pass {
  template <typename T>
  struct apply : sg::apply_each<sg::node::apply<T, apply<T>>, Ms...> {
    using base = sg::apply_each<sg::node::apply<T, apply<T>>, Ms...>;
    static constexpr const char *name = Id == 0 ? "a" : Id == 1 ? "b" : "c";
    template <class... A>
    apply(pass, A &&...a) : base{std::forward<A>(a)...} {}
  };
};

using N = sg::node::apply<sg::detail::root>;
template <typename L> using deps = typename L::dependencies;
// Diamond: pass_base is applied once.
static_assert(std::is_same_v<deps<sg::apply_each<N, sized_pass, named_pass>>,
  sg::type_list<pass_base, sized_pass, named_pass>>);
// Deeper: framed's mixins merge with those already applied.
static_assert(std::is_same_v<deps<sg::apply_each<N, named_pass, framed>>,
  sg::type_list<pass_base, named_pass, sized_pass, framed>>);
// One layer per mixin, nothing in between.
static_assert(std::is_same_v<
  sg::detail::apply_each_of<N, named_pass, framed, pass_base>::type,
  framed::apply<sized_pass::apply<named_pass::apply<pass_base::apply<N>>>>>);
// A mixin already applied is skipped without instantiating its apply.
struct uses_once {
  using inherit = sg::type_list<once>;
  template <typename T> struct apply : T {};
};
static_assert(std::is_same_v<deps<sg::apply_each<N, once, uses_once, once>>,
  sg::type_list<once, uses_once>>);
static_assert(std::is_aggregate_v<sized_pass> && std::is_aggregate_v<framed>);

// One generic handler instead of named members: handle receives the CPO as
// the tag for execute, and the event's own tag for on_event.
struct generic {
  template <typename T> struct apply : sg::node::apply<T, apply<T>> {
    using base = sg::node::apply<T, apply<T>>;
    template <class... A>
    apply(generic, A &&...a) : base{std::forward<A>(a)...} {}
    int runs = 0, inits = 0;
    template <typename... A>
    void handle(std::in_place_index_t<base::node_index>, sg::execute_t,
      A &...) {
      ++runs;
    }
    template <typename Ctx>
    void handle(std::in_place_index_t<base::node_index>, init_t, Ctx &) {
      ++inits;
    }
  };
};

int main() {

  sg::dag g{
    pass<0, sized_pass, named_pass>{}, tag_state{1}, tag_state{2},
    pass<1, sized_pass, named_pass>{}, sg::connection<0>{},
    pass<2, named_pass, framed>{},     sg::connection<1>{},
    sg::node{}, sg::node{}};
  assert(pass_base_layers == 3);             // one per node
  assert((g.get_state<0, 1>()->value == 2)); // equal state types both kept
  static_assert(decltype(g)::node_count == 5);
  g.emit(init_t{});
  g.emit_from<1>(expire_t{});
  assert(g.get_node<0>()->inits == 1 && g.get_node<2>()->inits == 1);
  assert(g.get_node<0>()->expired == 0 && g.get_node<1>()->expired == 1);
  assert(g.get_node<2>()->expired == 1 && g.get_node<2>()->frames == 0);
  assert(g.get_node<2>()->label() == "c");
  {
    // handle serves nodes without named members; nodes without either are
    // skipped.
    sg::dag h{generic{}, sg::node{}, pass<0, sized_pass>{}, generic{}};
    h.execute();
    h.emit(init_t{});
    assert(h.get_node<0>()->runs == 1 && h.get_node<3>()->runs == 1);
    assert(h.get_node<0>()->inits == 1 && h.get_node<3>()->inits == 1);
    assert(h.get_node<2>()->inits == 1);
  }
  std::printf("ok\n");
}
