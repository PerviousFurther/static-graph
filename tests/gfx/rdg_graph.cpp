// rdg on a fake backend: a deferred-shading-like graph with connection
// variants, swapchain resize and a nested graph.
#include "fake_backend.hpp"

// Index source standing in for a swapchain. Its extent is its state.
struct swapchain {
  template <typename T> struct apply : pass_node<apply<T>, T, sized_pass> {
    using base = pass_node<apply<T>, T, sized_pass>;
    static constexpr const char *name = "swapchain";
    template <class... A>
    apply(swapchain, A &&...a) : base{std::forward<A>(a)...} {}

    int image = 7; // the acquired image, owned outside the registry

    template <typename B, typename Ctx> void declare(B &b, Ctx &ctx) {
      auto [size] = ctx.states();
      b.import("backbuffer", fake_desc{1, size.value}, image, present);
    }
  };
};

struct gbuffer {
  template <typename T> struct apply : pass_node<apply<T>, T, sized_pass> {
    using base = pass_node<apply<T>, T, sized_pass>;
    static constexpr const char *name = "gbuffer";
    template <class... A>
    apply(gbuffer, A &&...a) : base{std::forward<A>(a)...} {}

    template <typename B, typename Ctx> void declare(B &b, Ctx &ctx) {
      auto size = std::get<0>(ctx.states()).value;
      b.output("albedo", fake_desc{2, size}, color_write);
      b.output("depth", fake_desc{3, size}, depth_write);
    }
  };
};

// Independent of the swapchain: fixed size.
struct shadow {
  template <typename T> struct apply : pass_node<apply<T>, T, pass_base> {
    using base = pass_node<apply<T>, T, pass_base>;
    static constexpr const char *name = "shadow";
    template <class... A>
    apply(shadow, A &&...a) : base{std::forward<A>(a)...} {}

    template <typename B, typename Ctx> void declare(B &b, Ctx &) {
      b.output("shadow", fake_desc{3, {1024, 1024}}, depth_write);
    }
  };
};

struct lighting {
  template <typename T> struct apply : pass_node<apply<T>, T, sized_pass> {
    using base = pass_node<apply<T>, T, sized_pass>;
    static constexpr const char *name = "lighting";
    template <class... A>
    apply(lighting, A &&...a) : base{std::forward<A>(a)...} {}

    template <typename B, typename Ctx> void declare(B &b, Ctx &ctx) {
      b.input("albedo", sampled_fs);
      b.input("depth", sampled_fs);
      b.input("shadow", sampled_fs);
      b.output("hdr", fake_desc{4, std::get<0>(ctx.states()).value},
        color_write);
    }
  };
};

// Compute pass reading hdr alongside tonemap: hdr gets one barrier in front
// of bloom that covers both readers.
struct bloom {
  template <typename T> struct apply : pass_node<apply<T>, T, pass_base> {
    using base = pass_node<apply<T>, T, pass_base>;
    static constexpr const char *name = "bloom";
    template <class... A>
    apply(bloom, A &&...a) : base{std::forward<A>(a)...} {}

    template <typename B, typename Ctx> void declare(B &b, Ctx &) {
      b.input("hdr", sampled_cs);
      b.output("bloom", fake_desc{4, {256, 256}}, storage_cs);
    }
  };
};

// Connection state naming the binding point a tonemap variant writes to.
struct target {
  const char *name;
  template <typename T> struct apply : sg::state::apply<T, apply<T>> {
    using base = typename sg::state::template apply<T, apply<T>>;
    const char *name;
    template <class... A>
    apply(target t, A &&...a)
        : base{sg::state{}, std::forward<A>(a)...}, name(t.name) {}
  };
};

// Two connection variants: present to the backbuffer, or capture offscreen.
// The active variant is node state 1. Listing pass_base again is harmless:
// sized_pass already applied it to this node.
struct tonemap {
  template <typename T>
  struct apply : pass_node<apply<T>, T, sized_pass, pass_base> {
    using base = pass_node<apply<T>, T, sized_pass, pass_base>;
    static constexpr const char *name = "tonemap";
    template <class... A>
    apply(tonemap, A &&...a) : base{std::forward<A>(a)...} {}

    template <typename B, typename Ctx> void declare(B &b, Ctx &ctx) {
      auto states = ctx.states();
      auto &size = std::get<0>(states);
      auto &active = std::get<1>(states);
      b.input("hdr", sampled_fs);
      b.input("bloom", sampled_fs);
      bool ok = ctx.visit_connection(active.value, [&](auto c, auto states) {
        auto &[t] = states;
        auto usage = decltype(c)::value == 0 ? color_write : storage_cs;
        b.output(t.name, fake_desc{1, size.value}, usage);
      });
      SG_RDG_ASSERT(ok, "tonemap: no such variant");
    }
  };
};
static_assert(std::is_same_v<
  pass_node<void, sg::detail::root, sized_pass, pass_base>::dependencies,
  sg::type_list<pass_base, sized_pass>>);

int main() {
  using sg::connection;
  using sg::value_state;
  using sg::rdg::hazard;
  constexpr extent initial{800, 600};

  sg::dag g{
    swapchain{}, value_state{initial},                        // 0
    gbuffer{},   value_state{initial}, connection<0>{},       // 1
    shadow{},                                                 // 2
    lighting{},  value_state{initial}, connection<1, 2>{},    // 3
    bloom{},     connection<3>{},                             // 4
    tonemap{},   value_state{initial}, value_state{0u},       // 5
      connection<3, 4, 0>{}, target{"backbuffer"},
      connection<3, 4>{},    target{"capture"},
  };
  using G = decltype(g);

  // Compile-time order and reachability.
  static_assert(
    std::is_same_v<G::order_type, std::index_sequence<0, 1, 2, 3, 4, 5>>);
  static_assert(G::reaches(0, 3) && G::reaches(0, 5) && !G::reaches(0, 2));
  static_assert(G::reaches(2, 5) && !G::reaches(5, 0));

  fake_backend backend;
  registry r{backend};

  // init: every node declares its slots in execution order.
  sg::rdg::build(g, r);
  CHECK(backend.count("create") == 5); // albedo depth shadow hdr bloom
  CHECK(r.handle("backbuffer") == 7);

  // Passes are numbered in execution order; here that is the node index.
  CHECK(r.pass_count() == 6);
  for (std::size_t i = 0; i < r.pass_count(); ++i)
    CHECK(r.pass_node(i) == i);

  // Edges found from declarations; all agree with the static order.
  for (auto &e : r.edges())
    CHECK(G::reaches(e.from, e.to));
  auto has_edge = [&](std::size_t from, std::size_t to, const char *name,
                    hazard kind) {
    for (auto &e : r.edges())
      if (e.from == from && e.to == to && r.binding_name(e.binding) == name &&
        e.kind == kind)
        return true;
    return false;
  };
  CHECK(has_edge(1, 3, "albedo", hazard::raw));
  CHECK(has_edge(1, 3, "depth", hazard::raw));
  CHECK(has_edge(2, 3, "shadow", hazard::raw));
  CHECK(has_edge(3, 4, "hdr", hazard::raw));
  CHECK(has_edge(3, 5, "hdr", hazard::raw));
  CHECK(has_edge(4, 5, "bloom", hazard::raw));
  CHECK(has_edge(0, 5, "backbuffer", hazard::waw)); // the imported image
  CHECK(r.edges().size() == 7);

  // One batch per pass, each barrier at the latest point. Both reads of hdr
  // share one barrier in front of bloom, the first reader.
  cmd_t cmd;
  sg::rdg::execute(g, r, cmd);
  for (auto &l : cmd)
    std::printf("%s\n", l.c_str());
  const cmd_t expected{
    "pass swapchain",
    "pass gbuffer",
    "pass shadow",
    "barriers 100 color/out->read/fs, 101 depth/out->read/fs, "
    "102 depth/out->read/fs",
    "pass lighting",
    "barriers 103 color/out->read/fs+cs",
    "pass bloom",
    "barriers 7 present/none->color/out, 104 general/cs->read/fs",
    "pass tonemap",
  };
  CHECK(cmd == expected);
  CHECK(r.barriers().size() == 6);

  // Rebuilding with nothing changed keeps every handle.
  sg::rdg::build(g, r);
  CHECK(backend.count("create") == 5 && backend.count("destroy") == 0);

  // swapchain_expire_t reaches the swapchain and its dependents only.
  g.emit_from<0>(sg::rdg::swapchain_expire_t{1280, 720}, r);
  CHECK(g.get_node<0>()->expired == 1 && g.get_node<1>()->expired == 1);
  CHECK(g.get_node<3>()->expired == 1);
  CHECK(g.get_node<5>()->expired == 1);
  CHECK(g.get_state<0, 0>()->value == (extent{1280, 720}));

  // The next build recreates only the resized resources.
  sg::rdg::build(g, r);
  CHECK(backend.count("destroy") == 3); // albedo depth hdr
  CHECK(backend.count("create") == 8);
  CHECK(r.handle("shadow") == 102 && r.handle("bloom") == 104);

  // Switching tonemap to variant 1 writes "capture" instead.
  g.get_state<5, 1>()->value = 1;
  sg::rdg::build(g, r);
  CHECK(r.handle("capture").has_value());
  CHECK(!has_edge(0, 5, "backbuffer", hazard::waw));
  CHECK(has_edge(3, 5, "hdr", hazard::raw));
  // backbuffer is still imported by the swapchain; it is never destroyed.
  CHECK(r.handle("backbuffer") == 7);

  // Nested graphs: child nodes are passes of their own, numbered in the
  // parent's execution order, and get their barriers during execute.
  {
    sg::dag child{
      gbuffer{}, value_state{initial},
      fn_pass{"blit", [](registry::builder &b) {
        b.input("albedo", sampled_fs);
        b.input("backbuffer", color_write);
      }}, connection<0>{},
    };
    sg::dag parent{swapchain{}, value_state{initial}, std::move(child),
      connection<0>{}};
    fake_backend b3;
    registry r3{b3};
    const cmd_t expected{
      "pass swapchain",
      "pass gbuffer",
      "barriers 7 present/none->color/out, 100 color/out->read/fs",
      "pass blit",
    };
    CHECK(run(parent, r3) == expected);
    CHECK(r3.pass_count() == 3);
    // Ids flatten the nested dag: swapchain 0, gbuffer 1, blit 2.
    static_assert(decltype(parent)::id_count == 3);
    CHECK(r3.pass_node(0) == 0 && r3.pass_node(1) == 1);
    parent.emit_from<0>(sg::rdg::swapchain_expire_t{64, 64}, r3);
    CHECK(parent.get_node<1>()->get_node<0>()->expired == 1);
  }

  std::printf("All checks passed.\n");
}
