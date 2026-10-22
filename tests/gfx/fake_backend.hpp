#pragma once

// A fake rdg backend: handles are integers and every backend call is logged,
// so tests check what a Vulkan backend would be asked to do.
#include <cstdio>
#include <stdexcept>

#define SG_RDG_ASSERT(cond, msg)                                               \
  do {                                                                         \
    if (!(cond))                                                               \
      throw std::logic_error(msg);                                             \
  } while (false)
#include <sg/gfx/rdg.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#define CHECK(...)                                                             \
  do {                                                                         \
    if (!(__VA_ARGS__)) {                                                      \
      std::fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #__VA_ARGS__);    \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

struct extent {
  std::uint32_t width, height;
  bool operator==(const extent &) const = default;
};

struct fake_desc {
  int format = 0;
  extent size{};
  bool operator==(const fake_desc &) const = default;
};

// Stands in for layout and stage/access masks. Accesses in one layout merge
// by combining their stages, as one Vulkan barrier combines stage masks.
enum class fake_layout { color, depth, read, general, present };
enum fake_stage : unsigned { none = 0, out = 1, fs = 2, cs = 4 };
struct fake_usage {
  fake_layout layout;
  unsigned stages;
  bool operator==(const fake_usage &) const = default;
};
constexpr fake_usage color_write{fake_layout::color, out};
constexpr fake_usage depth_write{fake_layout::depth, out};
constexpr fake_usage sampled_fs{fake_layout::read, fs};
constexpr fake_usage sampled_cs{fake_layout::read, cs};
constexpr fake_usage storage_cs{fake_layout::general, cs};
constexpr fake_usage present{fake_layout::present, none};

inline std::string to_string(fake_usage u) {
  static const char *layouts[] = {"color", "depth", "read", "general",
    "present"};
  std::string s = std::string(layouts[int(u.layout)]) + "/";
  if (!u.stages)
    return s + "none";
  const char *names[] = {"out", "fs", "cs"};
  bool first = true;
  for (int i = 0; i < 3; ++i)
    if (u.stages & (1u << i)) {
      s += (first ? "" : "+") + std::string(names[i]);
      first = false;
    }
  return s;
}

struct fake_backend {
  using desc_type = fake_desc;
  using usage_type = fake_usage;
  using handle_type = int;

  int next = 100;
  std::vector<std::string> log;

  int create(const fake_desc &d) {
    log.push_back("create " + std::to_string(next) + " " +
      std::to_string(d.size.width) + "x" + std::to_string(d.size.height));
    return next++;
  }
  void destroy(int h) { log.push_back("destroy " + std::to_string(h)); }
  std::optional<fake_usage> merge(const fake_usage &a, const fake_usage &b) {
    if (a.layout != b.layout)
      return std::nullopt;
    return fake_usage{a.layout, a.stages | b.stages};
  }
  // One command per batch, like one vkCmdPipelineBarrier2.
  void barriers(std::vector<std::string> &cmd,
    std::span<const sg::rdg::transition<int, fake_usage>> ts) {
    std::string line = "barriers";
    for (auto &t : ts)
      line += (&t == ts.data() ? " " : ", ") + std::to_string(t.handle) +
        " " + to_string(t.src) + "->" + to_string(t.dst);
    cmd.push_back(line);
  }

  std::size_t count(std::string_view prefix) const {
    std::size_t n = 0;
    for (auto &l : log)
      n += l.starts_with(prefix);
    return n;
  }
};

using registry = sg::rdg::registry<fake_backend>;
using cmd_t = std::vector<std::string>;

// Mixin: the owning node's declare runs on init; execute logs its name.
struct pass_base {
  template <typename T> struct apply : T {
    using T::T;
    template <typename Ctx>
    void on_event(std::in_place_index_t<T::node_index>, sg::rdg::init_t,
      Ctx &ctx, registry &r) {
      auto b = r.declare(ctx);
      this->get_this_node()->declare(b, ctx);
    }
    template <typename... States>
    void execute(std::in_place_index_t<T::node_index>, cmd_t &cmd,
      registry &, States &...) {
      cmd.push_back(std::string("pass ") + this->get_this_node()->name);
    }
  };
};

// Mixin on top of pass_base: follows the swapchain extent, kept in node
// state 0.
struct sized_pass {
  using inherit = sg::type_list<pass_base>;
  template <typename T> struct apply : T {
    using T::T;
    using T::on_event;
    int expired = 0;
    template <typename Ctx>
    void on_event(std::in_place_index_t<T::node_index>,
      const sg::rdg::swapchain_expire_t &e, Ctx &ctx, registry &) {
      ++expired;
      std::get<0>(ctx.states()).value = extent{e.width, e.height};
    }
  };
};

// A pass node with mixins Ms; Self supplies the name and declare.
template <typename Self, typename T, typename... Ms>
using pass_node = sg::apply_each<sg::node::apply<T, Self>, Ms...>;

// A pass whose declarations are a plain function, for small graphs.
using declare_fn = void (*)(registry::builder &);
struct fn_pass {
  const char *name;
  declare_fn fn;
  template <typename T> struct apply : pass_node<apply<T>, T, pass_base> {
    using base = pass_node<apply<T>, T, pass_base>;
    const char *name;
    declare_fn fn;
    template <class... A>
    apply(fn_pass p, A &&...a)
        : base{std::forward<A>(a)...}, name(p.name), fn(p.fn) {}

    template <typename B, typename Ctx> void declare(B &b, Ctx &) { fn(b); }
  };
};

// Builds g and runs it once; returns the recorded commands.
template <typename G> cmd_t run(G &g, registry &r) {
  sg::rdg::build(g, r);
  cmd_t cmd;
  sg::rdg::execute(g, r, cmd);
  return cmd;
}

// Whether building g fails an SG_RDG_ASSERT.
template <typename G> bool build_throws(G &g) {
  fake_backend b;
  registry r{b};
  try {
    sg::rdg::build(g, r);
  } catch (const std::logic_error &) {
    return true;
  }
  return false;
}
