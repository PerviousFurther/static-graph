#pragma once

// Render dependency graph on top of dag.hpp. The dag fixes the execution
// order at compile time; resource edges are found at run time, when the init
// protocol walks the nodes in that order and each node declares its slots.
//
// The registry does not know any graphics API. A backend supplies:
//   using desc_type;    // resource description, equality comparable
//   using usage_type;   // how a slot accesses a resource
//   using handle_type;
//   handle_type create(const desc_type &);
//   void destroy(handle_type);
//   // Combined usage of two accesses covered by one barrier, or nullopt when
//   // one barrier cannot cover both, e.g. different image layouts.
//   std::optional<usage_type> merge(const usage_type &, const usage_type &);
//   // Records the barriers in front of one pass; only for execute.
//   void barriers(Command &, std::span<const transition<handle_type,
//                 usage_type>>);

#include "../core/dag.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef SG_RDG_ASSERT
#define SG_RDG_ASSERT(cond, msg) assert((cond) && (msg))
#endif

namespace sg::rdg {

// Protocols. A tag may carry the protocol's own parameters.

// Delivered to every node, in execution order, by build(g, registry).
// Handler: on_event(init_t, Ctx &, registry<B> &).
struct init_t {};
inline constexpr init_t init{};

// Delivered by the swapchain-owning node and its dependents through
// g.emit_from<N>(swapchain_expire_t{...}, ...).
struct swapchain_expire_t {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

// The hazard a dependency resolves.
enum class hazard { raw, war, waw };

template <typename Handle, typename Usage> struct transition {
  Handle handle;
  Usage src;
  Usage dst;
};

template <typename Backend> class registry {
public:
  using desc_type = typename Backend::desc_type;
  using usage_type = typename Backend::usage_type;
  using handle_type = typename Backend::handle_type;
  using transition_type = transition<handle_type, usage_type>;

  // Passes are numbered by declaration, which follows the execution order,
  // nested graphs included.
  struct edge {
    std::size_t from;
    std::size_t to;
    std::size_t binding;
    hazard kind;
  };
  // Recorded in front of pass `pass`, at the latest point.
  struct barrier {
    std::size_t pass;
    std::size_t binding;
    hazard kind;
    usage_type src;
    usage_type dst;
  };

  explicit registry(Backend &backend) : backend_(backend) {}
  registry(const registry &) = delete;
  registry &operator=(const registry &) = delete;
  ~registry() {
    for (auto &b : bindings_)
      release(b);
  }

  // Declarations of one pass during the init protocol.
  class builder {
  public:
    // Writes a new version of `name`. Creates the resource on first use and
    // recreates it when desc changes between builds.
    handle_type output(
      std::string_view name, const desc_type &desc, const usage_type &usage) {
      auto &b = r_.find_or_add(name);
      if (b.declared) {
        SG_RDG_ASSERT(b.desc == desc,
          "rdg: output desc differs from the binding point's resource");
      } else {
        b.declared = true;
        if (b.handle && (b.imported || !(b.desc == desc)))
          r_.release(b);
        b.desc = desc;
      }
      if (!b.handle)
        b.handle = r_.backend_.create(desc);
      r_.write(b, pass_, usage);
      return *b.handle;
    }

    // Provides an externally owned resource, such as a swapchain image, as
    // the first version of `name`. The registry never destroys it.
    void import(std::string_view name, const desc_type &desc,
      handle_type handle, const usage_type &usage) {
      auto &b = r_.find_or_add(name);
      SG_RDG_ASSERT(!b.declared, "rdg: import of a declared binding point");
      if (b.handle && !b.imported)
        r_.release(b);
      b.declared = true;
      b.imported = true;
      b.desc = desc;
      b.handle = handle;
      r_.write(b, pass_, usage);
    }

    // Reads the latest version of `name`, which an earlier pass must have
    // written. Reading the same version twice merges the usages.
    handle_type input(std::string_view name, const usage_type &usage) {
      auto *b = r_.find(name);
      SG_RDG_ASSERT(b && b->declared && !b->versions.empty(),
        "rdg: input has no producer earlier in the execution order");
      auto &v = b->versions.back();
      SG_RDG_ASSERT(v.producer != pass_, "rdg: a pass reads its own output");
      if (!v.readers.empty() && v.readers.back().pass == pass_)
        v.readers.back().usage = r_.merge(v.readers.back().usage, usage);
      else
        v.readers.push_back(access{pass_, usage});
      return *b->handle;
    }

  private:
    friend class registry;
    builder(registry &r, std::size_t pass) : r_(r), pass_(pass) {}
    registry &r_;
    std::size_t pass_;
  };

  // Called by a node during the init protocol, usually as declare(ctx).
  builder declare(std::size_t node) {
    auto pass = this->pass(node);
    if (!pass) {
      pass = passes_.size();
      passes_.push_back(node);
    }
    return builder{*this, *pass};
  }
  template <std::size_t I, typename G, std::size_t Base>
  builder declare(const node_context<I, G, Base> &ctx) {
    return declare(ctx.node_id);
  }

  // Starts a build. Resources survive until end() so that unchanged
  // declarations keep their handles.
  void begin() {
    passes_.clear();
    edges_.clear();
    barriers_.clear();
    for (auto &b : bindings_) {
      b.declared = false;
      b.versions.clear();
    }
  }
  // Releases resources that the build did not declare, then turns the
  // recorded accesses into edges and barriers.
  void end() {
    for (auto &b : bindings_)
      if (!b.declared)
        release(b);
    for (const auto &b : bindings_)
      for (std::size_t i = 0; i < b.versions.size(); ++i)
        resolve(b.index, b.versions[i],
          i + 1 < b.versions.size() ? &b.versions[i + 1] : nullptr);
    std::stable_sort(barriers_.begin(), barriers_.end(),
      [](const barrier &x, const barrier &y) { return x.pass < y.pass; });
    first_barrier_.assign(passes_.size() + 1, 0);
    for (const auto &x : barriers_)
      ++first_barrier_[x.pass + 1];
    for (std::size_t i = 1; i < first_barrier_.size(); ++i)
      first_barrier_[i] += first_barrier_[i - 1];
  }

  std::size_t pass_count() const noexcept { return passes_.size(); }
  std::size_t pass_node(std::size_t pass) const { return passes_[pass]; }
  std::optional<std::size_t> pass(std::size_t node) const {
    auto it = std::find(passes_.begin(), passes_.end(), node);
    if (it == passes_.end())
      return std::nullopt;
    return std::size_t(it - passes_.begin());
  }

  const std::vector<edge> &edges() const noexcept { return edges_; }
  const std::vector<barrier> &barriers() const noexcept { return barriers_; }

  std::optional<handle_type> handle(std::string_view name) const {
    auto *b = find(name);
    if (!b || !b->declared)
      return std::nullopt;
    return b->handle;
  }
  std::optional<std::size_t> binding_index(std::string_view name) const {
    auto *b = find(name);
    if (!b)
      return std::nullopt;
    return b->index;
  }
  const std::string &binding_name(std::size_t index) const {
    return bindings_[index].name;
  }

  // Records the barriers in front of `node` as one batch. A node that did
  // not declare anything has none.
  template <typename Command>
  void record_barriers(Command &cmd, std::size_t node) {
    auto pass = this->pass(node);
    if (!pass || first_barrier_[*pass] == first_barrier_[*pass + 1])
      return;
    scratch_.clear();
    for (auto i = first_barrier_[*pass]; i < first_barrier_[*pass + 1]; ++i) {
      const auto &x = barriers_[i];
      scratch_.push_back(
        transition_type{*bindings_[x.binding].handle, x.src, x.dst});
    }
    backend_.barriers(cmd, std::span<const transition_type>(scratch_));
  }

private:
  struct access {
    std::size_t pass;
    usage_type usage;
  };
  struct version {
    std::size_t producer;
    usage_type usage;
    std::vector<access> readers; // in pass order, one per pass
  };
  struct binding {
    std::string name;
    std::size_t index;
    desc_type desc{};
    std::optional<handle_type> handle;
    bool imported = false;
    bool declared = false;
    std::vector<version> versions;
  };

  binding *find(std::string_view name) {
    auto it = names_.find(name);
    return it == names_.end() ? nullptr : &bindings_[it->second];
  }
  const binding *find(std::string_view name) const {
    auto it = names_.find(name);
    return it == names_.end() ? nullptr : &bindings_[it->second];
  }
  binding &find_or_add(std::string_view name) {
    if (auto *b = find(name))
      return *b;
    auto &b = bindings_.emplace_back();
    b.name = name;
    b.index = bindings_.size() - 1;
    names_.emplace(b.name, b.index);
    return b;
  }

  usage_type merge(const usage_type &a, const usage_type &b) {
    auto m = backend_.merge(a, b);
    SG_RDG_ASSERT(m.has_value(), "rdg: one barrier cannot cover both usages");
    return m ? *m : a;
  }

  void write(binding &b, std::size_t pass, const usage_type &usage) {
    SG_RDG_ASSERT(b.versions.empty() || b.versions.back().producer != pass,
      "rdg: a pass writes a binding point twice");
    b.versions.push_back(version{pass, usage, {}});
  }

  // Dependencies of version v, followed by version next if any:
  //  - RAW: one barrier in front of the first reader, covering every reader.
  //    A pass that reads v and writes next is a reader whose write usage is
  //    covered too.
  //  - WAR: one barrier in front of the next writer, from every other reader.
  //  - WAW: when nobody reads v.
  void resolve(std::size_t binding, const version &v, const version *next) {
    const auto &readers = v.readers;
    if (!readers.empty()) {
      auto dst = readers.front().usage;
      for (const auto &x : readers) {
        edges_.push_back(edge{v.producer, x.pass, binding, hazard::raw});
        if (&x != &readers.front())
          dst = merge(dst, x.usage);
      }
      if (next && next->producer == readers.front().pass)
        dst = merge(dst, next->usage);
      barriers_.push_back(
        barrier{readers.front().pass, binding, hazard::raw, v.usage, dst});
    }
    if (!next)
      return;
    if (readers.empty()) {
      edges_.push_back(edge{v.producer, next->producer, binding, hazard::waw});
      barriers_.push_back(
        barrier{next->producer, binding, hazard::waw, v.usage, next->usage});
      return;
    }
    std::optional<usage_type> src;
    for (const auto &x : readers) {
      if (x.pass == next->producer)
        continue;
      edges_.push_back(edge{x.pass, next->producer, binding, hazard::war});
      src = src ? merge(*src, x.usage) : x.usage;
    }
    if (src)
      barriers_.push_back(
        barrier{next->producer, binding, hazard::war, *src, next->usage});
  }

  void release(binding &b) {
    if (b.handle && !b.imported)
      backend_.destroy(*b.handle);
    b.handle.reset();
    b.imported = false;
  }

  Backend &backend_;
  std::vector<binding> bindings_;
  std::map<std::string, std::size_t, std::less<>> names_;
  std::vector<std::size_t> passes_;  // node ids
  std::vector<edge> edges_;
  std::vector<barrier> barriers_;           // sorted by pass after end()
  std::vector<std::size_t> first_barrier_;  // per pass, into barriers_
  std::vector<transition_type> scratch_;
};

// Runs the init protocol over the whole graph.
template <typename G, typename Backend>
void build(G &g, registry<Backend> &r) {
  r.begin();
  g.emit(init, r);
  r.end();
}

// Records each pass's barriers, then calls node.execute(std::in_place_index<I>,
// cmd, registry, args..., node_states...), in execution order.
template <typename G, typename Backend, typename Command, typename... Args>
void execute(G &g, registry<Backend> &r, Command &cmd, Args &...args) {
  sg::enumerate(g, [&](const auto &ctx, const auto &...outer) {
    r.record_barriers(cmd, ctx.node_id);
    sg::detail::invoke_execute(std::tie(cmd, r, args...), ctx, outer...);
  });
}

} // namespace sg::rdg
