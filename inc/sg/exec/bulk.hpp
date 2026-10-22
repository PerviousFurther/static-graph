#pragma once

// sg::exec: bulk, bulk_chunked, bulk_unchunked.

#include "../core/graph.hpp"
#include "schedulers.hpp"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <exception>
#include <execution>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sg::exec {

namespace detail {
template <class P>
concept execution_policy = std::is_execution_policy_v<std::remove_cvref_t<P>>;
template <class P>
inline constexpr bool parallel_policy =
  std::is_same_v<std::remove_cvref_t<P>, std::execution::parallel_policy> ||
  std::is_same_v<std::remove_cvref_t<P>,
    std::execution::parallel_unsequenced_policy>;

// bulk with a parallel policy splits its work across these schedulers.
template <class Sch>
concept splits_bulk = requires(const Sch &sch) { get_max_concurrency(sch); };
// Those that also run the chunks themselves; the others get a schedule
// operation per chunk.
template <class Sch>
concept runs_bulk = requires(const Sch &sch, bulk_job &job) {
  { sch.schedule_bulk(job) } noexcept -> std::same_as<bool>;
};

enum class bulk_kind { each, chunked, unchunked };
struct no_scheduler {};

// Adapts the user function to run over a range [b, e).
template <bulk_kind Kind, class Shape, class F> struct bulk_function {
  F function;
  template <class... Vs>
  static constexpr bool nothrow = Kind == bulk_kind::chunked
    ? std::is_nothrow_invocable_v<F &, Shape, Shape, Vs &...>
    : std::is_nothrow_invocable_v<F &, Shape, Vs &...>;
  template <class... Vs> void operator()(Shape b, Shape e, Vs &...vs) {
    if constexpr (Kind == bulk_kind::chunked)
      std::invoke(function, b, e, vs...);
    else
      for (; b != e; ++b)
        std::invoke(function, b, vs...);
  }
};

template <class Sch, class Env> struct bulk_schedule_sigs {
  using type = concat_sigs_t<exception_sigs<false>,
    sig_map_t<completion_signatures_of_t<schedule_result_t<Sch &>, Env>,
      non_value_sig>>;
};
template <class Env> struct bulk_schedule_sigs<no_scheduler, Env> {
  using type = type_list<>;
};

namespace desc {
// Runs the function over [0, shape) on each value completion, then passes
// the values on. With a scheduler (Sch is not no_scheduler) the values are
// stored, the range is split into chunks that run on it, and the last chunk
// to finish completes. Otherwise the range runs inline.
template <bulk_kind Kind, class Shape, class F, class Sch> struct bulk {
  using fn = bulk_function<Kind, Shape, F>;
  static constexpr bool parallel = !std::is_same_v<Sch, no_scheduler>;
  template <class A>
  using arg_t =
    std::conditional_t<parallel, std::decay_t<A>, std::remove_reference_t<A>>;

  Shape shape;
  fn function;
  [[no_unique_address]] Sch sch;
  static constexpr bool is_source = false;

  template <class Sig> struct sig_of {
    using type = completion_signatures<Sig>;
  };
  template <class... As> struct sig_of<set_value_t(As...)> {
    using type = concat_sigs_t<completion_signatures<set_value_t(arg_t<As>...)>,
      exception_sigs<fn::template nothrow<arg_t<As>...>>>;
  };
  template <class In, class Env>
  using completions = concat_sigs_t<sig_map_t<In, sig_of>,
    typename bulk_schedule_sigs<Sch, Env>::type>;

  template <class T, class In>
  struct inline_layer : layer<T, inline_layer<T, In>> {
    using base = layer<T, inline_layer<T, In>>;
    Shape shape;
    fn function;
    template <class... Rest>
    inline_layer(bulk d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, shape(d.shape),
          function(std::move(d.function)) {}
    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, set_value_t>)
        this->pass(C{}, std::forward<As>(as)...);
      else if (this->template guard<fn::template nothrow<arg_t<As>...>>(
                 [&] { function(Shape(0), shape, as...); }))
        this->pass(set_value_t{}, std::forward<As>(as)...);
    }
  };

  template <class T, class In>
  struct parallel_layer : layer<T, parallel_layer<T, In>> {
    using base = layer<T, parallel_layer<T, In>>;
    using outer_env = env_of_t<const T &>;
    enum : int { running, failed, stopped };

    struct chunk_receiver {
      using receiver_concept = receiver_t;
      parallel_layer *self;
      std::size_t index;
      void set_value() && noexcept { self->run_chunk(index); }
      template <class E> void set_error(E &&e) && noexcept {
        self->fail(std::forward<E>(e));
        self->arrive();
      }
      void set_stopped() && noexcept { self->stop_chunk(); }
      outer_env get_env() const noexcept { return self->downstream_env(); }
    };
    struct slot {
      connect_result_t<schedule_result_t<Sch &>, chunk_receiver> op;
      slot(Sch &sch, chunk_receiver r)
          : op(exec::connect(exec::schedule(sch), r)) {}
    };
    struct job : bulk_job {
      parallel_layer *self;
    };
    using chunks_type =
      std::conditional_t<runs_bulk<Sch>, job, std::vector<std::optional<slot>>>;
    using errors_type = gather_signatures_t<set_error_t,
      concat_sigs_t<
        sig_map_t<completion_signatures_of_t<schedule_result_t<Sch &>,
                    std::decay_t<outer_env>>,
          error_sig>,
        exception_sigs<false>>,
      single_t, monostate_variant>;

    Sch sch;
    Shape shape;
    fn function;
    std::size_t chunks = 0;
    std::atomic<int> state{running};
    std::atomic<std::size_t> pending{0};
    gather_signatures_t<set_value_t, In, decayed_tuple, monostate_variant>
      values;
    errors_type errors;
    chunks_type tasks;

    template <class... Rest>
    parallel_layer(bulk d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, sch(std::move(d.sch)),
          shape(d.shape), function(std::move(d.function)) {
      if constexpr (runs_bulk<Sch>) {
        tasks.self = this;
        tasks.run_chunk = [](bulk_job &j, std::size_t i) noexcept {
          auto &self = *static_cast<job &>(j).self;
          if (get_stop_token(self.downstream_env()).stop_requested())
            self.stop_chunk();
          else
            self.run_chunk(i);
        };
      }
    }

    template <class C, class... As> void complete(C, As &&...as) noexcept {
      if constexpr (!std::is_same_v<C, set_value_t>) {
        this->pass(C{}, std::forward<As>(as)...);
      } else {
        if (!this->template guard<false>([&] {
              values.template emplace<decayed_tuple<As...>>(
                std::forward<As>(as)...);
              chunks = chunk_count();
              if constexpr (!runs_bulk<Sch>)
                if (chunks > 1) {
                  tasks = std::vector<std::optional<slot>>(chunks);
                  for (std::size_t i = 0; i < chunks; ++i)
                    tasks[i].emplace(sch, chunk_receiver{this, i});
                }
            }))
          return;
        if (chunks <= 1) {
          run_range(Shape(0), shape);
          finish();
          return;
        }
        // One extra arrival keeps the operation alive through all starts.
        pending.store(chunks + 1, std::memory_order_relaxed);
        if constexpr (runs_bulk<Sch>) {
          tasks.count = chunks;
          if (!sch.schedule_bulk(tasks)) { // No chunk will run.
            fail(closed_error());
            pending.store(1, std::memory_order_relaxed);
          }
        } else {
          for (auto &s : tasks)
            exec::start(s->op);
        }
        arrive(); // Last access: finish may destroy the whole graph.
      }
    }

  private:
    std::size_t chunk_count() const noexcept {
      if (shape <= Shape(0))
        return 0;
      auto n = static_cast<std::size_t>(shape);
      if constexpr (Kind == bulk_kind::unchunked)
        return n;
      else
        return std::min(n, get_max_concurrency(sch));
    }

    template <class Fn> void visit_values(Fn &&fn) noexcept {
      std::visit(
        [&](auto &v) noexcept {
          if constexpr (!std::is_same_v<std::decay_t<decltype(v)>,
                          std::monostate>)
            std::apply(fn, v);
        },
        values);
    }

    void run_range(Shape b, Shape e) noexcept {
      visit_values([&](auto &...vs) noexcept {
        try_invoke<
          fn::template nothrow<std::remove_reference_t<decltype(vs)>...>>(
          [&] { function(b, e, vs...); },
          [&](auto ep) noexcept { fail(std::move(ep)); });
      });
    }

    // Chunk i of [0, shape); the first shape % chunks chunks get one more.
    void run_chunk(std::size_t i) noexcept {
      if (state.load(std::memory_order_relaxed) == running) {
        auto n = static_cast<std::size_t>(shape);
        std::size_t size = n / chunks, extra = n % chunks;
        std::size_t b = i * size + std::min(i, extra);
        run_range(Shape(b), Shape(b + size + (i < extra)));
      }
      arrive();
    }

    void stop_chunk() noexcept {
      int expected = running;
      state.compare_exchange_strong(
        expected, stopped, std::memory_order_acq_rel);
      arrive();
    }

    template <class E> void fail(E &&e) noexcept {
      if (state.exchange(failed, std::memory_order_acq_rel) == failed)
        return;
      using error = std::decay_t<E>;
      try_invoke<std::is_nothrow_constructible_v<error, E>>(
        [&] { errors.template emplace<error>(std::forward<E>(e)); },
        [this](auto ep) noexcept {
          errors.template emplace<std::exception_ptr>(std::move(ep));
        });
    }

    void arrive() noexcept {
      if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
        finish();
    }

    void finish() noexcept {
      switch (state.load(std::memory_order_relaxed)) {
      case running:
        visit_values([this](auto &...vs) noexcept {
          this->pass(set_value_t{}, std::move(vs)...);
        });
        break;
      case failed:
        std::visit(
          [this](auto &e) noexcept {
            if constexpr (!std::is_same_v<std::decay_t<decltype(e)>,
                            std::monostate>)
              this->pass(set_error_t{}, std::move(e));
          },
          errors);
        break;
      default:
        this->pass(set_stopped_t{});
      }
    }
  };

  template <class T, class In>
  using apply =
    std::conditional_t<parallel, parallel_layer<T, In>, inline_layer<T, In>>;
};
} // namespace desc

// The scheduler a parallel bulk runs on: where s completes, if that resource
// is parallel and P is a parallel policy.
template <class P, class S> auto bulk_scheduler_of(const S &s) noexcept {
  if constexpr (parallel_policy<P> &&
    requires { get_completion_scheduler<set_value_t>(exec::get_env(s)); }) {
    auto sch = get_completion_scheduler<set_value_t>(exec::get_env(s));
    if constexpr (splits_bulk<decltype(sch)>)
      return sch;
    else
      return no_scheduler{};
  } else {
    return no_scheduler{};
  }
}

template <bulk_kind Kind> struct bulk_adaptor : adaptor<bulk_adaptor<Kind>> {
  using adaptor<bulk_adaptor>::operator();
  template <sender S, execution_policy P, std::integral Shape, class F>
  auto operator()(S &&s, P &&, Shape shape, F f) const {
    auto sch = bulk_scheduler_of<P>(s);
    return append(std::forward<S>(s),
      desc::bulk<Kind, Shape, F, decltype(sch)>{
        shape, {std::move(f)}, std::move(sch)});
  }
};
} // namespace detail

using bulk_t = detail::bulk_adaptor<detail::bulk_kind::each>;
using bulk_chunked_t = detail::bulk_adaptor<detail::bulk_kind::chunked>;
using bulk_unchunked_t = detail::bulk_adaptor<detail::bulk_kind::unchunked>;
inline constexpr bulk_t bulk{};
inline constexpr bulk_chunked_t bulk_chunked{};
inline constexpr bulk_unchunked_t bulk_unchunked{};

} // namespace sg::exec
