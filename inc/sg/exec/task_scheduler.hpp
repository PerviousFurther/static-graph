#pragma once

// sg::exec: task_scheduler, affine_on.

#include "continues_on.hpp"
#include "sync_wait.hpp"

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace sg::exec {

namespace detail {
// The receiving end of a task_scheduler's wrapped schedule operation.
struct schedule_sink {
  inplace_stop_token token;
  virtual void set_value() noexcept = 0;
  virtual void set_error(std::exception_ptr) noexcept = 0;
  virtual void set_stopped() noexcept = 0;

protected:
  ~schedule_sink() = default;
};

struct sink_receiver {
  using receiver_concept = receiver_t;
  schedule_sink *sink;
  void set_value() && noexcept { sink->set_value(); }
  template <class E> void set_error(E &&e) && noexcept {
    sink->set_error(as_exception_ptr(std::forward<E>(e)));
  }
  void set_stopped() && noexcept { sink->set_stopped(); }
  prop<get_stop_token_t, inplace_stop_token> get_env() const noexcept {
    return {{}, sink->token};
  }
};

struct erased_operation {
  virtual ~erased_operation() = default;
  virtual void start() noexcept = 0;
};

namespace desc {
// Connects the wrapped scheduler's schedule sender on the heap. A stop token
// of the receiver reaches it if it is an inplace_stop_token. Sch is
// task_scheduler, which is still incomplete here.
template <class Sch> struct task_schedule {
  Sch sch;
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t(),
    set_error_t(std::exception_ptr), set_stopped_t()>;
  sched_attrs<Sch> attrs() const noexcept { return {sch}; }
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    struct relay final : schedule_sink {
      apply *self;
      explicit relay(apply *a) noexcept : self(a) {}
      void set_value() noexcept override { self->pass(set_value_t{}); }
      void set_error(std::exception_ptr e) noexcept override {
        self->pass(set_error_t{}, std::move(e));
      }
      void set_stopped() noexcept override { self->pass(set_stopped_t{}); }
    };
    relay sink;
    std::unique_ptr<erased_operation> op;
    template <class... Rest>
    apply(task_schedule d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, sink(this),
          op(d.sch.connect(sink)) {}
    void start() noexcept {
      using token = stop_token_of_t<decltype(this->downstream_env())>;
      if constexpr (std::is_same_v<token, inplace_stop_token>)
        sink.token = get_stop_token(this->downstream_env());
      op->start();
    }
  };
};
} // namespace desc
} // namespace detail

// A scheduler of any type behind one type, shared by copies. Each schedule
// operation connects the wrapped scheduler's on the heap.
class task_scheduler {
  struct model_base {
    virtual ~model_base() = default;
    virtual std::unique_ptr<detail::erased_operation> connect(
      detail::schedule_sink &) const = 0;
    virtual bool equals(const model_base &) const noexcept = 0;
  };
  template <class Sch> struct model final : model_base {
    Sch sch;
    struct operation final : detail::erased_operation {
      connect_result_t<schedule_result_t<const Sch &>, detail::sink_receiver>
        op;
      operation(const Sch &sch, detail::schedule_sink &sink)
          : op(exec::connect(
              exec::schedule(sch), detail::sink_receiver{&sink})) {}
      void start() noexcept override { exec::start(op); }
    };
    explicit model(Sch s) : sch(std::move(s)) {}
    std::unique_ptr<detail::erased_operation> connect(
      detail::schedule_sink &sink) const override {
      return std::make_unique<operation>(sch, sink);
    }
    bool equals(const model_base &other) const noexcept override {
      auto *m = dynamic_cast<const model *>(&other);
      return m && m->sch == sch;
    }
  };

  std::shared_ptr<const model_base> sch_;
  template <class> friend struct detail::desc::task_schedule;
  std::unique_ptr<detail::erased_operation> connect(
    detail::schedule_sink &sink) const {
    return sch_->connect(sink);
  }

public:
  using scheduler_concept = scheduler_t;
  template <class Sch>
    requires(!std::is_same_v<std::remove_cvref_t<Sch>, task_scheduler> &&
      scheduler<Sch>)
  explicit task_scheduler(Sch &&sch)
      : sch_(std::make_shared<const model<std::remove_cvref_t<Sch>>>(
          std::forward<Sch>(sch))) {}
  auto schedule() const noexcept {
    return basic_sender{detail::desc::task_schedule<task_scheduler>{*this}};
  }
  friend bool operator==(
    const task_scheduler &a, const task_scheduler &b) noexcept {
    return a.sch_ == b.sch_ || a.sch_->equals(*b.sch_);
  }
};

// Completes s on sch. inline_scheduler leaves s as it is.
struct affine_on_t : detail::adaptor<affine_on_t> {
  using adaptor::operator();
  template <sender S, scheduler Sch> auto operator()(S &&s, Sch sch) const {
    if constexpr (std::is_same_v<Sch, inline_scheduler>)
      return std::remove_cvref_t<S>(std::forward<S>(s));
    else
      return continues_on(std::forward<S>(s), std::move(sch));
  }
};
inline constexpr affine_on_t affine_on{};

} // namespace sg::exec
