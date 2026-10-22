#pragma once

// sg::exec: Execution resources.

#include "layers.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace sg::exec {

namespace detail::desc {
// Completes inline, on the thread that starts it. Sch is inline_scheduler,
// which is still incomplete here.
template <class Sch> struct inline_schedule {
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t()>;
  sched_attrs<Sch> attrs() const noexcept { return {Sch{}}; }
  template <class T, class In> struct apply : layer<T, apply<T, In>> {
    using base = layer<T, apply<T, In>>;
    template <class... Rest>
    apply(inline_schedule, Rest &&...rest)
        : base{std::forward<Rest>(rest)...} {}
    void start() noexcept { this->pass(set_value_t{}); }
  };
};
} // namespace detail::desc

// Completes schedule() inline, on the thread that starts it.
class inline_scheduler {
public:
  using scheduler_concept = scheduler_t;
  auto schedule() const noexcept {
    return basic_sender{detail::desc::inline_schedule<inline_scheduler>{}};
  }
  friend bool operator==(inline_scheduler, inline_scheduler) noexcept = default;
};

namespace detail {
struct task {
  task *next = nullptr;
  void (*execute)(task *) noexcept = nullptr;
};

class task_queue;

// Chunks [0, count) of a bulk operation as one queued task. Each worker that
// pops it claims the next chunk and queues it again while chunks remain, so
// the job fans out across the workers without a task per chunk.
struct bulk_job : task {
  std::size_t count = 0;
  void (*run_chunk)(bulk_job &, std::size_t) noexcept = nullptr;
  std::atomic<std::size_t> next{0};
  task_queue *queue = nullptr;
};

// A FIFO of intrusive tasks, drained by run() on any number of threads. After
// finish(), queued work still runs, push fails, and run() returns once the
// queue is empty.
class task_queue {
  std::mutex mutex_;
  std::condition_variable ready_;
  task *head_ = nullptr;
  task *tail_ = nullptr;
  bool finishing_ = false;

public:
  std::size_t workers = 1; // threads that run() it

  // Notify under the lock: once run() can observe the change, the owner may be
  // destroyed as soon as the mutex is released.
  bool push(task &job) noexcept {
    std::lock_guard lock(mutex_);
    if (finishing_)
      return false;
    job.next = nullptr;
    if (tail_)
      tail_->next = &job;
    else
      head_ = &job;
    tail_ = &job;
    ready_.notify_one();
    return true;
  }
  bool push(bulk_job &job) noexcept {
    job.queue = this;
    job.next.store(0, std::memory_order_relaxed);
    job.execute = [](task *t) noexcept {
      auto &job = static_cast<bulk_job &>(*t);
      auto run = job.run_chunk;
      std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
      std::size_t end = i + 1;
      // Not queued again: no one else can claim, so run the rest here.
      if (end < job.count && !job.queue->push(static_cast<task &>(job)))
        end = job.count;
      // A claimed chunk keeps job alive until it runs; after the last, job
      // may be gone, so only locals are used.
      for (; i < end; ++i)
        run(job, i);
    };
    return push(static_cast<task &>(job));
  }
  void finish() noexcept {
    std::lock_guard lock(mutex_);
    finishing_ = true;
    ready_.notify_all();
  }
  void run() {
    for (;;) {
      task *job;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return head_ || finishing_; });
        if (!head_)
          return;
        job = head_;
        head_ = job->next;
        if (!head_)
          tail_ = nullptr;
      }
      job->execute(job); // Already unlinked. Completion may destroy job.
    }
  }
};

inline std::exception_ptr closed_error() noexcept {
  try {
    throw std::runtime_error("execution resource is finishing");
  } catch (...) {
    return std::current_exception();
  }
}

namespace desc {
template <class Owner> struct queue_schedule;
}

// The scheduler of an execution resource that runs a task_queue. Owner only
// makes the type distinct.
template <class Owner> class queue_scheduler {
  task_queue *queue_;

public:
  using scheduler_concept = scheduler_t;
  explicit queue_scheduler(task_queue *queue) noexcept : queue_(queue) {}
  basic_sender<desc::queue_schedule<Owner>> schedule() const noexcept {
    return basic_sender{desc::queue_schedule<Owner>{queue_}};
  }
  constexpr forward_progress_guarantee query(
    get_forward_progress_guarantee_t) const noexcept {
    return forward_progress_guarantee::parallel;
  }
  std::size_t query(get_max_concurrency_t) const noexcept {
    return queue_->workers;
  }
  // Runs a parallel bulk's chunks; false if the queue is finishing.
  bool schedule_bulk(bulk_job &job) const noexcept { return queue_->push(job); }
  friend bool operator==(
    const queue_scheduler &, const queue_scheduler &) noexcept = default;
};

namespace desc {
// Queues a task; when it runs, completes with set_stopped if stop was
// requested, else set_value. Fails if the queue is finishing.
template <class Owner> struct queue_schedule {
  task_queue *queue;
  static constexpr bool is_source = true;
  template <class, class>
  using completions = completion_signatures<set_value_t(),
    set_error_t(std::exception_ptr), set_stopped_t()>;
  sched_attrs<queue_scheduler<Owner>> attrs() const noexcept {
    return {queue_scheduler<Owner>{queue}};
  }
  template <class T, class In> struct apply : layer<T, apply<T, In>>, task {
    using base = layer<T, apply<T, In>>;
    task_queue *queue;
    template <class... Rest>
    apply(queue_schedule d, Rest &&...rest)
        : base{std::forward<Rest>(rest)...}, queue(d.queue) {}
    void start() noexcept {
      execute = [](task *t) noexcept {
        auto &self = *static_cast<apply *>(t);
        if (get_stop_token(self.downstream_env()).stop_requested())
          self.pass(set_stopped_t{});
        else
          self.pass(set_value_t{});
      };
      if (!queue->push(*this))
        this->pass(set_error_t{}, closed_error());
    }
  };
};
} // namespace desc
} // namespace detail

// A FIFO execution resource drained by run(). After finish(), queued work
// still runs, and new schedule operations complete with an error.
class run_loop {
  detail::task_queue queue_;

public:
  using scheduler = detail::queue_scheduler<run_loop>;
  run_loop() = default;
  run_loop(const run_loop &) = delete;
  run_loop &operator=(const run_loop &) = delete;

  scheduler get_scheduler() noexcept { return scheduler{&queue_}; }
  void run() { queue_.run(); }
  void finish() noexcept { queue_.finish(); }
};

namespace detail {
struct parallel_resource;
}
// The scheduler of the process-wide pool returned by get_parallel_scheduler().
using parallel_scheduler = detail::queue_scheduler<detail::parallel_resource>;
inline parallel_scheduler get_parallel_scheduler();

// A fixed number of threads draining one shared queue. request_stop() rejects
// new work; the destructor requests stop, runs the queued work and joins.
class static_thread_pool {
  detail::task_queue queue_;
  std::vector<std::thread> threads_;

  void join() noexcept {
    for (auto &t : threads_)
      t.join();
  }

public:
  using scheduler = detail::queue_scheduler<static_thread_pool>;
  explicit static_thread_pool(
    std::size_t threads = std::thread::hardware_concurrency()) {
    threads = std::max<std::size_t>(threads, 1);
    queue_.workers = threads;
    threads_.reserve(threads);
    try {
      while (threads_.size() < threads)
        threads_.emplace_back([this] { queue_.run(); });
    } catch (...) {
      request_stop();
      join();
      throw;
    }
  }
  static_thread_pool(const static_thread_pool &) = delete;
  static_thread_pool &operator=(const static_thread_pool &) = delete;
  ~static_thread_pool() {
    request_stop();
    join();
  }

  scheduler get_scheduler() noexcept { return scheduler{&queue_}; }
  void request_stop() noexcept { queue_.finish(); }
  friend parallel_scheduler get_parallel_scheduler();
};

// A pool with one thread per hardware thread, started on first use.
inline parallel_scheduler get_parallel_scheduler() {
  static static_thread_pool pool;
  return parallel_scheduler{&pool.queue_};
}

} // namespace sg::exec
