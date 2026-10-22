// rdg dependency rules on a fake backend: WAR, read-write passes and
// rejected declarations.
#include "fake_backend.hpp"

int main() {
  using sg::connection;
  using sg::rdg::hazard;

  // WAR: the next writer waits for every reader with one barrier.
  {
    sg::dag w{
      fn_pass{"w0", [](registry::builder &b) {
        b.output("x", fake_desc{}, color_write);
      }},
      fn_pass{"a", [](registry::builder &b) { b.input("x", sampled_fs); }},
        connection<0>{},
      fn_pass{"b", [](registry::builder &b) { b.input("x", sampled_cs); }},
        connection<0>{},
      fn_pass{"w1", [](registry::builder &b) {
        b.output("x", fake_desc{}, color_write);
      }}, connection<1, 2>{},
    };
    fake_backend b2;
    registry r2{b2};
    const cmd_t expected{
      "pass w0",
      "barriers 100 color/out->read/fs+cs",
      "pass a",
      "pass b",
      "barriers 100 read/fs+cs->color/out",
      "pass w1",
    };
    CHECK(run(w, r2) == expected);
    CHECK(r2.barriers()[1].kind == hazard::war);
    CHECK(r2.edges().size() == 4);
  }

  // A pass that reads and writes one binding point: its write usage is part
  // of the read barrier, and there is no WAR barrier on itself.
  {
    sg::dag w{
      fn_pass{"w0", [](registry::builder &b) {
        b.output("x", fake_desc{}, storage_cs);
      }},
      fn_pass{"rw", [](registry::builder &b) {
        b.input("x", storage_cs);
        b.output("x", fake_desc{}, storage_cs);
      }}, connection<0>{},
    };
    fake_backend b2;
    registry r2{b2};
    const cmd_t expected{
      "pass w0",
      "barriers 100 general/cs->general/cs",
      "pass rw",
    };
    CHECK(run(w, r2) == expected);
    CHECK(r2.edges().size() == 1);
  }

  // Readers that one barrier cannot cover are rejected when the build ends.
  {
    sg::dag w{
      fn_pass{"w0", [](registry::builder &b) {
        b.output("x", fake_desc{}, color_write);
      }},
      fn_pass{"a", [](registry::builder &b) { b.input("x", sampled_fs); }},
        connection<0>{},
      fn_pass{"b", [](registry::builder &b) { b.input("x", storage_cs); }},
        connection<0>{},
    };
    CHECK(build_throws(w));
  }

  // An input without an earlier producer is rejected during init.
  {
    sg::dag w{fn_pass{"orphan", [](registry::builder &b) {
      b.input("missing", sampled_fs);
    }}};
    CHECK(build_throws(w));
  }

  std::printf("All checks passed.\n");
}
