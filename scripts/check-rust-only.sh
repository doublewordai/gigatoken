#!/usr/bin/env bash
set -euo pipefail

cargo check --locked --no-default-features
cargo test --locked --no-default-features --no-run

if cargo tree --locked --no-default-features --edges normal | grep -Eq '(^|[[:space:]])(pyo3|numpy) v'; then
  echo "Rust-only dependency graph contains Python bindings" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Linux" ]]; then
  cargo build --locked --release --no-default-features
  library="${CARGO_TARGET_DIR:-target}/release/libgigatoken_rs.so"

  if ldd "$library" | grep -qi python; then
    echo "Rust-only shared library links to Python" >&2
    exit 1
  fi
fi
