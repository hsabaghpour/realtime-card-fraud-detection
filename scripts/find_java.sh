#!/usr/bin/env bash
# Print a JAVA_HOME that Spark 4 supports (Java 17 or 21), or fail with install advice.
# Order: an already-set JAVA_HOME, Homebrew kegs, macOS java_home, common Linux paths.
set -uo pipefail

major_version() {
  "$1/bin/java" -version 2>&1 | awk -F'"' '/version/ {print $2}' | cut -d. -f1
}

candidates=()
[[ -n "${JAVA_HOME:-}" ]] && candidates+=("$JAVA_HOME")
for v in 17 21; do
  for prefix in /opt/homebrew/opt /usr/local/opt; do
    candidates+=("$prefix/openjdk@$v/libexec/openjdk.jdk/Contents/Home")
  done
  # Note: java_home -v silently returns the newest JDK when no match exists,
  # which is why every candidate's real version is checked below.
  if [[ -x /usr/libexec/java_home ]]; then
    candidates+=("$(/usr/libexec/java_home -v "$v" 2>/dev/null || true)")
  fi
  candidates+=(/usr/lib/jvm/java-"$v"-openjdk* /usr/lib/jvm/temurin-"$v"*)
done

for c in "${candidates[@]}"; do
  if [[ -n "$c" && -x "$c/bin/java" ]]; then
    v="$(major_version "$c")"
    if [[ "$v" == "17" || "$v" == "21" ]]; then
      echo "$c"
      exit 0
    fi
  fi
done

echo "ERROR: Spark 4 needs Java 17 or 21 and none was found." >&2
echo "  macOS:  brew install openjdk@17" >&2
echo "  Ubuntu: sudo apt-get install -y openjdk-17-jdk-headless" >&2
echo "  Or skip local Java entirely with: docker compose up" >&2
exit 1
