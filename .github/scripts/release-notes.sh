#!/usr/bin/env bash
# Build GitHub Release notes from the conventional commits since the previous
# tag, writing them to the path given as $1.
#
# Shared by both release paths -- ci.yml's auto-publish job and release.yml's
# dispatch job -- so the two can never drift into describing the same commits
# differently.
#
# Neither path drafts, so this output ships to users unreviewed: the commit
# subjects it collects are the release notes, not a starting point for them.
#
# Runs on both ubuntu-latest and macos-latest, and macOS ships bash 3.2, so
# nothing here may use bash 4 syntax (no arrays, no `${x,,}`).
#
# Required env:
#   VERSION   the new version, without the `v` prefix (e.g. 0.9.1)
#   REPO_URL  https://github.com/<owner>/<repo>
set -euo pipefail

out="$1"
: >"$out"

# Must be read BEFORE the new tag is created, or the range collapses to nothing.
prev="$(git describe --tags --abbrev=0 2>/dev/null || true)"

commit_subjects() {
  if [ -n "$prev" ]; then
    git log --no-merges --format='%s' "$prev..HEAD"
  else
    # First release: no previous tag, so the range is the whole history. Spelled
    # as two calls rather than an interpolated range, because the empty-range
    # form needs an unquoted expansion and bash 3.2 has no safe empty array.
    git log --no-merges --format='%s'
  fi
}

# grep exits 1 when a type is absent from the range, which `set -o pipefail`
# would otherwise turn into a failed run -- hence the `|| true`.
section() {
  body="$(commit_subjects | grep -E "^$1(\(.+\))?: " | sed -E "s/^$1(\(.+\))?: /- /" || true)"
  if [ -n "$body" ]; then
    printf '### %s\n\n%s\n\n' "$2" "$body" >>"$out"
  fi
}

# chore/test/ci/build are deliberately omitted -- the compare link below covers
# them, and they are noise in user-facing notes.
section feat "Features"
section fix "Fixes"
section perf "Performance"
section refactor "Refactoring"
section docs "Documentation"

if [ -n "$prev" ]; then
  printf '**Full changelog:** %s/compare/%s...v%s\n' \
    "$REPO_URL" "$prev" "$VERSION" >>"$out"
fi
