#!/bin/bash
# Progress report: everything added to the project, from the first commit to now.
# Usage: bash tools/progress_report.sh      -> writes docs/progress_report.md and prints it
cd "$(git rev-parse --show-toplevel)" || exit 1
mkdir -p docs
OUT=docs/progress_report.md
FIRST=$(git rev-list --max-parents=0 HEAD | tail -1)
{
echo "# A3 Humanoid Project - Progress Report"
echo
echo "Generated: $(date '+%Y-%m-%d %H:%M')  |  Author(s): $(git log --format='%an' | sort -u | paste -sd ', ')"
echo
echo "## Summary"
echo
echo "| Item | Value |"
echo "| --- | --- |"
echo "| First commit | $(git log --reverse --format='%ad' --date=format:'%Y-%m-%d %H:%M' | head -1) |"
echo "| Latest commit | $(git log -1 --format='%ad' --date=format:'%Y-%m-%d %H:%M') |"
echo "| Number of commits (saves) | $(git rev-list --count HEAD) |"
echo "| Files in the project | $(git ls-files | wc -l) |"
echo "| Python files | $(git ls-files '*.py' | wc -l) |"
echo "| Lines of Python code | $(git ls-files -z '*.py' | xargs -0 cat 2>/dev/null | wc -l) |"
echo "| Milestones (tags) | $(git tag | paste -sd ', ' | sed 's/^$/none yet/') |"
echo
echo "## Timeline: what was added in each step"
echo
git log --reverse --format='%h|%ad|%s' --date=format:'%Y-%m-%d %H:%M' | while IFS='|' read -r h d s; do
  echo "### $d - $s"
  echo
  echo "Commit \`$h\`"
  echo
  added=$(git show --diff-filter=A --name-only --format='' "$h" | grep -v '^$')
  changed=$(git show --diff-filter=M --name-only --format='' "$h" | grep -v '^$')
  [ -n "$added" ] && { echo "Added:"; echo "$added" | sed 's/^/- `/; s/$/`/'; echo; }
  [ -n "$changed" ] && { echo "Changed:"; echo "$changed" | sed 's/^/- `/; s/$/`/'; echo; }
done
echo "## All files in the project now"
echo
echo '```text'
git ls-files | awk -F/ '{print (NF > 1 ? $1 "/" : "top-level-files")}' | sort | uniq -c | awk '{printf "%-25s %s files\n", $2, $1}'
echo '```'
echo
echo "## Total change from the first commit to now"
echo
echo '```text'
git diff --stat "$FIRST" HEAD | tail -1
echo '```'
} > "$OUT"
cat "$OUT"
echo
echo ">>> Saved to $OUT"
