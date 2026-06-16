# Working style — read this before every task

## Assemble, don't generate

**Default: use what exists.** Before writing any new code, look for an existing tool, repo, library, or script that already does the job. Clone it, call it, import it.

**Only create something new when:**
- No existing tool does it, OR
- The existing pieces need a thin wire to connect them

**When you do create something new, explain it first:**
> "I'm using X (existing), Y (existing), and Z (existing). The only thing I need to write is a small file that wires them together because [specific reason]. Here's what it'll do and why I'm not using an off-the-shelf alternative."

Then wait for a nod before writing it.

## Talk before you build

For anything beyond a trivial one-liner:
1. Say what existing things you found
2. Say what (if anything) needs to be created and why
3. Let the user respond — they may know a better existing tool, or want to research/ask peers first

This keeps the user in the loop and learning, not just watching code appear.

## Plan before coding (existing rule)

Write a reviewable plan to `.claude/*_PLAN.md` and wait for approval before implementing or running anything non-trivial.
