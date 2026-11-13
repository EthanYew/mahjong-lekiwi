"""Save a hardware session's logged samples to main, in one step (README, "Data files and merging").

    python scripts/save_samples.py                 # the samples changed here, named after the latest real session; asks first
    python scripts/save_samples.py --dry-run       # what it would commit, without committing
    python scripts/save_samples.py --session 20261113_100710_bench

The scripts append samples to files in data/ as they run (hover and return corrections, photo reviews). They go to main from
the main checkout, committed and pushed together so a commit never sits here without being on GitHub. This does the README's
three commands. The files are the ones .gitattributes merges by union, less the review log, which only a review's PR writes.
It never adds another file, and it stops if the pull or push fails, saying what to do.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import checkout, config
from mahjong.session import is_sim

def latest_session(sessions_dir=None):
    """The newest real (not --sim) session folder's name, or None. Newest first, so it reads one events file, not ~300."""
    return next((f.name for f in sorted(Path(sessions_dir or config.SESSIONS).glob("2*_*"), reverse=True)
                 if f.is_dir() and not is_sim(f)), None)


def git(*args, repo=None):
    print("$ git " + " ".join(args))
    return subprocess.run(["git", *args], cwd=repo or config.REPO, env=checkout.ENV).returncode


def save(files, session, repo=None):
    """Commit the files as "Samples from <session>", rebase on GitHub's main and push. Returns a problem, or None."""
    if git("add", "--", *files, repo=repo) or git("commit", "-m", f"Samples from {session}", "--", *files, repo=repo):
        return "the commit failed (above): nothing was pushed"
    if git("pull", "--rebase", "--autostash", "origin", "main", repo=repo):    # other data files here are usually dirty
        return ("the pull failed (above), and the commit is here but not on GitHub: resolve it (git status), then "
                "git rebase --continue && git push origin main")
    if git("push", "origin", "main", repo=repo):
        return "the push failed (above): the commit is here but not on GitHub; git push origin main once it can reach it"
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--session", help="the session named in the commit message (default: the latest real one)")
    p.add_argument("--dry-run", action="store_true", help="say what it would commit, and stop")
    return p.parse_args(argv)


def main(argv=None, ask=input, repo=None, sessions_dir=None):
    args = parse_args(argv)
    if checkout.is_worktree(repo):
        raise SystemExit("run this from the main checkout: samples reach main from where the hardware runs")
    if checkout.git("rev-parse", "--abbrev-ref", "HEAD", repo=repo) != "main":
        raise SystemExit("the main checkout isn't on main: switch back first")
    if (commits := checkout.unpushed(repo)):
        raise SystemExit(f"main here has {len(commits)} commit(s) GitHub doesn't ({'; '.join(commits[:3])}): the push would send them "
                         "with the samples, untested. Sort them out first (merge_pr.py's pull check says the same).")
    files = checkout.unsaved_samples(repo)
    if not files:
        print("No unsaved samples.")
        return
    session = args.session or latest_session(sessions_dir) or "a hardware session"
    print(f"Samples from {session}:")
    subprocess.run(["git", "diff", "--stat", "--", *files], cwd=repo or config.REPO, env=checkout.ENV)
    if args.dry_run:
        return
    if ask(f"Commit these {len(files)} file(s) to main and push? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing saved.")
        return
    if (problem := save(files, session, repo)):
        raise SystemExit(problem)
    print("Saved to main and pushed.")


if __name__ == "__main__":
    main()
