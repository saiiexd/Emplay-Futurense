import subprocess

old = "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"

commits = subprocess.check_output(
    ["git", "rev-list", "--reverse", "--all"],
    text=True
).splitlines()

for commit in commits:
    msg = subprocess.check_output(
        ["git", "show", "-s", "--format=%B", commit],
        text=True
    )

    new_msg = msg.replace(old, "").strip() + "\n"

    if new_msg != msg:
        print(f"Cleaning {commit[:12]}")
        print(new_msg)

        # This script only prepares the messages for inspection.
