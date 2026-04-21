# Intentionally buggy file used by the code-review-swarm example.
# DO NOT use as a reference — every reviewer should find at least one issue.

import os
import sqlite3

PASSWORD = "hunter2"  # security: hardcoded secret


def get_user(name):  # style: missing type hints
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # security: SQL injection via f-string
    cur.execute(f"SELECT * FROM users WHERE name = '{name}'")
    rows = cur.fetchall()
    return rows


def find_duplicates(items):
    # performance: O(n^2) duplicate detection where O(n) suffices
    dupes = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j] and items[i] not in dupes:
                dupes.append(items[i])
    return dupes


def write_log(message):
    # coverage: no tests for this function; also opens file every call
    with open("/tmp/app.log", "a") as f:
        f.write(message + "\n")


def main():
    # style: top-level side effect, no if-name-main guard around real work
    user = os.environ.get("USER", "")
    rows = get_user(user)
    print(f"found {len(rows)} rows")
    print("dupes:", find_duplicates([1, 2, 2, 3, 3, 3, 4]))


main()
