#!/usr/bin/python
import glob
import subprocess

with open("pyproject.toml") as f:
    found_start = 0
    cl = f.readlines()
    for l in cl:
        if l.startswith("version = "):
            spockversion = l[-7:-2]

print(spockversion)
# find changelog
with open("changelog.md") as f:
    found_start = 0
    changelog = ""
    cl = f.readlines()
    for l in cl:
        if found_start == 0 and l.startswith("### Version"):
            if spockversion in l:
                found_start = 1
                continue
        if found_start == 1 and l.startswith("### Version"):
            found_start = 2
        if found_start == 1:
            changelog += l

if found_start != 2 or len(changelog.strip())<5:
    raise RuntimeError("Changelog not found")

with open("_changelog.tmp", "w") as f:
    f.writelines(changelog.strip()+"\n")

print("----")
print("Changelog:\n")
print(changelog.strip())
print("----")
print("Next:")
print("\ngit commit -a -m \""+spockversion+"\"")
print("git tag "+spockversion+" && git push --tags")
print("gh release create "+spockversion+" --notes-file _changelog.tmp")
