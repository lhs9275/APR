import os
import shutil
import subprocess
import time


def map_git_to_bugsinpy_project_name(git_project_name: str):
    if git_project_name == "cli":
        bugsinpy_project_name = "httpie"
    elif git_project_name == "spaCy":
        bugsinpy_project_name = "spacy"
    else:
        bugsinpy_project_name = git_project_name
    return bugsinpy_project_name


def bugsinpy_checkout(project_name, bug_id, checkout_path) -> bool:
    # if ti success, it always has "Removing bugsinpy_run_test.sh"
    if os.path.isdir(checkout_path):
        print(f"checkout path {checkout_path} exists, delete it first!")
        shutil.rmtree(checkout_path)
    print("start checkout...")
    command = [
        "bugsinpy-checkout",
        "-p", project_name,
        "-i", bug_id,
        "-w", checkout_path
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # p = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(f"{out.decode()}")
    if p.returncode != 0:
        print(f"Checkout failed for {project_name}-{bug_id}\n")
        print(f"out: {out}\n")
        print(f"err: {err}\n")
        return False
    if not os.path.isdir(checkout_path):
        print(f"Checkout failed for {project_name}-{bug_id}: not os.path.isdir(checkout_path)")
        return False
    print("checkout succeeded")
    print("finish checkout...\n\n")
    if "Removing bugsinpy_run_test.sh" in str(out):
        return True
    return True


def bugsinpy_compile(project_dir) -> bool:
    os.chdir(project_dir)
    print("start compile...")
    print(f"current work dir is: {os.getcwd()}")
    # if wrong, it always has "This is not a checkout project folder"
    p = subprocess.Popen(["bugsinpy-compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(f"{out.decode() or err.decode()}")
    print("finish compile...\n\n")
    if "This is not a checkout project folder" in str(out):
        return False
    return True


def bugsinpy_test(project_dir) -> str:
    os.chdir(project_dir)
    print("\n------start test------")
    print(f"current work dir is: {os.getcwd()}")
    out, err = command_with_timeout(["bugsinpy-test"], timeout=120)
    print(f"{out.decode() or err.decode()}")
    print("------finish test------")
    # if there are 1 passed and 1 failed, it will return False
    # unittest return "FAILED" or "OK"
    # pytest return "failed" or "passed"
    if "FAILED" in str(out) or "failed" in str(out):
        return 'Fail'
    if "passed" in str(out) or "OK" in str(out):
        return 'Plausible'
    # It is possible return something like "module is not found"
    return 'Fail'


def command_with_timeout(cmd, timeout=300):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    out, err = p.communicate()
    return out, err
