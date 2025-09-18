import os
import subprocess
import shutil
import time

defects4j_project_name_repository_map = {
    'Chart': 'jfreechart',
    'Cli': 'commons-cli',
    'Closure': 'closure-compiler',
    'Codec': 'commons-codec',
    'Collections': 'commons-collections',
    'Compress': 'commons-compress',
    'Csv': 'commons-csv',
    'Gson': 'gson',
    'JacksonCore': 'jackson-core',
    'JacksonDatabind': 'jackson-databind',
    'JacksonXml': 'jackson-dataformat-xml',
    'Jsoup': 'jsoup',
    'JxPath': 'commons-jxpath',
    'Lang': 'commons-lang',
    'Math': 'commons-math',
    'Mockito': 'mockito',
    'Time': 'joda-time'
}


def clean_tmp_folder(tmp_dir):
    if os.path.isdir(tmp_dir):
        for files in os.listdir(tmp_dir):
            file_p = os.path.join(tmp_dir, files)
            try:
                if os.path.isfile(file_p):
                    os.unlink(file_p)
                elif os.path.isdir(file_p):
                    shutil.rmtree(file_p)
            except Exception as e:
                print(e)
    else:
        os.makedirs(tmp_dir)


def defects4j_checkout(project_name, bug_id, checkout_path) -> bool:
    # delete the checkout path
    if os.path.isdir(checkout_path):
        print(f"checkout path {checkout_path} exists, delete it first!")
        shutil.rmtree(checkout_path)
    print("start checkout...")
    command = [
        "defects4j", "checkout",
        "-p", project_name,
        "-v", f"{bug_id}f",
        "-w", checkout_path
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    return True


def defects4j_compile(project_dir) -> bool:
    os.chdir(project_dir)
    print("start compile...")
    print(f"current work dir is: {os.getcwd()}")
    p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(f"{out.decode()}")
    print("finish compile...\n\n")
    if "FAIL" in str(err) or "FAIL" in str(out):
        return False
    return True


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


def defects4j_test(project_dir, timeout=300) -> str:
    print("\n------start test------")
    os.chdir(project_dir)
    print(f"current work dir is: {os.getcwd()}")
    out, err = command_with_timeout(["defects4j", "test", "-r"], timeout)
    print(f"{out.decode() or err.decode()}")
    print("------finish test------")

    if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
        correctness = 'Timeout'
    elif 'FAIL' in str(err) or 'FAIL' in str(out):
        correctness = 'Fail'
    elif "Failing tests: 0" in str(out):
        correctness = 'Plausible'
    else:
        correctness = 'Fail'
    return correctness


def defects4j_trigger(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout)
    return out, err


def defects4j_relevant(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.relevant"], timeout)
    return out, err


def defects4j_test_one(project_dir, test_case, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "test", "-t", test_case], timeout)
    return out, err
