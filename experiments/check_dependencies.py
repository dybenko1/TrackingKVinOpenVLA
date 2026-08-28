from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion


# Change these if your paths are different
OPENVLA_DIR = Path("/opt/openvla")
LIBERO_DIR = Path("/opt/LIBERO")

OPENVLA_MIN_REQUIREMENTS = Path(
    "/opt/openvla/requirements-min.txt"
)

LIBERO_REQUIREMENTS = Path(
    "/opt/openvla/experiments/robot/libero/libero_requirements.txt"
)


# Packages we should NOT blindly reinstall on Jetson.
# These are commonly tied to the NVIDIA/CUDA stack.
PROTECTED_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "triton",
}


def normalize_name(name):
    return name.lower().replace("_", "-")


def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def find_requirement_files(repo_dir):
    """
    Look for common dependency files without installing anything.
    """
    candidates = list(repo_dir.glob("requirements*.txt"))

    return [path for path in candidates if path.exists()]


def parse_requirements_file(path):
    """
    Parse normal pip requirements.

    Skips:
      - comments
      - blank lines
      - editable installs
      - git URLs
      - -r includes
      - pip options
    """
    requirements = []

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            # Remove inline comments where possible
            if " #" in line:
                line = line.split(" #", 1)[0].strip()

            if line.startswith((
                "-e ",
                "git+",
                "http://",
                "https://",
                "-r ",
                "--",
            )):
                print(f"  [SKIP special requirement] {line}")
                continue

            try:
                requirements.append(Requirement(line))
            except Exception:
                print(f"  [COULD NOT PARSE] {line}")

    return requirements


def check_requirement(req):
    name = req.name
    normalized = normalize_name(name)

    installed = get_installed_version(name)

    if installed is None:
        return {
            "name": name,
            "required": str(req.specifier) or "any",
            "installed": None,
            "status": "MISSING",
            "protected": normalized in PROTECTED_PACKAGES,
        }

    # No version constraint
    if not req.specifier:
        status = "OK"
    else:
        try:
            status = (
                "OK"
                if Version(installed) in req.specifier
                else "VERSION_MISMATCH"
            )
        except InvalidVersion:
            status = "UNKNOWN_VERSION"

    return {
        "name": name,
        "required": str(req.specifier) or "any",
        "installed": installed,
        "status": status,
        "protected": normalized in PROTECTED_PACKAGES,
    }
def check_requirement_file(name, path):
    print("\n" + "=" * 80)
    print(name)
    print(f"Requirements: {path}")
    print("=" * 80)

    if not path.exists():
        print("Requirements file not found.")
        return []

    requirements = parse_requirements_file(path)

    results = []

    for req in requirements:
        results.append(check_requirement(req))

    return results

def check_repo(repo_name, repo_dir):
    print("\n" + "=" * 80)
    print(f"{repo_name}")
    print(f"Path: {repo_dir}")
    print("=" * 80)

    if not repo_dir.exists():
        print("Repository not found.")
        return []

    requirement_files = find_requirement_files(repo_dir)

    if not requirement_files:
        print("No requirements*.txt found.")
        print("Dependencies may instead be defined in pyproject.toml/setup.py.")
        return []

    all_results = []

    for req_file in requirement_files:
        print(f"\nReading: {req_file}")

        requirements = parse_requirements_file(req_file)

        for req in requirements:
            result = check_requirement(req)
            all_results.append(result)

    return all_results


def print_report(results):
    if not results:
        return

    # De-duplicate by package + constraint
    unique = {}
    for result in results:
        key = (normalize_name(result["name"]), result["required"])
        unique[key] = result

    results = list(unique.values())

    print("\n")
    print("=" * 80)
    print("DEPENDENCY REPORT")
    print("=" * 80)

    for result in sorted(results, key=lambda x: x["name"].lower()):
        marker = {
            "OK": "[ OK ]",
            "MISSING": "[MISS]",
            "VERSION_MISMATCH": "[VERS]",
            "UNKNOWN_VERSION": "[????]",
        }[result["status"]]

        protected = "  *** PROTECTED / JETSON ***" if result["protected"] else ""

        print(
            f"{marker} "
            f"{result['name']:<30} "
            f"required={result['required']:<18} "
            f"installed={str(result['installed']):<20}"
            f"{protected}"
        )

    print("\n")
    print("=" * 80)
    print("PACKAGES THAT MAY NEED ACTION")
    print("=" * 80)

    action_items = [
        result
        for result in results
        if result["status"] != "OK"
    ]

    if not action_items:
        print("Everything checked is satisfied.")
        return

    for result in action_items:
        print(
            f"{result['name']}: "
            f"{result['status']} "
            f"(required {result['required']}, "
            f"installed {result['installed']})"
        )

        if result["protected"]:
            print(
                "    WARNING: Jetson-sensitive package. "
                "Do NOT automatically pip install/upgrade it."
            )

        


def main():
    openvla_results = check_requirement_file(
        "OpenVLA minimal inference dependencies",
        OPENVLA_MIN_REQUIREMENTS,
    )

    libero_results = check_requirement_file(
        "OpenVLA LIBERO evaluation dependencies",
        LIBERO_REQUIREMENTS,
    )

    print_report(
        openvla_results + libero_results
    )


if __name__ == "__main__":
    main()