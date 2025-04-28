import argparse
import concurrent.futures
import os
import subprocess
import time
from pathlib import Path
from typing import List, Set

# Add here integrations that cannot be tested in the CI
IGNORE_FOR_TESTING = [
    "llama-index-integrations/embeddings/llama-index-embeddings-ipex-llm",  # index returning 403
    "llama-index-integrations/llms/llama-index-llms-ipex-llm",  # index returning 403
    "llama-index-integrations/llms/llama-index-llms-mlx",  # OSX only
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find packages affected by changes since base branch"
    )
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Base branch to compare against (e.g., 'main')",
    )
    parser.add_argument(
        "--repo-root", default="./", help="Root directory of the repository"
    )
    parser.add_argument(
        "--workers", default=8, help="Number of concurrent processes running pytest"
    )
    return parser.parse_args()


def get_changed_files(base_ref: str, repo_root: str) -> List[str]:
    """Get list of files changed since the base branch."""
    try:
        cmd = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")

        return [f for f in result.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"Exception occurred: {e!s}")
        raise


def find_integrations(repo_root: Path) -> list[Path]:
    """
    Find all Python packages in the repo.
    Returns a dict mapping package names to their root directories.
    """
    package_roots: list[Path] = []

    for category_path in repo_root.iterdir():
        if not category_path.is_dir():
            continue

        if category_path.name == "storage":
            # The "storage" category has sub-folders
            package_roots += find_integrations(category_path)
            continue

        for package_name in category_path.iterdir():
            if not package_name.is_dir() or str(package_name) in IGNORE_FOR_TESTING:
                continue

            package_roots.append(package_name)

    return package_roots


def find_packs(repo_root: Path) -> list[Path]:
    """
    Find all Python packages in the repo.
    Returns a dict mapping package names to their root directories.
    """
    package_roots = []

    for package_name in repo_root.iterdir():
        if not package_name.is_dir() or str(package_name) in IGNORE_FOR_TESTING:
            continue

        package_roots.append(package_name)

    return package_roots


def find_utils(repo_root: Path) -> list[Path]:
    """
    Find all Python packages in the repo.
    Returns a dict mapping package names to their root directories.
    """
    package_roots = []

    for package_name in repo_root.iterdir():
        if not package_name.is_dir() or str(package_name) in IGNORE_FOR_TESTING:
            continue

        package_roots.append(package_name)

    return package_roots


def get_affected_packages(
    changed_files: list[str], all_packages: list[Path]
) -> Set[str]:
    """Get packages containing changed files."""
    affected_packages = set()

    for file_path in changed_files:
        # Find the package containing this file
        for pkg_dir in all_packages:
            if file_path.startswith(str(pkg_dir)):
                affected_packages.add(pkg_dir)
                break

    return affected_packages


def run_pytest(package_dir):
    env = os.environ.copy()
    if "VIRTUAL_ENV" in env:
        del env["VIRTUAL_ENV"]

    start = time.perf_counter()
    result = subprocess.run(
        [
            "uv",
            "run",
            "--",
            "pytest",
            "-q",
            "--disable-warnings",
            "--disable-pytest-warnings",
            "-W",
            "ignore::DeprecationWarning",
            "-W",
            "ignore::pytest.PytestDeprecationWarning",
        ],
        cwd=package_dir,
        text=True,
        capture_output=True,
        env=env,
    )
    success = result.returncode in (0, 5)
    no_tests = result.returncode == 5
    elapsed_time = time.perf_counter() - start
    return {
        "package": str(package_dir),
        "success": success,
        "no_tests": no_tests,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time": f"{elapsed_time:.2f}s",
    }


def main():
    args = parse_args()

    # Get changed files
    changed_files = get_changed_files(args.base_ref, args.repo_root)

    # Find all packages in the repo
    integrations_root = Path(args.repo_root) / "llama-index-integrations"
    all_integrations = find_integrations(integrations_root)
    packs_root = Path(args.repo_root) / "llama-index-packs"
    all_packs = find_packs(packs_root)
    utils_root = Path(args.repo_root) / "llama-index-utils"
    all_utils = find_utils(utils_root)

    all_packages = [
        Path("llama-index-core"),
        *all_integrations,
        Path("llama-index-networks"),
        *all_packs,
        *all_utils,
    ]

    # Map files to directly affected packages
    directly_affected = get_affected_packages(changed_files, all_packages)

    if "llama-index-core" in directly_affected:
        directly_affected = {str(p) for p in all_packages}

    # Run pytest for each affected package in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        future_to_package = {
            executor.submit(run_pytest, package): package
            for package in sorted(directly_affected)
        }

        for future in concurrent.futures.as_completed(future_to_package):
            result = future.result()
            results.append(result)

            # Print results as they complete
            package = result["package"]
            if result["success"]:
                no_tests_warning = "(no tests found)" if result["no_tests"] else ""
                print(f"✅ {package} succeeded in {result['time']} {no_tests_warning}")
            else:
                print(f"❌ {package} failed")
                print(f"Error:\n{result['stderr']}")
                print(f"Output:\n{result['stdout']}")

    # Print summary
    failed = [r["package"] for r in results if not r["success"]]
    if failed:
        print(f"\n{len(failed)} packages had test failures:")
        for p in failed:
            print(p)
        exit(1)
    else:
        print(f"\nAll tests passed for {len(results)} affected packages!")


if __name__ == "__main__":
    main()
