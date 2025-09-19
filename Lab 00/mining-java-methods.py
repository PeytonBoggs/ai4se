import requests
import re
import os
import csv
from typing import List, Dict, Optional

GITHUB_API = "https://api.github.com"
PER_PAGE = 100

def get_java_files(repo, sha, token, verbose=False):
    # Recursively lists all Java files at a repo/commit
    url = f"{GITHUB_API}/repos/{repo}/git/trees/{sha}?recursive=1"
    headers = {'Authorization': f'token {token}'} if token else {}
    if verbose: print(f"Fetching tree: {url}")
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    tree = r.json()["tree"]
    return [item["path"] for item in tree if item["path"].endswith(".java") and item["type"] == "blob"]

def get_file_content(repo, path, ref, token, verbose=False):
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}?ref={ref}"
    headers = {'Authorization': f'token {token}'} if token else {}
    if verbose: print(f"Downloading file: {url}")
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["content"].encode("ascii") if r.json().get("encoding") == "base64" else r.json()["content"]

def extract_java_methods(java_code: str) -> List[Dict]:
    # Simple regex-based Java method extraction. Can be improved.
    method_regex = re.compile(
        r"(?P<doc_comment>/\*\*.*?\*/)?\s*(?P<signature>(public|protected|private|static|\s)+[\w<>\[\]]+\s+[\w<>]+\s*\([^\)]*\))\s*\{(?P<body>.*?)\n\}", 
        re.DOTALL,
    )
    methods = []
    for match in method_regex.finditer(java_code):
        doc_comment = match.group("doc_comment") or ""
        sig = match.group("signature").strip()
        # Simple name extraction
        name_match = re.search(r"([\w<>]+)\s*\(", sig)
        name = name_match.group(1) if name_match else "unknown"
        tokens = re.findall(r'\w+', sig)
        methods.append({
            "name": name,
            "signature": sig,
            "doc_comment": doc_comment.strip(),
            "original_code": match.group(0),
            "code_tokens": tokens,
        })
    return methods

def get_license(repo, token, verbose=False):
    url = f"{GITHUB_API}/repos/{repo}/license"
    headers = {'Authorization': f'token {token}'} if token else {}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("license", {}).get("spdx_id", "UNKNOWN")
    return "UNKNOWN"

def scrape_methods_to_csv(repo, commit_sha, dataset_split, out_csv, token=None, verbose=True):
    # License and metadata
    license = get_license(repo, token, verbose)
    java_files = get_java_files(repo, commit_sha, token, verbose)
    url = f"https://github.com/{repo}"
    results = []
    for path in java_files:
        content = get_file_content(repo, path, commit_sha, token, verbose)
        try:
            decoded = content.decode("utf-8")
        except AttributeError:
            decoded = content
        methods = extract_java_methods(decoded)
        for m in methods:
            # Find lines
            lines = decoded.splitlines()
            idx = decoded.find(m["original_code"])
            start_line = decoded[:idx].count("\n") + 1
            end_line = start_line + m["original_code"].count("\n")
            qualified_name = f"{path.replace('/', '.')}" + "#" + m["name"]
            example_id = f"{repo}@{commit_sha[:7]}:{path}#{start_line}-{end_line}"
            result = {
                "dataset_split": dataset_split,
                "example_id": example_id,
                "repo.name": repo.split("/")[-1],
                "repo.url": url,
                "repo.commit_sha": commit_sha,
                "repo.license": license,
                "file.path": path,
                "file.language": "Java",
                "method.name": m["name"],
                "method.qualified_name": qualified_name,
                "method.start_line": start_line,
                "method.end_line": end_line,
                "method.signature": m["signature"],
                "method.original_code": m["original_code"],
                "method.doc_comment": m["doc_comment"],
                "code_tokens": ' '.join(m["code_tokens"]),
            }
            results.append(result)
    fieldnames = [
        "dataset_split", "example_id", "repo.name", "repo.url", "repo.commit_sha", "repo.license", 
        "file.path", "file.language", "method.name", "method.qualified_name", "method.start_line", 
        "method.end_line", "method.signature", "method.original_code", "method.doc_comment", "code_tokens"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    if verbose:
        print(f"Wrote {len(results)} methods to {out_csv}")

# Usage: Set environment variable GITHUB_TOKEN or pass directly
if __name__ == "__main__":
    # Minimal example
    repo = "acme/awesome-lib"
    commit_sha = "3f9c2b1a7b0f41e3b7ad483f0b2f5e2f8f7e3a22"
    dataset_split = "train"
    token = os.environ.get("GITHUB_TOKEN")
    scrape_methods_to_csv(repo, commit_sha, dataset_split, "java_methods.csv", token)