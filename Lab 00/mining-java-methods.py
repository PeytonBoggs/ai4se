#!/usr/bin/env python3
"""
mine_java_methods_fixedrepos.py

Mine Java methods from a hard-coded set of popular GitHub repositories.

- Skips repo search API entirely (avoids rate limits).
- Hard-coded repo list known to contain many Java methods.
- Collects methods into one CSV file (default: 25k rows).
- First N (default 20k) go into dataset_split=train, rest eval.

Usage:
  export GITHUB_TOKEN="ghp_..."
  python mine_java_methods_fixedrepos.py --out-file ./java_methods.csv

Requirements:
  pip install requests tqdm javalang  (javalang optional; fallback parser included)
"""

import os, sys, time, base64, csv, re, argparse, hashlib
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

# Try javalang, fallback if unavailable
try:
    import javalang
    HAVE_JAVALANG = True
except Exception:
    HAVE_JAVALANG = False
    print("[warning] javalang not installed — using fallback parser.")

# ----------- Hard-coded repos -----------
POPULAR_JAVA_REPOS = [
    "spring-projects/spring-framework",
    "elastic/elasticsearch",
    "apache/hadoop",
    "apache/cassandra",
    "apache/kafka",
    "apache/lucene",
    "eclipse/eclipse.jdt.ls",
    "google/guava",
    "square/okhttp",
    "square/retrofit",
    "apache/tomcat",
    "apache/zookeeper",
    "apache/flink",
    "netty/netty",
    "dropwizard/dropwizard",
]

# ----------- Config -----------
MAX_WORKERS = 8
MIN_METHOD_LINES = 2
MAX_METHOD_LINES = 1000

CSV_FIELDS = [
    "dataset_split","example_id","repo_name","repo_url","repo_commit_sha","repo_license",
    "file_path","file_language","method_name","method_qualified_name",
    "method_start_line","method_end_line","method_signature",
    "method_original_code","method_doc_comment","code_tokens"
]

# ----------- GitHub API helpers -----------
def make_session(token):
    s = requests.Session()
    s.headers.update({
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "mine-java-fixedrepos/1.0"
    })
    return s

def retry_get(session, url, params=None, tries=6, backoff=2.0):
    wait = 1
    for i in range(tries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429) or 500 <= r.status_code < 600:
            time.sleep(wait)
            wait *= backoff
            continue
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text[:200]}")
    raise RuntimeError(f"GET {url} failed after {tries} tries")

def get_repo_commit_sha(session, owner, repo, branch):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{quote_plus(branch)}"
    return retry_get(session, url).json()["sha"]

def get_repo_tree(session, owner, repo, commit_sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit_sha}"
    return retry_get(session, url, params={"recursive":"1"}).json()

def get_file_contents(session, owner, repo, path, ref=None):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {} if not ref else {"ref": ref}
    j = retry_get(session, url, params=params).json()
    if isinstance(j, dict) and j.get("encoding") == "base64":
        return base64.b64decode(j["content"]).decode("utf-8", errors="replace")
    return None

# ----------- Java parsing helpers -----------
def find_method_end_line(lines, start_idx):
    depth = 0
    started = False
    for j in range(start_idx, len(lines)):
        for ch in lines[j]:
            if ch == "{": depth += 1; started = True
            elif ch == "}": depth -= 1
        if started and depth == 0:
            return j
    return len(lines)-1

def extract_javadoc_above(lines, idx):
    i = idx-1
    while i >= 0 and lines[i].strip() == "": i -= 1
    if i >= 0 and "*/" in lines[i]:
        out=[]
        while i >= 0:
            out.append(lines[i])
            if "/**" in lines[i]: return "\n".join(reversed(out))
            i -= 1
    return ""

def tokenize_code(code):
    if HAVE_JAVALANG:
        try: return [t.value for t in javalang.tokenizer.tokenize(code)]
        except: return re.findall(r"\w+|[^\s\w]", code)
    return re.findall(r"\w+|[^\s\w]", code)

def extract_methods(source, repo, file_path, commit_sha, license_str, train_cutoff, collected):
    rows=[]
    lines = source.splitlines()
    if HAVE_JAVALANG:
        try: tree = javalang.parse.parse(source)
        except: return []
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            if not node.position: continue
            start = node.position.line-1
            end = find_method_end_line(lines, start)
            if end-start+1 < MIN_METHOD_LINES or end-start+1 > MAX_METHOD_LINES: continue
            code="\n".join(lines[start:end+1])
            doc=extract_javadoc_above(lines,start)
            sig=node.name
            tokens=tokenize_code(code)
            exid=f"{repo}@{commit_sha}:{file_path}#{start+1}-{end+1}"
            split="train" if collected < train_cutoff else "eval"
            rows.append({
                "dataset_split":split,"example_id":exid,
                "repo_name":repo,"repo_url":f"https://github.com/{repo}",
                "repo_commit_sha":commit_sha,"repo_license":license_str,
                "file_path":file_path,"file_language":"Java",
                "method_name":node.name,
                "method_qualified_name":f"{repo}#{file_path}:{node.name}@{start+1}-{end+1}",
                "method_start_line":start+1,"method_end_line":end+1,
                "method_signature":sig,"method_original_code":code,
                "method_doc_comment":doc,"code_tokens":" ".join(tokens)
            })
    else:
        # fallback regex
        pat=re.compile(r'(public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{',re.M)
        for m in pat.finditer(source):
            start=source.count("\n",0,m.start())
            end=find_method_end_line(lines,start)
            if end-start+1<MIN_METHOD_LINES or end-start+1>MAX_METHOD_LINES: continue
            code="\n".join(lines[start:end+1])
            tokens=tokenize_code(code)
            name=m.group(2)
            exid=f"{repo}@{commit_sha}:{file_path}#{start+1}-{end+1}"
            split="train" if collected<train_cutoff else "eval"
            rows.append({
                "dataset_split":split,"example_id":exid,
                "repo_name":repo,"repo_url":f"https://github.com/{repo}",
                "repo_commit_sha":commit_sha,"repo_license":license_str,
                "file_path":file_path,"file_language":"Java",
                "method_name":name,
                "method_qualified_name":f"{repo}#{file_path}:{name}@{start+1}-{end+1}",
                "method_start_line":start+1,"method_end_line":end+1,
                "method_signature":m.group(0).strip(),"method_original_code":code,
                "method_doc_comment":"","code_tokens":" ".join(tokens)
            })
    return rows

# ----------- main loop -----------
def mine_fixed(token,out_file,target=25000,train=20000):
    s=make_session(token)
    collected=0; seen=set()
    os.makedirs(os.path.dirname(out_file) or ".",exist_ok=True)
    with open(out_file,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=CSV_FIELDS); w.writeheader()
        pbar=tqdm(total=target,desc="Collected methods")
        for repo in POPULAR_JAVA_REPOS:
            if collected>=target: break
            owner,name=repo.split("/")
            try: sha=get_repo_commit_sha(s,owner,name,"master")
            except: sha=get_repo_commit_sha(s,owner,name,"main")
            tree=get_repo_tree(s,owner,name,sha)
            java_files=[t["path"] for t in tree.get("tree",[]) if t["type"]=="blob" and t["path"].endswith(".java")]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs=[ex.submit(get_file_contents,s,owner,name,p,ref=sha) for p in java_files]
                for path,fut in zip(java_files,futs):
                    if collected>=target: break
                    code=fut.result()
                    if not code: continue
                    rows=extract_methods(code,repo,path,sha,"",train,collected)
                    for r in rows:
                        if collected>=target: break
                        h=hashlib.sha1(r["method_original_code"].encode()).hexdigest()
                        if h in seen: continue
                        seen.add(h)
                        if collected<train: r["dataset_split"]="train"
                        else: r["dataset_split"]="eval"
                        w.writerow(r); collected+=1; pbar.update(1)
        pbar.close()
    print(f"[+] Done. Collected {collected} methods -> {out_file}")

# ----------- CLI -----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--token",default=os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--out-file",default="./java_methods_fixed.csv")
    ap.add_argument("--target-methods",type=int,default=25000)
    ap.add_argument("--train-count",type=int,default=20000)
    args=ap.parse_args()
    if not args.token: sys.exit("Missing GitHub token.")
    mine_fixed(args.token,args.out_file,args.target_methods,args.train_count)