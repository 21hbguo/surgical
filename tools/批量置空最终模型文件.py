#!/usr/bin/env python3
import argparse
from pathlib import Path

def split_targets(root: Path):
    finals=sorted([p for p in root.rglob("model_final.pth") if p.is_file()])
    targets=[]
    skipped=[]
    for p in finals:
        if p.with_name("model_best.pth").is_file():
            targets.append(p)
        else:
            skipped.append(p)
    return targets,skipped

def get_size(path: Path):
    try:
        return path.stat().st_size
    except Exception:
        return 0

def format_mb(num_bytes: int):
    return f"{float(num_bytes)/(1024.0*1024.0):.2f}MB"

def main():
    parser=argparse.ArgumentParser(description="Recursively truncate all model_final.pth files to 0 bytes (files are kept, not deleted).")
    parser.add_argument("root",type=str,nargs="?",default=None,help="Target root directory")
    args=parser.parse_args()
    root_text=args.root if args.root is not None else input("Input target root directory: ").strip()
    root=Path(root_text).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Invalid directory: {root}")
        raise SystemExit(1)
    targets,skipped=split_targets(root)
    if not targets and not skipped:
        print(f"[INFO] No model_final.pth found under: {root}")
        return
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} model_final.pth without sibling model_best.pth:")
        for i,p in enumerate(skipped,1):
            print(f"{i}. {p}")
    if not targets:
        print("[INFO] No eligible model_final.pth to process.")
        return
    print(f"[INFO] Found {len(targets)} eligible files to process (requires sibling model_best.pth; will truncate to 0 bytes, not delete):")
    for i,p in enumerate(targets,1):
        print(f"{i}. {p}")
    before_total=sum(get_size(p) for p in targets)
    print(f"[INFO] Total size before: {before_total}B ({format_mb(before_total)})")
    ans=input("Proceed to truncate all listed files to 0 bytes (keep files)? [y/N]: ").strip().lower()
    if ans!="y":
        print("[INFO] Cancelled. No files were modified.")
        return
    ok=0
    fail=0
    for p in targets:
        try:
            p.write_bytes(b"")
            ok+=1
        except Exception as e:
            fail+=1
            print(f"[ERROR] Failed: {p} | {e}")
    after_total=sum(get_size(p) for p in targets)
    released=before_total-after_total
    print(f"[DONE] Truncated-to-0B: {ok}, Failed: {fail}")
    print(f"[DONE] Skipped(no model_best.pth): {len(skipped)}")
    print(f"[STATS] Total size after: {after_total}B ({format_mb(after_total)})")
    print(f"[STATS] Released: {released}B ({format_mb(released)})")

if __name__=="__main__":
    main()
