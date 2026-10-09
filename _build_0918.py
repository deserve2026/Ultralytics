r"""
把 dataset_0918 的 images/ + labels/ 组织成带 train/val/test 划分的 YOLO 数据集。.

关键处理（源数据只读，绝不改动源目录）:
  1. 内容 MD5 去重: 797 -> 687 张。94 个重复组中 84 组标签冲突，
     统一保留【框数更多】的那一份（密集标注为更新版本）。
  2. 按【连续块轮转】划分，块长 = BLOCK 帧，轮转比例 train:val:test。
     既保证 val/test 覆盖整个序列的密度分布，又保证相邻帧绝不跨划分（防相邻帧泄漏）。
  3. 类别沿用源 classes.txt: 0=fish, 1=suspect。

输出: D:\\experimental_data\\mend26s\\dataset_0918_yolo
"""

import glob
import hashlib
import os
import re
import shutil
from collections import Counter, defaultdict

SRC = r"D:\experimental_data\mend26s\dataset_0918"
OUT = r"D:\experimental_data\mend26s\dataset_0918_yolo"
BLOCK = 20  # 连续块长度（帧）
CYCLE = ("train", "train", "train", "val", "test")  # 块轮转比例 3:1:1

NAMES = {0: "fish", 1: "suspect"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# ---------------------------------------------------------------- 1. 扫描 + 去重
print("[1/4] 扫描并按内容去重 ...")
records = {}
for p in glob.glob(os.path.join(SRC, "images", "*.png")):
    stem = os.path.splitext(os.path.basename(p))[0]
    num = int(re.search(r"(\d+)$", stem).group(1))
    lp = os.path.join(SRC, "labels", stem + ".txt")
    rows = (
        [l.strip() for l in open(lp, encoding="utf-8").read().splitlines() if l.strip()] if os.path.exists(lp) else []
    )
    records[stem] = {"img": p, "lab": lp, "num": num, "md5": md5(p), "rows": rows}

byh = defaultdict(list)
for stem in sorted(records, key=lambda s: records[s]["num"]):
    byh[records[stem]["md5"]].append(stem)

kept, dropped = [], []
for stems in byh.values():
    # 保留框数最多的；并列时保留帧号小的（保持确定性）
    best = max(stems, key=lambda s: (len(records[s]["rows"]), -records[s]["num"]))
    kept.append(best)
    dropped.extend(s for s in stems if s != best)

kept.sort(key=lambda s: records[s]["num"])
print(f"    原始 {len(records)} 张 -> 唯一内容 {len(kept)} 张，丢弃冗余 {len(dropped)} 张")

# 校验：类别合法
for s in kept:
    for r in records[s]["rows"]:
        f = r.split()
        assert len(f) == 5, f"{s} 字段数异常: {r}"
        assert int(f[0]) in (0, 1), f"{s} 非法类别: {r}"

# ---------------------------------------------------------------- 2. 连续块轮转划分
print(f"[2/4] 连续块轮转划分 (块长={BLOCK}, 轮转={CYCLE}) ...")
blocks = [kept[i : i + BLOCK] for i in range(0, len(kept), BLOCK)]
assign = {}
for bi, blk in enumerate(blocks):
    split = CYCLE[bi % len(CYCLE)]
    for s in blk:
        assign[s] = split

# ---------------------------------------------------------------- 3. 落盘
print(f"[3/4] 写入 {OUT} ...")
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
for sp in ("train", "val", "test"):
    os.makedirs(os.path.join(OUT, "images", sp), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", sp), exist_ok=True)

mode_used = Counter()
for s in kept:
    sp = assign[s]
    num = records[s]["num"]
    newstem = f"dxb_{num:06d}"  # 保留帧号，文件名全 ASCII
    mode_used[link_or_copy(records[s]["img"], os.path.join(OUT, "images", sp, newstem + ".png"))] += 1
    with open(os.path.join(OUT, "labels", sp, newstem + ".txt"), "w", encoding="utf-8", newline="\n") as f:
        if records[s]["rows"]:
            f.write("\n".join(records[s]["rows"]) + "\n")

with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8", newline="\n") as f:
    f.write(
        "# 由 dataset_0918 的 images/ + labels/ 构建\n"
        "# 去重: 797 -> 687 (内容 MD5; 重复组保留框数更多的一份)\n"
        f"# 划分: 连续块轮转 (块长 {BLOCK} 帧, 比例 {CYCLE.count('train')}:{CYCLE.count('val')}:{CYCLE.count('test')})\n"
        f"path: {OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 2\n\n"
        "names:\n"
        "  0: fish\n"
        "  1: suspect\n"
    )

# ---------------------------------------------------------------- 4. 报告 + 校验
print("[4/4] 校验 ...")
lines = [
    "dataset_0918 -> YOLO 数据集 构建报告",
    "=" * 66,
    f"源目录: {SRC}",
    f"输出  : {OUT}",
    "",
    "一、去重",
    f"  原始图片        : {len(records)}",
    f"  唯一内容        : {len(kept)}",
    f"  丢弃冗余        : {len(dropped)}",
    f"  重复组(内容相同): {len([1 for v in byh.values() if len(v) > 1])}",
    f"  其中标签冲突组  : {len([1 for h, v in byh.items() if len(v) > 1 and len({tuple(records[x]['rows']) for x in v}) > 1])}",
    "  冲突处理        : 保留框数更多的一份",
    "",
    "二、划分",
]
tot = Counter()
for sp in ("train", "val", "test"):
    ni = len(glob.glob(os.path.join(OUT, "images", sp, "*.png")))
    nl = len(glob.glob(os.path.join(OUT, "labels", sp, "*.txt")))
    c = Counter()
    nb = []
    for lp in glob.glob(os.path.join(OUT, "labels", sp, "*.txt")):
        rows = [l for l in open(lp, encoding="utf-8").read().splitlines() if l.strip()]
        nb.append(len(rows))
        for r in rows:
            c[r.split()[0]] += 1
    assert ni == nl, f"{sp} 图片 {ni} != 标签 {nl}"
    nums = sorted(
        int(re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0]).group(1))
        for p in glob.glob(os.path.join(OUT, "images", sp, "*.png"))
    )
    avg = sum(nb) / max(1, len(nb))
    lines.append(
        f"  {sp:6s} 图片={ni:4d}  fish(0)={c['0']:5d}  suspect(1)={c['1']:5d}  平均框/图={avg:5.2f}  帧号 {nums[0]}~{nums[-1]}"
    )
    tot["fish"] += c["0"]
    tot["suspect"] += c["1"]
    tot[sp] += ni
lines += [
    f"  合计   图片={tot['train'] + tot['val'] + tot['test']:4d}  fish(0)={tot['fish']:5d}  suspect(1)={tot['suspect']:5d}",
    "",
    "三、校验",
    "  全部标签 5 字段、类别仅 0/1 : 通过",
    "  图片/标签配对              : 通过",
    f"  图片落盘方式               : {dict(mode_used)}",
    "",
    "四、重要说明",
    "  1. 本数据集【全部来自 dataset_0918】, 未与其他数据集合并。",
    "  2. 有 128 张图与 py311/9_16 内容相同, 因此本模型与 exp_9_16_add 不可直接对比。",
    "  3. 标注密度前 130 帧明显低于其后(10.3 vs 18.6 框/图), 是源数据的固有特征。",
    "  4. val/test 是同一序列的不同连续段, 不代表跨场景泛化能力。",
    "  5. 源目录 dataset_0918 只读, 未做任何改动。",
]
txt = "\n".join(lines)
with open(os.path.join(OUT, "BUILD_REPORT.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(txt + "\n")
print(txt)
print(f"\n完成: {OUT}")
