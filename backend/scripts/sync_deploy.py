"""把 backend / frontend / knowledge 同步进 deploy/online，消除双份后端副本（P0-9）。

单一代码源在仓库根目录，deploy/online 只是「发布制品」。每次发布前跑一次本脚本，
保证线上副本与源码一致；不删除 deploy/online 下的 .model_cache_v2（离线模型）
与 start.py / .env（部署入口与环境配置）。
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # backend/scripts -> repo root
ONLINE = ROOT / "deploy" / "online"

EXCLUDES = {
    "__pycache__",
    ".model_cache",
    ".model_cache_v2",
    ".qdrant_data",
    ".workbuddy",
    ".git",
    ".env",
    ".env.*",
    ".gitignore",
    ".pytest_cache",
    ".test",
    # 发布清单由本脚本生成，不属于源码同步范围，避免被镜像删除逻辑清掉
    "manifest.txt",
}


def _prune(src: Path, dst: Path, removed: list[Path]) -> None:
    """删除 dst 中「源码已不存在」的文件，保证 deploy/online 是真镜像。

    只增不删会让线上残留本地已删除的文档（已脱敏的旧版 / 测试文件），
    公开部署下可能泄露真实客户名，因此这里必须做镜像删除。
    """
    if not dst.exists():
        return
    for item in dst.iterdir():
        if item.name in EXCLUDES:
            continue
        src_item = src / item.name
        if item.is_dir():
            if not src_item.is_dir():
                shutil.rmtree(item)
                removed.append(item)
            else:
                _prune(src_item, item, removed)
        else:
            if not src_item.is_file():
                item.unlink()
                removed.append(item)


def _write_manifest(raw_dir: Path, out_file: Path) -> None:
    """生成发布清单（复用 backend.rag.loader.write_manifest，避免两份实现漂移）。

    清单用于线上过滤持久卷里的历史残留文件，必须与本次同步结果严格一致。
    """
    from backend.rag.loader import write_manifest

    if not raw_dir.exists():
        return
    write_manifest(raw_dir)
    n = len([p for p in raw_dir.rglob("*") if p.is_file()])
    print(f"生成发布清单 knowledge/manifest.txt（{n} 个文档）")


def _sync(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in EXCLUDES:
            continue
        target = dst / item.name
        if item.is_dir():
            _sync(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    assert ONLINE.exists(), f"未找到 {ONLINE}，请确认项目结构"
    removed: list[Path] = []
    for sub in ("backend", "frontend", "knowledge"):
        s = ROOT / sub
        if s.exists():
            print(f"同步 {sub} -> deploy/online/{sub}")
            _sync(s, ONLINE / sub)
            _prune(s, ONLINE / sub, removed)
    if removed:
        print(f"清理线上残留 {len(removed)} 项（源码中已删除）：")
        for p in removed:
            print(f"  - {p.relative_to(ONLINE)}")
    else:
        print("无残留文件")
    # 生成发布清单：列出本次发布实际包含的 raw 文档相对路径。
    # 部署平台复用持久卷时，线上会残留历史文件；loader 依据本清单只加载
    # 清单内的文档，保证「线上可检索内容 == 本次发布的源码内容」。
    _write_manifest(ONLINE / "knowledge" / "raw", ONLINE / "knowledge" / "manifest.txt")

    # 根目录的依赖/构建配置也同步一份到部署包，方便独立打包
    if (ROOT / "pyproject.toml").exists():
        shutil.copy2(ROOT / "pyproject.toml", ONLINE / "pyproject.toml")
        print("同步 pyproject.toml")
    print("完成：deploy/online 已与源码同步（保留 .model_cache / start.py / .env）")


if __name__ == "__main__":
    main()
