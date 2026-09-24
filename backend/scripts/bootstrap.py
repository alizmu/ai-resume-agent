"""一键换成你自己的资料：源文件 / 网页文本 -> knowledge/raw -> persona.json -> 自动入库。

两种使用方式：

1) 命令行（本地换资料时用）
    python -m backend.scripts.bootstrap --name 张三 --profile resume.pdf \
        --projects p1.md p2.md --qa qa.md

    # 只生成文件与 persona.json、暂不下载模型入库（先看效果）：
    python -m backend.scripts.bootstrap --name 张三 --profile me.md --dry-run

2) 作为库函数（Web 端「更换资料」接口 /api/bootstrap 走的就是这条路径）
    from backend.scripts.bootstrap import bootstrap
    bootstrap(name="张三", profile_text="# 个人档案\\n…", project_texts=["# 项目A\\n…"])

设计要点：
- **知识库即数据库**：换一个人 = 换 knowledge/ 下的数据，框架代码零改动。
- **两种输入殊途同归**：上传的文件和网页里直接填的文本，最终都落盘成
  knowledge/raw 下的真实文档。这样重启服务或触发 reindex 都能从数据层完整重建，
  不会出现"网页能问、重启就丢"的假入库。
- **覆盖式换人**：只要本次有资料传入，就先清空三类目录，避免上一个人的碎片残留。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from loguru import logger

from backend.config.settings import settings

SUPPORTED = {".md", ".markdown", ".txt", ".docx", ".pdf"}

# 各资料类型对应的 raw 子目录
_TYPE_DIRS = {
    "profile": "01-profile",
    "projects": "02-projects",
    "qa": "03-qa",
}

# 文件名非法字符（含路径分隔符，防目录穿越）
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')


def _resolve_src(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"源文件不存在: {p}")
    if p.suffix.lower() not in SUPPORTED:
        raise ValueError(f"不支持的源文件格式: {p.suffix}（支持 {sorted(SUPPORTED)}）")
    return p


def safe_filename(name: str, default: str = "未命名") -> str:
    """把任意标题变成安全文件名：剔除非法字符、截断、空值兜底。

    文件名可能来自用户上传的 original filename 或网页填写的 H1 标题，
    直接拼路径会有目录穿越（../）与写入失败风险，必须净化。
    """
    s = _ILLEGAL_CHARS.sub("", (name or "").strip()).strip(" .")
    return (s[:60] or default)


def _clear_dir(dir_path: Path) -> None:
    """清空子目录下的所有文件（保留目录本身），实现「覆盖式换人」。"""
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return
    for f in dir_path.iterdir():
        if f.is_file():
            f.unlink()


def _copy_in(src: Path, raw_dir: Path, sub: str) -> str:
    """把已有文件拷进 raw 子目录，返回相对路径（便于回显给用户）。"""
    target_dir = raw_dir / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / safe_filename(src.name, default=src.stem or "资料")
    # 拷贝前补齐后缀：上传文件可能丢后缀（浏览器一般保留，但不能依赖）
    if dest.suffix.lower() not in SUPPORTED:
        dest = dest.with_suffix(".md")
    shutil.copy2(src, dest)
    logger.info(f"已写入 {sub}/{dest.name}")
    return f"{sub}/{dest.name}"


def _write_text(text: str, raw_dir: Path, sub: str, title: str) -> str:
    """把网页填写的文本落盘成 md 文档，返回相对路径。

    落盘而非只进向量库，是为了让 knowledge/raw 始终是唯一数据源：
    之后重启、reindex、换机器部署都能完整重建。
    """
    target_dir = raw_dir / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = safe_filename(title) + ".md"
    dest = target_dir / fname
    dest.write_text(text.strip() + "\n", encoding="utf-8")
    logger.info(f"已写入 {sub}/{fname}（{len(text)} 字）")
    return f"{sub}/{fname}"


def _title_from_text(text: str) -> str | None:
    """从文本里取首个 H1 作为标题（与 loader._infer_title 口径一致）。"""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return None


def _derive_title(path: Path) -> str:
    """从文件推断一个项目锚点名：md 取首个 H1，其它取文件名（去后缀）。"""
    try:
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = _title_from_text(text)
            if title:
                return title
    except Exception:
        pass
    return path.stem


def build_persona(name: str, anchors: list[str], keywords: list[str]) -> dict:
    """构造 persona 数据。anchors 是项目白名单，供指代消解与拒答判定使用。"""
    return {
        "name": name,
        "brand": "RAG 简历助手",
        "project_anchors": anchors,
        "project_keywords": keywords,
    }


def _backup_knowledge(raw_dir: Path, persona_path: Path, backup_dir: Path) -> None:
    """换人前备份 knowledge/raw 与 persona.json。

    必须备份的原因：bootstrap 是覆盖式的，会先清空 raw 三类目录再写入。
    一旦写入成功但后续入库失败（模型缺失、向量库异常等），
    用户的新资料进了 raw、旧资料已被删除，而向量库还是旧数据——资料实际丢失且不可回滚。
    """
    if raw_dir.exists():
        shutil.copytree(raw_dir, backup_dir / "raw", dirs_exist_ok=True)
    if persona_path.exists():
        shutil.copy2(persona_path, backup_dir / "persona.json")


def _restore_knowledge(raw_dir: Path, persona_path: Path, backup_dir: Path) -> None:
    """失败回滚：把换人前的 raw 与 persona.json 还原。"""
    try:
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
        src = backup_dir / "raw"
        if src.exists():
            shutil.copytree(src, raw_dir, dirs_exist_ok=True)
        if (backup_dir / "persona.json").exists():
            shutil.copy2(backup_dir / "persona.json", persona_path)
        logger.warning(f"已回滚知识库到换人前的状态: {raw_dir}")
    except Exception as e:  # pragma: no cover
        logger.error(f"回滚失败，请手动检查 {raw_dir}: {e}")


def bootstrap(
    name: str,
    profile_files: list[Path] | None = None,
    project_files: list[Path] | None = None,
    qa_files: list[Path] | None = None,
    profile_text: str | None = None,
    project_texts: list[str] | None = None,
    qa_text: str | None = None,
    anchors: list[str] | None = None,
    keywords: list[str] | None = None,
    do_ingest: bool = True,
) -> dict:
    """把「一个人的资料」写进知识库并（可选）自动切片入库。

    返回统计字典，便于 CLI 打印、也给 Web 接口回显：
        {name, written: [...], persona: {...}, ingested: bool,
         docs, chunks, stored, elapsed}
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("name 不能为空")

    profile_files = list(profile_files or [])
    project_files = list(project_files or [])
    qa_files = list(qa_files or [])
    project_texts = list(project_texts or [])

    raw_dir = settings.knowledge_raw_dir
    persona_path = settings.knowledge_raw_dir.parent / "persona.json"
    written: list[str] = []

    has_input = bool(
        profile_files
        or project_files
        or qa_files
        or (profile_text and profile_text.strip())
        or any(t and t.strip() for t in project_texts)
        or (qa_text and qa_text.strip())
    )

    # 覆盖式换人会先清空 raw 三类目录，一旦后续入库失败用户的资料就没了。
    # 因此先备份，异常时整体回滚（含用旧资料重建向量库）。
    backup_dir: Path | None = None
    if has_input:
        backup_dir = Path(tempfile.mkdtemp(prefix="bootstrap_backup_"))
        _backup_knowledge(raw_dir, persona_path, backup_dir)

    try:
        if has_input:
            # 覆盖式清空三类目录，避免上一个人的碎片残留
            for sub in _TYPE_DIRS.values():
                _clear_dir(raw_dir / sub)

            # 1) 上传/指定的文件
            for src in profile_files:
                written.append(_copy_in(src, raw_dir, _TYPE_DIRS["profile"]))
            for src in project_files:
                written.append(_copy_in(src, raw_dir, _TYPE_DIRS["projects"]))
            for src in qa_files:
                written.append(_copy_in(src, raw_dir, _TYPE_DIRS["qa"]))

            # 2) 网页直接填写的文本
            if profile_text and profile_text.strip():
                written.append(
                    _write_text(
                        profile_text,
                        raw_dir,
                        _TYPE_DIRS["profile"],
                        _title_from_text(profile_text) or "个人档案",
                    )
                )
            for i, text in enumerate(project_texts, start=1):
                if not (text and text.strip()):
                    continue
                written.append(
                    _write_text(
                        text,
                        raw_dir,
                        _TYPE_DIRS["projects"],
                        _title_from_text(text) or f"项目{i}",
                    )
                )
            if qa_text and qa_text.strip():
                written.append(
                    _write_text(
                        qa_text,
                        raw_dir,
                        _TYPE_DIRS["qa"],
                        _title_from_text(qa_text) or "面试问答对",
                    )
                )
        else:
            logger.warning("未提供任何资料，本次只更新 persona 的姓名，知识库保持原样。")

        # persona 锚点：显式指定优先，否则从项目文件 / 文本的标题自动推断
        if anchors is None:
            anchors = [_derive_title(p) for p in project_files]
            for text in project_texts:
                title = _title_from_text(text or "")
                if title:
                    anchors.append(title)
            # 去重并保持顺序
            anchors = list(dict.fromkeys(a for a in anchors if a))

        persona = build_persona(name, anchors, keywords or [])
        persona_path.parent.mkdir(parents=True, exist_ok=True)
        persona_path.write_text(
            json.dumps(persona, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            f"已生成 persona.json: {persona_path}（name={persona['name']}, "
            f"anchors={persona['project_anchors']}）"
        )

        result: dict = {
            "name": name,
            "written": written,
            "persona": persona,
            "ingested": False,
        }

        # persona 文件已变，缓存必须立即失效，否则提示词里还是旧名字
        from backend.core.persona import reload_persona

        reload_persona()

        if not do_ingest:
            return result

        # 延迟导入：只有真正要入库时才拉起向量库依赖（模型包较重）
        from backend.rag.retriever import reload_retriever
        from backend.scripts.ingest import build_index

        logger.info("开始自动入库（首次会下载 bge-small + bm25，约 100MB）…")
        # 新写入的资料必须同步进发布清单，否则会被 load_documents 的清单过滤挡掉，
        # 表现为「上传成功但检索不到」。
        from backend.rag.loader import write_manifest

        write_manifest()
        stats = build_index(recreate=True)
        # 集合被重建过，检索器若沿用旧实例可能仍指向已删除的集合句柄
        reload_retriever()

        result.update(stats)
        result["ingested"] = True
        return result
    except Exception:
        # 走到这里说明换人失败了，而 raw 可能已被覆盖 —— 必须回滚，否则资料丢失。
        if backup_dir is not None:
            _restore_knowledge(raw_dir, persona_path, backup_dir)
            try:
                from backend.core.persona import reload_persona

                reload_persona()
            except Exception:  # pragma: no cover
                pass
            # 尽量把向量库也恢复到旧资料，避免「raw 是旧的、库却是空的」
            try:
                from backend.rag.retriever import reload_retriever
                from backend.scripts.ingest import build_index

                build_index(recreate=True)
                reload_retriever()
                logger.warning("回滚完成，已用换人前的资料重建向量库")
            except Exception as e:  # pragma: no cover
                logger.error(f"回滚后重建向量库失败，请手动执行入库: {e}")
        raise
    finally:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键换成你自己的资料并自动切片入库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="候选人姓名（用于提示词与拒答判定）")
    parser.add_argument("--profile", help="个人档案/简历源文件（pdf/docx/md/txt）")
    parser.add_argument("--projects", nargs="+", default=[], help="项目文档（可多个）")
    parser.add_argument("--qa", nargs="+", default=[], help="面试问答对（可多个）")
    parser.add_argument("--anchors", nargs="+", help="项目白名单（默认取项目文件标题）")
    parser.add_argument("--keywords", nargs="+", help="Query 改写用的项目关键词")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成 knowledge/raw 与 persona.json，不下载模型入库",
    )
    args = parser.parse_args()

    profile_src = _resolve_src(args.profile) if args.profile else None
    project_srcs = [_resolve_src(p) for p in args.projects]
    qa_srcs = [_resolve_src(q) for q in args.qa]

    result = bootstrap(
        name=args.name,
        profile_files=[profile_src] if profile_src else None,
        project_files=project_srcs or None,
        qa_files=qa_srcs or None,
        anchors=args.anchors,
        keywords=args.keywords,
        do_ingest=not args.dry_run,
    )

    if args.dry_run:
        logger.success(
            f"dry-run 完成：已写入 {len(result['written'])} 个文件，persona.json 已生成。"
            "未执行入库（跳过模型下载）。去掉 --dry-run 重新运行即可入库。"
        )
        return

    logger.success(
        f"换人完成 ✅  {result.get('docs', 0)} 篇文档 / {result.get('chunks', 0)} 片段已入库，"
        f"耗时 {result.get('elapsed', 0)}s。启动服务即可对话："
        f"`uvicorn backend.api.main:app --port 8000`"
    )


if __name__ == "__main__":
    main()
