"""Parse Markdown files and batch-write them into Milvus Lite."""

from __future__ import annotations

import argparse
import multiprocessing
from multiprocessing import Queue
from pathlib import Path

from utils.env_utils import PROJECT_ROOT
from utils.log_utils import log


# 采用分布式，多进程的方式把数据写入到 Milvus 数据库
def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1: 解析目录下所有md文件并分批放入队列"""
    from documents.markdown_parser import MarkdownParser

    log.info(f"解析进程开始扫描目录:{dir_path}")

    # 获取目录下所有md文件
    md_files = [
        str(path)
        for path in sorted(Path(dir_path).iterdir(), key=lambda p: p.name)
        if path.is_file() and path.suffix.lower() == ".md"
    ]

    if not md_files:
        log.warning("未发现任何md文件")
        output_queue.put(None)  # 发送终止信号
        return

    parser = MarkdownParser()
    doc_batch = []
    for file_path in md_files:
        try:
            docs = parser.parse_markdown_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)

            # 达到批次大小时发送到队列中
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空当前缓冲区的所有批次数据
        except Exception as e:
            log.error(f"解析失败 {file_path}: {str(e)}")
            log.exception(e)
    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(md_files)}个文件")


def milvus_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入Milvus"""
    from documents.milvus_db import MilvusVectorSaver

    log.info("Milvus写入进程启动...")

    mv = MilvusVectorSaver()
    mv.create_collection()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            data = input_queue.get()  # 阻塞的函数
            if data is None:
                break
            if isinstance(data, list):
                mv.add_documents(data)
                total_count += len(data)
                log.info(f"累计已写入: {total_count} 个文档")

        except Exception as e:
            log.error(f"写入数据失败 ！")
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


def run_ingestion(
    data_dir: str | Path | None = None,
    batch_size: int = 20,
    queue_size: int = 20,
) -> None:
    """Run the parser and writer processes for one data directory."""
    configured_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "md"
    md_dir = configured_dir.expanduser()
    if not md_dir.is_absolute():
        md_dir = PROJECT_ROOT / md_dir
    md_dir = md_dir.resolve()

    if not md_dir.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_size)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(str(md_dir), docs_queue, batch_size),
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,),
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    writer_proc.join()
    if writer_proc.exitcode != 0:
        parser_proc.terminate()
    parser_proc.join()

    if writer_proc.exitcode != 0:
        raise RuntimeError(
            f"Milvus writer exited with code {writer_proc.exitcode}; "
            "check the writer logs above."
        )
    log.info("所有任务完成")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Markdown files (default: data/md).",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--queue-size", type=int, default=20)
    args = parser.parse_args()
    run_ingestion(args.data_dir, args.batch_size, args.queue_size)


if __name__ == "__main__":
    main()
