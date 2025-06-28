#!/usr/bin/env python3
"""
format CLI: 文字起こし結果の構造化変換統合ツール

文字起こし結果を様々な形式に変換するためのワンストップCLIツール
"""

import argparse
import sys
from pathlib import Path

from .summary import SummaryFormatter
from .minutes import MinutesFormatter
from .action_items import ActionItemsFormatter


def main():
    """メインCLI関数"""
    parser = argparse.ArgumentParser(
        description="文字起こし結果の構造化変換ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
利用可能なフォーマット:
  summary      要約を生成
  minutes      議事録を作成
  action_items アクションアイテムを抽出

使用例:
  # 要約生成
  python -m format.cli summary transcript.md

  # 議事録作成
  python -m format.cli minutes transcript.md -o meeting_minutes.md

  # アクションアイテム抽出
  python -m format.cli action_items transcript.md

  # 複数形式を一括処理
  python -m format.cli all transcript.md
        """
    )

    parser.add_argument(
        "format_type",
        choices=["summary", "minutes", "action_items", "all"],
        help="変換形式"
    )

    parser.add_argument(
        "transcript_path",
        help="文字起こしファイルのパス"
    )

    parser.add_argument(
        "-o", "--output",
        help="出力ファイルパス（省略時は自動生成）"
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="使用するGeminiモデル名"
    )

    parser.add_argument(
        "--output-dir",
        help="出力ディレクトリ（allモード時）"
    )

    args = parser.parse_args()

    # 入力ファイルの存在確認
    transcript_path = Path(args.transcript_path)
    if not transcript_path.exists():
        print(f"❌ 文字起こしファイルが見つかりません: {args.transcript_path}")
        sys.exit(1)

    try:
        if args.format_type == "all":
            # 全形式で変換
            generated_files = process_all_formats(args, transcript_path)
            print(f"✅ {len(generated_files)}個のファイルを生成しました:")
            for file_path in generated_files:
                print(f"   📄 {file_path}")
        else:
            # 単一形式で変換
            output_path = process_single_format(args, transcript_path)
            print(f"✅ 変換完了: {output_path}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)


def process_single_format(args, transcript_path: Path) -> str:
    """単一形式での変換処理"""
    formatters = {
        "summary": SummaryFormatter,
        "minutes": MinutesFormatter,
        "action_items": ActionItemsFormatter,
    }

    formatter_class = formatters[args.format_type]
    formatter = formatter_class(model_name=args.model)

    return formatter.format(str(transcript_path), args.output)


def process_all_formats(args, transcript_path: Path) -> list:
    """全形式での変換処理"""
    formatters = {
        "summary": SummaryFormatter,
        "minutes": MinutesFormatter,
        "action_items": ActionItemsFormatter,
    }

    generated_files = []

    # 出力ディレクトリの設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = transcript_path.parent

    for format_name, formatter_class in formatters.items():
        try:
            formatter = formatter_class(model_name=args.model)

            # 出力ファイル名を決定
            output_path = output_dir / f"{transcript_path.stem}_{format_name}.md"

            # 変換実行
            result_path = formatter.format(str(transcript_path), str(output_path))
            generated_files.append(result_path)

            print(f"  ✓ {format_name}: {result_path}")

        except Exception as e:
            print(f"  ❌ {format_name} 変換失敗: {e}")
            continue

    return generated_files


if __name__ == "__main__":
    main()