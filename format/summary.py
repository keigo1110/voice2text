"""
summary: 文字起こし結果から要約を生成

文字起こし結果を解析し、重要なポイントを抽出して要約を作成します。
"""

import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

from .base_formatter import BaseFormatter


class SummaryFormatter(BaseFormatter):
    """要約生成フォーマッター"""

    def get_format_name(self) -> str:
        return "summary"

    def format(self, transcript_path: str, output_path: Optional[str] = None) -> str:
        """
        文字起こしから要約を生成

        Args:
            transcript_path: 文字起こしファイルのパス
            output_path: 出力ファイルパス

        Returns:
            str: 出力ファイルパス
        """
        # 文字起こしデータを読み込み
        data = self.load_transcript(transcript_path)

        # 出力パスの決定
        if output_path is None:
            input_path = Path(transcript_path)
            output_path = input_path.parent / f"{input_path.stem}_summary.md"

        # 要約プロンプト
        prompt = self._create_summary_prompt(data)

        # 文字起こし内容を結合
        transcript_content = self._extract_transcript_content(data)

        # Geminiで要約生成
        summary_text = self.generate_with_gemini(prompt, transcript_content)

        # 出力データを作成
        output_data = {"content": self._format_summary_output(data, summary_text)}

        # 保存
        self.save_output(output_data, str(output_path))

        return str(output_path)

    def _create_summary_prompt(self, data: dict) -> str:
        """要約生成プロンプトを作成"""

        # 話者情報の抽出
        speakers_info = ""
        if "話者一覧" in data.get("sections", {}):
            speakers_info = f"\n話者情報:\n{data['sections']['話者一覧']}"

        prompt = f"""以下の文字起こしから、重要なポイントを抽出して要約を作成してください。

## 要約作成の指示

1. **全体要約**: 全体の内容を200-300文字程度で要約
2. **主要な議題・トピック**: 話し合われた主要な内容をリストアップ
3. **重要な発言**: 特に重要な発言や意見があれば抽出
4. **決定事項**: 会議で決定された事項があれば明記
5. **アクションアイテム**: 今後の行動項目があれば抽出
6. **次回への持ち越し**: 未解決の課題や次回への持ち越し事項

## 注意事項
- 話者の発言を正確に反映してください
- 重要度に応じて優先順位をつけてください
- 具体的な数字や固有名詞は正確に記載してください{speakers_info}"""

        return prompt

    def _extract_transcript_content(self, data: dict) -> str:
        """文字起こし内容を抽出"""
        if "transcripts" in data and data["transcripts"]:
            return "\n\n".join(data["transcripts"])
        else:
            # フォールバック: raw_contentから抽出
            content = data.get("raw_content", "")
            # ## 文字起こし内容 以降を抽出
            if "## 文字起こし内容" in content:
                return content.split("## 文字起こし内容")[1].strip()
            return content

    def _format_summary_output(self, data: dict, summary_text: str) -> str:
        """要約出力をフォーマット"""
        title = data.get("title", "音声文字起こし")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        output = f"""# {title} - 要約

## 生成情報
- 生成日時: {timestamp}
- フォーマット: 要約
- AI モデル: {self.model_name}

---

{summary_text}

---

*この要約は AI により自動生成されました。*
"""

        return output


def main():
    """CLI エントリーポイント"""
    parser = argparse.ArgumentParser(description="文字起こし結果から要約を生成")
    parser.add_argument("transcript_path", help="文字起こしファイルのパス")
    parser.add_argument("-o", "--output", help="出力ファイルパス（省略時は自動生成）")
    parser.add_argument(
        "--model", default="gemini-2.5-flash", help="使用するGeminiモデル名"
    )

    args = parser.parse_args()

    try:
        formatter = SummaryFormatter(model_name=args.model)
        output_path = formatter.format(args.transcript_path, args.output)
        print(f"✅ 要約を生成しました: {output_path}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        exit(1)


if __name__ == "__main__":
    main()
