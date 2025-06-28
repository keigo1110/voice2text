"""
minutes: 文字起こし結果から議事録を生成

文字起こし結果を解析し、会議の議事録形式にフォーマットします。
"""

import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

from .base_formatter import BaseFormatter


class MinutesFormatter(BaseFormatter):
    """議事録生成フォーマッター"""

    def get_format_name(self) -> str:
        return "minutes"

    def format(self, transcript_path: str, output_path: Optional[str] = None) -> str:
        """
        文字起こしから議事録を生成

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
            output_path = input_path.parent / f"{input_path.stem}_minutes.md"

        # 議事録プロンプト
        prompt = self._create_minutes_prompt(data)

        # 文字起こし内容を結合
        transcript_content = self._extract_transcript_content(data)

        # Geminiで議事録生成
        minutes_text = self.generate_with_gemini(prompt, transcript_content)

        # 出力データを作成
        output_data = {"content": self._format_minutes_output(data, minutes_text)}

        # 保存
        self.save_output(output_data, str(output_path))

        return str(output_path)

    def _create_minutes_prompt(self, data: dict) -> str:
        """議事録生成プロンプトを作成"""

        # 話者情報の抽出
        speakers_info = ""
        if "話者一覧" in data.get("sections", {}):
            speakers_info = f"\n参加者情報:\n{data['sections']['話者一覧']}"

        prompt = f"""以下の文字起こしから、正式な議事録を作成してください。

## 議事録作成の指示

以下の構成で議事録を作成してください：

### 1. 会議概要
- 会議名: （内容から推定）
- 開催日時: （記録可能な場合）
- 参加者: （話者一覧から抽出）

### 2. 議事内容
各議題について以下の形式で記録：
- **議題**: 〇〇について
- **発言要旨**: 主要な発言・意見を整理
- **決定事項**: 決定された内容
- **保留事項**: 次回への持ち越し

### 3. アクションアイテム
- **担当者**: 〇〇さん
- **期限**: 〇〇まで
- **内容**: 〇〇を実施

### 4. 次回予定
- 次回開催予定や継続討議事項

## 注意事項
- 発言者の意図を正確に反映してください
- 決定事項と意見・提案を明確に区別してください
- 具体的な数字、日付、固有名詞は正確に記載してください
- 議事録として適切な敬語・表現を使用してください{speakers_info}"""

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

    def _format_minutes_output(self, data: dict, minutes_text: str) -> str:
        """議事録出力をフォーマット"""
        title = data.get("title", "会議")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        output = f"""# {title} 議事録

## 作成情報
- 作成日時: {timestamp}
- フォーマット: 議事録
- AI モデル: {self.model_name}

---

{minutes_text}

---

*この議事録は AI により自動生成されました。内容の正確性については必要に応じて確認してください。*
"""

        return output


def main():
    """CLI エントリーポイント"""
    parser = argparse.ArgumentParser(description="文字起こし結果から議事録を生成")
    parser.add_argument("transcript_path", help="文字起こしファイルのパス")
    parser.add_argument("-o", "--output", help="出力ファイルパス（省略時は自動生成）")
    parser.add_argument(
        "--model", default="gemini-2.5-flash", help="使用するGeminiモデル名"
    )

    args = parser.parse_args()

    try:
        formatter = MinutesFormatter(model_name=args.model)
        output_path = formatter.format(args.transcript_path, args.output)
        print(f"✅ 議事録を生成しました: {output_path}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        exit(1)


if __name__ == "__main__":
    main()
