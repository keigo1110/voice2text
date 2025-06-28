"""
action_items: 文字起こし結果からアクションアイテムを抽出

文字起こし結果を解析し、今後の行動項目を抽出・整理します。
"""

import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

from .base_formatter import BaseFormatter


class ActionItemsFormatter(BaseFormatter):
    """アクションアイテム抽出フォーマッター"""

    def get_format_name(self) -> str:
        return "action_items"

    def format(self, transcript_path: str, output_path: Optional[str] = None) -> str:
        """
        文字起こしからアクションアイテムを抽出

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
            output_path = input_path.parent / f"{input_path.stem}_action_items.md"

        # アクションアイテム抽出プロンプト
        prompt = self._create_action_items_prompt(data)

        # 文字起こし内容を結合
        transcript_content = self._extract_transcript_content(data)

        # Geminiでアクションアイテム抽出
        action_items_text = self.generate_with_gemini(prompt, transcript_content)

        # 出力データを作成
        output_data = {
            'content': self._format_action_items_output(data, action_items_text)
        }

        # 保存
        self.save_output(output_data, str(output_path))

        return str(output_path)

    def _create_action_items_prompt(self, data: dict) -> str:
        """アクションアイテム抽出プロンプトを作成"""

        # 話者情報の抽出
        speakers_info = ""
        if "話者一覧" in data.get('sections', {}):
            speakers_info = f"\n参加者情報:\n{data['sections']['話者一覧']}"

        prompt = f"""以下の文字起こしから、アクションアイテム（今後の行動項目）を抽出してください。

## アクションアイテム抽出の指示

以下の形式でアクションアイテムを整理してください：

### 1. 緊急度別分類
#### 🔴 高優先度（緊急）
- 担当者: 〇〇さん
- 期限: 〇〇まで
- 内容: 〇〇を実施
- 背景: なぜこの作業が必要か

#### 🟡 中優先度（重要）
- 担当者: 〇〇さん
- 期限: 〇〇まで
- 内容: 〇〇を実施
- 背景: なぜこの作業が必要か

#### 🟢 低優先度（通常）
- 担当者: 〇〇さん
- 期限: 〇〇まで
- 内容: 〇〇を実施
- 背景: なぜこの作業が必要か

### 2. 未割り当てタスク
明確な担当者が決まっていないが実施が必要な項目

### 3. 継続監視項目
定期的な確認や長期的な取り組みが必要な項目

## 抽出ルール
- 「〜する」「〜してください」「〜しましょう」等の行動を表す表現を重視
- 「宿題」「TODO」「やること」「検討する」「確認する」等のキーワードに注目
- 担当者が明示されている場合は正確に記録
- 期限が言及されている場合は正確に記録
- 曖昧な表現の場合は推定であることを明記

## 注意事項
- 単なる意見や提案ではなく、実際の行動項目のみを抽出
- 担当者や期限が不明確な場合は「要確認」と記載
- 優先度は文脈から判断して設定{speakers_info}"""

        return prompt

    def _extract_transcript_content(self, data: dict) -> str:
        """文字起こし内容を抽出"""
        if 'transcripts' in data and data['transcripts']:
            return '\n\n'.join(data['transcripts'])
        else:
            # フォールバック: raw_contentから抽出
            content = data.get('raw_content', '')
            # ## 文字起こし内容 以降を抽出
            if '## 文字起こし内容' in content:
                return content.split('## 文字起こし内容')[1].strip()
            return content

    def _format_action_items_output(self, data: dict, action_items_text: str) -> str:
        """アクションアイテム出力をフォーマット"""
        title = data.get('title', '音声文字起こし')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        output = f"""# {title} - アクションアイテム

## 抽出情報
- 抽出日時: {timestamp}
- フォーマット: アクションアイテム
- AI モデル: {self.model_name}

---

{action_items_text}

---

## 📝 次のステップ

1. **担当者の確認**: 未確定の担当者について関係者と調整
2. **期限の設定**: 期限が不明確な項目について具体的な日程を決定
3. **進捗管理**: 定期的な進捗確認の仕組みを設定

---

*このアクションアイテムリストは AI により自動抽出されました。内容の正確性や優先度については必要に応じて調整してください。*
"""

        return output


def main():
    """CLI エントリーポイント"""
    parser = argparse.ArgumentParser(
        description="文字起こし結果からアクションアイテムを抽出"
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

    args = parser.parse_args()

    try:
        formatter = ActionItemsFormatter(model_name=args.model)
        output_path = formatter.format(args.transcript_path, args.output)
        print(f"✅ アクションアイテムを抽出しました: {output_path}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        exit(1)


if __name__ == "__main__":
    main()