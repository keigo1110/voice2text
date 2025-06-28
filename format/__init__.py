"""
format: 文字起こし結果の構造化変換フレームワーク

voice2structured.pyで生成された文字起こし結果を入力として、
様々な形式への構造化変換を提供します。

利用可能なフォーマット:
- summary: 要約生成
- minutes: 議事録形式
- action_items: アクションアイテム抽出
- topics: トピック分析
- speakers: 話者分析

使用例:
    # 要約生成
    python -m format.summary transcript.md

    # 議事録作成
    python -m format.minutes transcript.md

    # アクションアイテム抽出
    python -m format.action_items transcript.md
"""

from .base_formatter import BaseFormatter
from .summary import SummaryFormatter
from .minutes import MinutesFormatter
from .action_items import ActionItemsFormatter

__all__ = [
    "BaseFormatter",
    "SummaryFormatter",
    "MinutesFormatter",
    "ActionItemsFormatter",
]

__version__ = "1.0.0"
