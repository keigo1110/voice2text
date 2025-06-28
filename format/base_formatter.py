"""
base_formatter: 文字起こし構造化変換の基底クラス

全ての変換フォーマッターが継承する共通機能を提供します。
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class BaseFormatter(ABC):
    """文字起こし構造化変換の基底クラス"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        初期化

        Args:
            model_name: 使用するGeminiモデル名
        """
        self.model_name = model_name
        self._setup_gemini()

    def _setup_gemini(self):
        """Gemini APIのセットアップ"""
        # API Keyの取得（環境変数を優先、次に設定ファイル）
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            # 設定ファイルからAPI Keyを読み取り
            config_path = Path("config.yaml")
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                    api_key = config.get("llm", {}).get("api", {}).get("api_key", "")
                except Exception as e:
                    logger.warning(f"Failed to load config file: {e}")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not found and no api_key in config.yaml"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def load_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        文字起こしファイルを読み込み

        Args:
            transcript_path: 文字起こしファイルのパス

        Returns:
            Dict: 解析されたデータ
        """
        path = Path(transcript_path)

        if path.suffix.lower() == ".md":
            return self._load_markdown(path)
        elif path.suffix.lower() == ".json":
            return self._load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_markdown(self, path: Path) -> Dict[str, Any]:
        """Markdownファイルの読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 基本的なパース（タイトル、セクション等）
        lines = content.split("\n")
        data = {"raw_content": content, "title": "", "sections": {}, "transcripts": []}

        current_section = None
        current_transcript = ""

        for line in lines:
            if line.startswith("# "):
                data["title"] = line[2:].strip()
            elif line.startswith("## "):
                current_section = line[3:].strip()
                data["sections"][current_section] = ""
            elif line.startswith("### [") and "]" in line:
                # タイムスタンプ付きセクション
                if current_transcript:
                    data["transcripts"].append(current_transcript.strip())
                current_transcript = line + "\n"
            elif current_transcript:
                current_transcript += line + "\n"
            elif current_section:
                data["sections"][current_section] += line + "\n"

        # 最後のトランスクリプトを追加
        if current_transcript:
            data["transcripts"].append(current_transcript.strip())

        return data

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """JSONファイルの読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_output(self, output_data: Dict[str, Any], output_path: str):
        """
        変換結果を保存

        Args:
            output_data: 変換結果データ
            output_path: 出力ファイルパス
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".md":
            self._save_markdown(output_data, path)
        elif path.suffix.lower() == ".json":
            self._save_json(output_data, path)
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")

    def _save_markdown(self, data: Dict[str, Any], path: Path):
        """Markdown形式で保存"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(data.get("content", ""))

    def _save_json(self, data: Dict[str, Any], path: Path):
        """JSON形式で保存"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def generate_with_gemini(self, prompt: str, transcript_content: str) -> str:
        """
        Geminiを使用してコンテンツを生成

        Args:
            prompt: 生成プロンプト
            transcript_content: 文字起こし内容

        Returns:
            str: 生成されたコンテンツ
        """
        try:
            full_prompt = f"{prompt}\n\n文字起こし内容:\n{transcript_content}"

            response = self.model.generate_content(
                contents=[full_prompt],
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                },
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    @abstractmethod
    def format(self, transcript_path: str, output_path: Optional[str] = None) -> str:
        """
        文字起こしを指定形式に変換

        Args:
            transcript_path: 文字起こしファイルのパス
            output_path: 出力ファイルパス（Noneの場合は自動生成）

        Returns:
            str: 出力ファイルパス
        """
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """フォーマット名を取得"""
        pass
