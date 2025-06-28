"""
voice2structured: 音声文字起こし専用パイプライン

音声ファイルを以下の2つのモードで文字起こし：
1. transcript: 講演や議論の正確な文字起こし（話者一貫性保持）
2. lifelog: 日常音声から行動推論を含む詳細記録

特徴：
- 話者の一貫性管理
- 詳細なコンテキスト情報の保持
- チャンク間の文脈保存強化
- 中断・再開機能
- 純粋な文字起こしに特化（構造化変換は/formatで実行）

使用例:
    # 全文文字起こしモード（デフォルト）
    python voice2structured.py audio.wav --mode transcript

    # ライフログモード
    python voice2structured.py audio.wav --mode lifelog

    # 中断した処理の再開
    python voice2structured.py audio.wav --resume
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
import yaml
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import argparse
import pickle

# プロンプトファイルのインポート
try:
    from prompts_transcript import (
        SYSTEM_PROMPT_TRANSCRIPT_CHUNK,
        USER_PROMPT_TRANSCRIPT_CHUNK,
    )
    from prompts_lifelog import (
        SYSTEM_PROMPT_LIFELOG_CHUNK,
        USER_PROMPT_LIFELOG_CHUNK,
    )
except ImportError as e:
    print(f"Error: プロンプトファイルが見つかりません: {e}")
    print("prompts_transcript.py と prompts_lifelog.py が必要です。")
    sys.exit(1)

# 必要なライブラリの遅延インポート
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print(
        "Error: google-generativeai not found. Install with: pip install google-generativeai"
    )
    sys.exit(1)

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
except ImportError:
    print("Error: pydub not found. Install with: pip install pydub")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("voice2structured.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class GeminiAPIConfig:
    """Gemini API設定"""

    api_key: str = ""
    base_url: str = ""
    timeout_sec: int = 120
    max_retries: int = 3


@dataclass
class GeminiModelConfig:
    """Geminiモデル設定"""

    name: str = "gemini-2.5-flash"
    fallback_models: List[str] = field(
        default_factory=lambda: ["gemini-1.5-flash", "gemini-1.5-pro"]
    )
    generation_config: Dict = field(
        default_factory=lambda: {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )
    safety_settings: Dict = field(
        default_factory=lambda: {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
    )


@dataclass
class ProcessingConfig:
    """処理設定"""

    chunk_duration_min: int = 8  # 分（より安全なサイズ）
    chunk_duration_max: int = 12  # 分（API制限とフィルター回避）
    target_tokens: int = 15000  # トークン数を削減
    hard_token_cap: int = 20000  # より厳しい制限
    max_concurrency: int = 3
    rolling_summary_tokens: int = 1000
    retry_max_attempts: int = 5
    retry_backoff_sec: int = 2
    retry_exponential_backoff: bool = True
    cost_budget_usd: float = 5.0
    cost_warn_threshold_usd: float = 4.0


@dataclass
class OutputConfig:
    """出力設定"""

    mode: str = "transcript"  # transcript or lifelog
    output_dir: str = "./outputs"


@dataclass
class ChunkMetadata:
    """チャンクメタデータ"""

    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    file_path: str
    tokens_estimated: int
    status: str = "pending"  # pending, processing, completed, failed


@dataclass
class EnhancedContext:
    """拡張コンテキスト情報"""

    rolling_summary: str = ""
    speaker_mapping: Dict[str, str] = field(default_factory=dict)
    speaker_descriptions: Dict[str, str] = field(default_factory=dict)
    key_topics: List[str] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    action_items: List[Dict] = field(default_factory=list)
    current_topic: str = ""
    last_speakers: List[str] = field(default_factory=list)  # 最近の話者履歴


@dataclass
class ProcessingState:
    """処理状態（中断・再開用）"""

    audio_path: str
    mode: str
    completed_chunks: List[int] = field(default_factory=list)
    context: EnhancedContext = field(default_factory=EnhancedContext)
    results: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=str)


class Voice2Structured:
    """音声文字起こし専用メインクラス"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化

        Args:
            config_path: YAML設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self._load_config()

        # 設定クラスの初期化
        self.api_config = GeminiAPIConfig()
        self.model_config = GeminiModelConfig()
        self.processing_config = ProcessingConfig()
        self.output_config = OutputConfig()

        # 設定を解析
        self._parse_config()

        # Gemini API設定
        self._configure_gemini_api()
        self.model = self._create_gemini_model()

        # ディレクトリ作成
        self.storage_dir = Path("./storage")
        self.temp_dir = self.storage_dir / "temp"
        self.chunks_dir = self.storage_dir / "chunks"
        self.transcripts_dir = self.storage_dir / "transcripts"
        self.checkpoints_dir = self.storage_dir / "checkpoints"

        for directory in [
            self.storage_dir,
            self.temp_dir,
            self.chunks_dir,
            self.transcripts_dir,
            self.checkpoints_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # コンテキスト管理
        self.context = EnhancedContext()

        # 現在処理中の音声ファイルパス（時刻計算用）
        self.current_audio_path = None

        logger.info(
            f"Voice2Structured initialized with model: {self.model_config.name}"
        )

    def _load_config(self) -> Dict:
        """YAML設定ファイルの読み込み"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(
                    f"Config file {self.config_path} not found. Using defaults."
                )
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _parse_config(self):
        """設定の解析"""
        # LLM設定
        llm_config = self.config.get("llm", {})

        # API設定
        api_config = llm_config.get("api", {})
        self.api_config.api_key = api_config.get("api_key", "")
        self.api_config.base_url = api_config.get("base_url", "")
        self.api_config.timeout_sec = api_config.get("timeout_sec", 120)
        self.api_config.max_retries = api_config.get("max_retries", 3)

        # モデル設定
        model_config = llm_config.get("model", {})
        self.model_config.name = model_config.get("name", "gemini-2.5-flash")
        self.model_config.fallback_models = model_config.get(
            "fallback_models", ["gemini-1.5-flash", "gemini-1.5-pro"]
        )

        # 生成設定
        generation_config = model_config.get("generation_config", {})
        self.model_config.generation_config.update(generation_config)

        # 安全設定
        safety_settings = model_config.get("safety_settings", {})
        self.model_config.safety_settings.update(safety_settings)

        # 処理設定
        processing_config = llm_config.get("processing", {})
        self.processing_config.max_concurrency = processing_config.get(
            "max_concurrency", 3
        )
        self.processing_config.rolling_summary_tokens = processing_config.get(
            "rolling_summary_tokens", 1000
        )

        # リトライ設定
        retry_config = processing_config.get("retry", {})
        self.processing_config.retry_max_attempts = retry_config.get("max_attempts", 5)
        self.processing_config.retry_backoff_sec = retry_config.get("backoff_sec", 2)
        self.processing_config.retry_exponential_backoff = retry_config.get(
            "exponential_backoff", True
        )

        # コスト設定
        cost_config = processing_config.get("cost_guard", {})
        self.processing_config.cost_budget_usd = cost_config.get("budget_usd", 5.0)
        self.processing_config.cost_warn_threshold_usd = cost_config.get(
            "warn_threshold_usd", 4.0
        )

        # チャンク設定
        chunk_config = self.config.get("chunk_policy", {})
        self.processing_config.chunk_duration_min = chunk_config.get("min_minutes", 30)
        self.processing_config.chunk_duration_max = chunk_config.get("max_minutes", 45)
        self.processing_config.target_tokens = chunk_config.get("target_tokens", 50000)
        self.processing_config.hard_token_cap = chunk_config.get(
            "hard_token_cap", 60000
        )

        # 出力設定
        io_config = self.config.get("io", {})
        input_config = io_config.get("input", {})
        self.output_config.mode = input_config.get("mode", "transcript")

        output_dir = io_config.get("output_dir", "./outputs")
        # ${job_id}を現在時刻で置換
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_config.output_dir = output_dir.replace("${job_id}", job_id)

    def _configure_gemini_api(self):
        """Gemini APIの設定"""
        # API Keyの取得（環境変数を優先）
        api_key = os.getenv("GOOGLE_API_KEY") or self.api_config.api_key

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not found and no api_key in config"
            )

        # Gemini APIの設定
        genai.configure(api_key=api_key)

        logger.info("Gemini API configured successfully")

    def validate_configuration(self):
        """設定の妥当性をチェック"""
        issues = []

        # API Key確認
        api_key = os.getenv("GOOGLE_API_KEY") or self.api_config.api_key
        if not api_key:
            issues.append(
                "❌ GOOGLE_API_KEY environment variable or config.api_key not set"
            )
        else:
            issues.append("✅ API Key found")

        # モデル設定確認
        if self.model_config.name:
            issues.append(f"✅ Model name: {self.model_config.name}")
        else:
            issues.append("❌ Model name not configured")

        # 生成設定確認
        gen_config = self.model_config.generation_config
        if gen_config.get("temperature", 0) < 0 or gen_config.get("temperature", 0) > 1:
            issues.append("⚠️  Temperature should be between 0.0 and 1.0")
        else:
            issues.append(f"✅ Temperature: {gen_config.get('temperature', 0.1)}")

        # 出力ディレクトリ確認
        try:
            Path(self.output_config.output_dir.replace("${job_id}", "test")).mkdir(
                parents=True, exist_ok=True
            )
            issues.append("✅ Output directory writable")
        except Exception as e:
            issues.append(f"❌ Output directory not writable: {e}")

        return issues

    def _create_gemini_model(self):
        """Geminiモデルの作成"""
        # モデル作成を試行（フォールバック付き）
        models_to_try = [self.model_config.name] + self.model_config.fallback_models

        for model_name in models_to_try:
            try:
                # 最新のAPIでは、モデル作成時には基本情報のみ渡す
                model = genai.GenerativeModel(model_name=model_name)

                # 軽量なテストでモデルが利用可能かチェック
                logger.info(f"Successfully created model: {model_name}")
                self.model_config.name = model_name  # 実際に使用するモデル名を更新
                return model

            except Exception as e:
                logger.warning(f"Failed to create model {model_name}: {e}")
                continue

        raise ValueError(f"All models failed: {models_to_try}")

    def _prepare_generation_config(self):
        """生成設定を準備"""
        return self.model_config.generation_config

    def _prepare_safety_settings(self):
        """安全設定を準備"""
        safety_settings = []
        for category, threshold in self.model_config.safety_settings.items():
            try:
                safety_settings.append(
                    {
                        "category": getattr(HarmCategory, category),
                        "threshold": getattr(HarmBlockThreshold, threshold),
                    }
                )
            except AttributeError as e:
                logger.warning(
                    f"Invalid safety setting: {category}={threshold}, skipping: {e}"
                )
                continue
        return safety_settings

    def detect_vad_segments(self, audio_path: str) -> List[Dict]:
        """
        VAD (Voice Activity Detection) でスピーチセグメントを検出

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            List[Dict]: 音声セグメント情報のリスト
        """
        logger.info(f"Detecting speech segments in: {audio_path}")

        try:
            # 音声ファイルを読み込み
            audio = AudioSegment.from_file(audio_path)

            # 16kHz, monoに変換（Gemini APIの推奨形式）
            audio = audio.set_frame_rate(16000).set_channels(1)

            # pydubの無音検知を使用
            # 無音でない部分を検出（最小長1秒、無音閾値-40dB）
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=1000,  # 1秒以上の無音
                silence_thresh=-40,  # -40dB以下を無音とする
                seek_step=100,  # 100ms単位で検索
            )

            segments = []
            for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
                segments.append(
                    {
                        "segment_id": i,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": end_ms - start_ms,
                        "start_sec": start_ms / 1000.0,
                        "end_sec": end_ms / 1000.0,
                        "duration_sec": (end_ms - start_ms) / 1000.0,
                    }
                )

            logger.info(f"Detected {len(segments)} speech segments")
            return segments

        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            # フォールバック: 全体を1セグメントとして扱う
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            return [
                {
                    "segment_id": 0,
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "duration_ms": duration_ms,
                    "start_sec": 0.0,
                    "end_sec": duration_ms / 1000.0,
                    "duration_sec": duration_ms / 1000.0,
                }
            ]

    def create_dynamic_chunks(self, audio_path: str) -> List[ChunkMetadata]:
        """
        動的チャンク分割（VAD + トークン制限考慮）

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            List[ChunkMetadata]: チャンクメタデータのリスト
        """
        logger.info("Creating dynamic chunks...")

        # VADでスピーチセグメントを検出
        segments = self.detect_vad_segments(audio_path)

        chunks = []
        current_chunk_start = 0
        current_chunk_duration = 0
        chunk_id = 0

        # 目標チャンク長（秒）
        target_duration = self.processing_config.chunk_duration_min * 60
        max_duration = self.processing_config.chunk_duration_max * 60

        i = 0
        while i < len(segments):
            segment = segments[i]
            segment_duration = segment["duration_sec"]

            # 現在のチャンクに追加した場合の長さ
            new_duration = current_chunk_duration + segment_duration

            # チャンク分割の判定
            should_split = False

            if new_duration > max_duration:
                # 最大長を超える場合は必ず分割
                should_split = True
            elif new_duration > target_duration:
                # 目標長を超えた場合、次のセグメントとの境界で分割
                should_split = True
            elif current_chunk_duration == 0:
                # 最初のセグメントは必ず含める
                should_split = False

            if should_split and current_chunk_duration > 0:
                # 現在のチャンクを確定
                chunk_end = current_chunk_start + current_chunk_duration
                chunk = self._create_chunk_file(
                    audio_path, chunk_id, current_chunk_start, chunk_end
                )
                chunks.append(chunk)

                # 次のチャンクの準備
                chunk_id += 1
                current_chunk_start = segment["start_sec"]
                current_chunk_duration = segment_duration
            else:
                # セグメントを現在のチャンクに追加
                if current_chunk_duration == 0:
                    current_chunk_start = segment["start_sec"]
                current_chunk_duration = segment["end_sec"] - current_chunk_start

            i += 1

        # 最後のチャンクを処理
        if current_chunk_duration > 0:
            chunk_end = current_chunk_start + current_chunk_duration
            chunk = self._create_chunk_file(
                audio_path, chunk_id, current_chunk_start, chunk_end
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} dynamic chunks")
        return chunks

    def _create_chunk_file(
        self, audio_path: str, chunk_id: int, start_sec: float, end_sec: float
    ) -> ChunkMetadata:
        """
        音声チャンクファイルを作成

        Args:
            audio_path: 元の音声ファイルパス
            chunk_id: チャンクID
            start_sec: 開始時刻（秒）
            end_sec: 終了時刻（秒）

        Returns:
            ChunkMetadata: チャンクメタデータ
        """
        try:
            audio = AudioSegment.from_file(audio_path)

            # 指定範囲を切り出し
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            chunk_audio = audio[start_ms:end_ms]

            # 16kHz, monoに変換
            chunk_audio = chunk_audio.set_frame_rate(16000).set_channels(1)

            # チャンクファイル保存
            chunk_filename = f"chunk_{chunk_id:03d}.wav"
            chunk_path = self.chunks_dir / chunk_filename
            chunk_audio.export(chunk_path, format="wav")

            # トークン数推定（1秒 = 32トークン）
            duration = end_sec - start_sec
            estimated_tokens = int(duration * 32)

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                start_time=start_sec,
                end_time=end_sec,
                duration=duration,
                file_path=str(chunk_path),
                tokens_estimated=estimated_tokens,
            )

            logger.info(
                f"Created chunk {chunk_id}: {duration:.1f}s, ~{estimated_tokens} tokens"
            )
            return metadata

        except Exception as e:
            logger.error(f"Failed to create chunk {chunk_id}: {e}")
            raise

    def extract_speakers_from_transcript(self, transcript: str) -> List[str]:
        """トランスクリプトから話者を抽出"""
        speakers = re.findall(r"^([^:：]+)[：:]", transcript, re.MULTILINE)
        # "自分"以外の話者をフィルタリング
        return [s.strip() for s in speakers if s.strip() and s.strip() != "自分"]

    def extract_key_information(self, transcript: str) -> Dict:
        """トランスクリプトから重要情報を抽出"""
        info = {"decisions": [], "action_items": [], "topics": [], "references": []}

        lines = transcript.split("\n")
        for line in lines:
            # 決定事項の検出
            if any(keyword in line for keyword in ["決定", "決まり", "確定", "合意"]):
                info["decisions"].append(line.strip())

            # アクションアイテムの検出
            if any(
                keyword in line
                for keyword in ["TODO", "やること", "宿題", "アクション", "〜する"]
            ):
                info["action_items"].append(line.strip())

            # 指示語の検出
            references = re.findall(r"(それ|これ|あれ|その件|あの件|先ほどの)", line)
            info["references"].extend(references)

        return info

    def update_context_from_result(self, result: Dict, context: EnhancedContext):
        """処理結果からコンテキストを更新"""
        if result["status"] != "completed":
            return

        # 話者の抽出と更新
        speakers = self.extract_speakers_from_transcript(result["transcript"])
        for speaker in speakers:
            if speaker not in context.speaker_mapping:
                context.speaker_mapping[speaker] = (
                    f"チャンク{result['chunk_id']}で初登場"
                )
                context.speaker_descriptions[speaker] = ""

            # 最近の話者履歴を更新
            if speaker not in context.last_speakers:
                context.last_speakers.append(speaker)
            if len(context.last_speakers) > 5:
                context.last_speakers.pop(0)

        # 重要情報の抽出
        key_info = self.extract_key_information(result["transcript"])

        # 決定事項の追加
        context.key_decisions.extend(key_info["decisions"][:3])

        # アクションアイテムの追加
        for item in key_info["action_items"]:
            context.action_items.append(
                {
                    "chunk_id": result["chunk_id"],
                    "content": item,
                    "timestamp": self._format_time(result["start_time"]),
                }
            )

        # 未解決の参照を更新
        context.unresolved_references = key_info["references"]

        # キートピックの更新（文字起こしから抽出）
        lines = result["transcript"].split("\n")
        topics = [line.strip() for line in lines if len(line.strip()) > 50]
        context.key_topics.extend(topics[:2])
        if len(context.key_topics) > 10:
            context.key_topics = context.key_topics[-10:]

    def format_context_for_prompt(self, context: EnhancedContext) -> str:
        """プロンプト用にコンテキストを整形"""
        context_info = f"""## コンテキスト情報

### これまでの要約
{context.rolling_summary}

### 登場人物
"""
        for speaker, first_appearance in context.speaker_mapping.items():
            desc = context.speaker_descriptions.get(speaker, "")
            context_info += f"- {speaker}: {first_appearance}"
            if desc:
                context_info += f" - {desc}"
            context_info += "\n"

        if context.key_topics:
            context_info += f"\n### 議論中のトピック\n"
            for topic in context.key_topics[-5:]:
                context_info += f"- {topic}\n"

        if context.key_decisions:
            context_info += f"\n### これまでの決定事項\n"
            for decision in context.key_decisions[-5:]:
                context_info += f"- {decision}\n"

        if context.unresolved_references:
            context_info += f"\n### 未解決の参照\n"
            context_info += f"前のチャンクで言及された内容: {', '.join(set(context.unresolved_references))}\n"

        if context.last_speakers:
            context_info += f"\n### 最近の話者\n"
            context_info += f"直前に話していた人: {', '.join(context.last_speakers)}\n"

        return context_info

    async def process_chunk(
        self, chunk: ChunkMetadata, context: EnhancedContext
    ) -> Dict:
        """
        単一チャンクの文字起こし処理（強化されたエラーハンドリング付き）

        Args:
            chunk: チャンクメタデータ
            context: 拡張コンテキスト

        Returns:
            Dict: 文字起こし結果のみ
        """
        logger.info(f"🎵 Processing chunk {chunk.chunk_id} ({chunk.duration:.1f}s)...")

        uploaded_file = None

        for attempt in range(self.processing_config.retry_max_attempts):
            try:
                # 音声ファイルをアップロード（リトライごとに再アップロード）
                if uploaded_file:
                    try:
                        genai.delete_file(uploaded_file.name)
                    except:
                        pass

                uploaded_file = genai.upload_file(chunk.file_path)

                # ファイルが処理されるまで待機
                while uploaded_file.state.name == "PROCESSING":
                    await asyncio.sleep(1)
                    uploaded_file = genai.get_file(uploaded_file.name)

                if uploaded_file.state.name == "FAILED":
                    raise Exception(f"File upload failed for chunk {chunk.chunk_id}")

                # コンテキスト情報を整形（リトライ時は簡素化）
                if attempt > 0:
                    # リトライ時はコンテキストを簡素化してフィルターを回避
                    context_info = "前のチャンクからの継続です。"
                else:
                    context_info = self.format_context_for_prompt(context)

                # プロンプト構築（モード別、リトライ時は短縮版）
                if self.output_config.mode == "transcript":
                    if attempt > 0:
                        # リトライ時は安全なプロンプト
                        system_prompt = "音声を正確に文字起こししてください。"
                        user_prompt = "この音声の内容を文字起こししてください。"
                    else:
                        system_prompt = SYSTEM_PROMPT_TRANSCRIPT_CHUNK.format(
                            context_info=context_info, chunk_id=chunk.chunk_id
                        )
                        user_prompt = USER_PROMPT_TRANSCRIPT_CHUNK
                else:  # lifelog
                    if attempt > 0:
                        # リトライ時は安全なプロンプト
                        system_prompt = (
                            "日常会話を自然な形でライフログとして記録してください。"
                        )
                        user_prompt = "この音声をライフログ形式で記録してください。"
                    else:
                        system_prompt = SYSTEM_PROMPT_LIFELOG_CHUNK.format(
                            context_info=context_info, chunk_id=chunk.chunk_id
                        )
                        user_prompt = USER_PROMPT_LIFELOG_CHUNK

                # 生成設定と安全設定を準備（リトライ時は温度を下げる）
                generation_config = self._prepare_generation_config()
                if attempt > 0:
                    generation_config["temperature"] = max(
                        0.0, generation_config["temperature"] - 0.05 * attempt
                    )

                safety_settings = self._prepare_safety_settings()

                # 文字起こし実行
                response = self.model.generate_content(
                    contents=[
                        f"System: {system_prompt}" if system_prompt else "",
                        user_prompt,
                        uploaded_file,
                    ],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                # レスポンスの詳細チェック
                if not response.candidates:
                    raise Exception("No candidates in response")

                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason

                if finish_reason == 2:  # SAFETY
                    logger.warning(
                        f"🛡️ Safety filter triggered for chunk {chunk.chunk_id}, attempt {attempt + 1}"
                    )
                    if attempt < self.processing_config.retry_max_attempts - 1:
                        await asyncio.sleep(
                            self.processing_config.retry_backoff_sec * (2**attempt)
                        )
                        continue
                    else:
                        # 最後の試行では部分的な結果を生成
                        transcript = f"[音声チャンク {chunk.chunk_id}: 安全性フィルターにより内容を表示できません]"
                elif finish_reason == 3:  # RECITATION
                    logger.warning(f"🔄 Recitation detected for chunk {chunk.chunk_id}")
                    transcript = f"[音声チャンク {chunk.chunk_id}: 既知のコンテンツが検出されました]"
                elif finish_reason == 1:  # STOP (正常終了)
                    transcript = response.text
                else:
                    transcript = (
                        response.text
                        if hasattr(response, "text")
                        else f"[音声チャンク {chunk.chunk_id}: 処理が不完全です]"
                    )

                # 成功時にはファイルを削除
                if uploaded_file:
                    try:
                        genai.delete_file(uploaded_file.name)
                    except:
                        pass

                result = {
                    "chunk_id": chunk.chunk_id,
                    "transcript": transcript,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration,
                    "status": "completed" if finish_reason == 1 else "partial",
                    "finish_reason": finish_reason,
                    "attempts": attempt + 1,
                }

                logger.info(
                    f"✅ Completed chunk {chunk.chunk_id} (attempt {attempt + 1}, reason: {finish_reason})"
                )
                return result

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"⚠️ Attempt {attempt + 1} failed for chunk {chunk.chunk_id}: {error_msg}"
                )

                if attempt < self.processing_config.retry_max_attempts - 1:
                    wait_time = self.processing_config.retry_backoff_sec * (2**attempt)
                    logger.info(f"🔄 Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # 最終試行失敗時のクリーンアップ
                    if uploaded_file:
                        try:
                            genai.delete_file(uploaded_file.name)
                        except:
                            pass

                    # 致命的でない場合は部分的な結果を返す
                    logger.error(
                        f"❌ Final attempt failed for chunk {chunk.chunk_id}: {error_msg}"
                    )
                    return {
                        "chunk_id": chunk.chunk_id,
                        "transcript": f"[音声チャンク {chunk.chunk_id}: 処理エラーが発生しました]",
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "duration": chunk.duration,
                        "status": "failed",
                        "error": error_msg,
                        "attempts": attempt + 1,
                    }

    def compress_rolling_summary(self, current_summary: str, new_content: str) -> str:
        """
        ローリング要約の圧縮

        Args:
            current_summary: 現在の要約
            new_content: 新しいコンテンツ

        Returns:
            str: 圧縮された要約
        """
        if not current_summary:
            return new_content[: self.processing_config.rolling_summary_tokens]

        # 既存要約と新コンテンツを結合
        combined = f"{current_summary}\n\n{new_content}"

        # トークン制限を超える場合は圧縮
        if len(combined) > self.processing_config.rolling_summary_tokens:
            try:
                compress_prompt = f"""
                以下のテキストを{self.processing_config.rolling_summary_tokens}文字以内で要約してください。
                重要な事実、決定事項、アクション、話者の情報を必ず保持してください。

                {combined}
                """

                response = self.model.generate_content(
                    contents=[compress_prompt],
                    generation_config=self._prepare_generation_config(),
                    safety_settings=self._prepare_safety_settings(),
                )
                compressed = response.text[
                    : self.processing_config.rolling_summary_tokens
                ]
                logger.info("Rolling summary compressed")
                return compressed

            except Exception as e:
                logger.warning(f"Summary compression failed: {e}")
                # フォールバック: 単純な切り詰め
                return combined[: self.processing_config.rolling_summary_tokens]

        return combined

    def save_checkpoint(self, state: ProcessingState):
        """処理状態のチェックポイント保存"""
        checkpoint_path = (
            self.checkpoints_dir / f"checkpoint_{state.mode}_{state.timestamp}.pkl"
        )
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_latest_checkpoint(self, mode: str) -> Optional[ProcessingState]:
        """最新のチェックポイントを読み込み"""
        checkpoints = list(self.checkpoints_dir.glob(f"checkpoint_{mode}_*.pkl"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest, "rb") as f:
                state = pickle.load(f)
            logger.info(f"Checkpoint loaded: {latest}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def process_audio_pipeline(
        self, audio_path: str, resume: bool = False
    ) -> Dict:
        """
        音声処理パイプラインの実行

        Args:
            audio_path: 音声ファイルのパス
            resume: 中断したところから再開するか

        Returns:
            Dict: 処理結果
        """
        logger.info(f"Starting audio processing pipeline for: {audio_path}")

        # 現在の音声ファイルパスを保存（時刻計算用）
        self.current_audio_path = audio_path

        try:
            # 処理状態の初期化または復元
            if resume:
                state = self.load_latest_checkpoint(self.output_config.mode)
                if state and state.audio_path == audio_path:
                    self.context = state.context
                    all_results = state.results
                    logger.info(f"Resuming from chunk {len(state.completed_chunks)}")
                else:
                    logger.warning("No valid checkpoint found. Starting fresh.")
                    resume = False

            if not resume:
                state = ProcessingState(
                    audio_path=audio_path,
                    mode=self.output_config.mode,
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                all_results = []

            # ステップ1: 動的チャンク分割
            chunks = self.create_dynamic_chunks(audio_path)

            # ステップ2: チャンクを順次文字起こし（改善されたプログレス表示付き）
            total_chunks = len(chunks)
            completed_count = 0
            success_count = 0
            partial_count = 0
            failed_count = 0

            logger.info(f"📊 Processing {total_chunks} chunks...")

            for chunk in chunks:
                # 既に処理済みのチャンクはスキップ
                if resume and chunk.chunk_id in state.completed_chunks:
                    logger.info(f"⏭️ Skipping already processed chunk {chunk.chunk_id}")
                    completed_count += 1
                    continue

                # プログレス表示
                progress = (completed_count / total_chunks) * 100
                logger.info(
                    f"📈 Progress: {progress:.1f}% ({completed_count}/{total_chunks})"
                )

                result = await self.process_chunk(chunk, self.context)
                all_results.append(result)

                # 結果の統計を更新
                completed_count += 1
                if result["status"] == "completed":
                    success_count += 1
                    # コンテキストを更新（要約なしで基本情報のみ）
                    self.update_context_from_result(result, self.context)
                elif result["status"] == "partial":
                    partial_count += 1
                else:
                    failed_count += 1

                # 処理状態を更新
                state.completed_chunks.append(chunk.chunk_id)
                state.context = self.context
                state.results = all_results

                # 定期的にチェックポイント保存（5チャンクごと）
                if len(state.completed_chunks) % 5 == 0:
                    self.save_checkpoint(state)
                    logger.info(
                        f"💾 Checkpoint saved (Success: {success_count}, Partial: {partial_count}, Failed: {failed_count})"
                    )

                # 次のチャンク処理前の短い待機
                await asyncio.sleep(0.5)

            # 最終統計の報告
            success_rate = (
                (success_count / total_chunks) * 100 if total_chunks > 0 else 0
            )
            logger.info(f"🎯 Processing completed!")
            logger.info(
                f"   ✅ Success: {success_count}/{total_chunks} ({success_rate:.1f}%)"
            )
            logger.info(f"   ⚠️ Partial: {partial_count}/{total_chunks}")
            logger.info(f"   ❌ Failed: {failed_count}/{total_chunks}")

            if failed_count > total_chunks * 0.5:
                logger.warning(
                    "⚠️ High failure rate detected. Consider reviewing chunk size or content."
                )
            elif success_rate >= 90:
                logger.info("🏆 Excellent processing quality!")

            # ステップ3: 簡易的なローリング要約を文字起こしから作成
            for result in all_results:
                if result["status"] == "completed":
                    # 文字起こしの最初の200文字を簡易要約として使用
                    simple_summary = result["transcript"][:200].replace("\n", " ")
                    self.context.rolling_summary = self.compress_rolling_summary(
                        self.context.rolling_summary, simple_summary
                    )

            # ステップ4: 結果の集約と出力生成
            output_results = await self.generate_outputs(all_results, audio_path)

            logger.info("Audio processing pipeline completed successfully")
            return output_results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            # エラー時もチェックポイント保存
            if "state" in locals():
                self.save_checkpoint(state)
            raise

    async def generate_outputs(self, results: List[Dict], audio_path: str) -> Dict:
        """
        最終出力の生成

        Args:
            results: チャンク処理結果のリスト
            audio_path: 元の音声ファイルパス

        Returns:
            Dict: 生成された出力ファイル情報
        """
        logger.info(f"Generating output for mode: {self.output_config.mode}")

        # 出力ディレクトリ作成
        output_dir = Path(self.output_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        # モード別出力生成
        if self.output_config.mode == "transcript":
            content = self._generate_transcript_output(results)
            output_file = output_dir / "transcript.md"

            # 話者一覧ファイルも生成
            speaker_info = self._generate_speaker_info()
            speaker_file = output_dir / "speakers.json"
            with open(speaker_file, "w", encoding="utf-8") as f:
                json.dump(speaker_info, f, ensure_ascii=False, indent=2)
            output_files["speakers"] = str(speaker_file)

        else:  # lifelog
            content = self._generate_lifelog_output(results)
            output_file = output_dir / "lifelog.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        output_files[self.output_config.mode] = str(output_file)

        # コンテキスト情報も保存
        context_file = output_dir / "context.json"
        with open(context_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.context), f, ensure_ascii=False, indent=2)
        output_files["context"] = str(context_file)

        logger.info(f"Generated {self.output_config.mode}: {output_file}")

        return output_files

    def _generate_speaker_info(self) -> Dict:
        """話者情報を生成"""
        speaker_info = {
            "total_speakers": len(self.context.speaker_mapping),
            "speakers": {},
        }

        for speaker, info in self.context.speaker_mapping.items():
            speaker_info["speakers"][speaker] = {
                "first_appearance": info,
                "description": self.context.speaker_descriptions.get(speaker, ""),
                "is_recent": speaker in self.context.last_speakers,
            }

        return speaker_info

    def _generate_transcript_output(self, results: List[Dict]) -> str:
        """全文文字起こしモードの出力生成（品質統計付き）"""
        # 処理統計の計算
        total_chunks = len(results)
        success_chunks = len([r for r in results if r["status"] == "completed"])
        partial_chunks = len([r for r in results if r["status"] == "partial"])
        failed_chunks = len([r for r in results if r["status"] == "failed"])
        success_rate = (success_chunks / total_chunks * 100) if total_chunks > 0 else 0

        content = f"""# 音声文字起こし

## 処理情報
- 処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- モデル: {self.model_config.name}
- チャンク数: {total_chunks}
- 処理モード: 全文文字起こし
- 識別された話者数: {len(self.context.speaker_mapping)}
- 処理品質: ✅成功 {success_chunks}, ⚠️部分 {partial_chunks}, ❌失敗 {failed_chunks} (成功率: {success_rate:.1f}%)

## 話者一覧
"""
        if self.context.speaker_mapping:
            for speaker, info in self.context.speaker_mapping.items():
                content += f"- **{speaker}**: {info}\n"
        else:
            content += "話者が識別されませんでした。\n"

        if self.context.key_decisions:
            content += "\n## 主要な決定事項\n"
            for decision in self.context.key_decisions[-5:]:
                content += f"- {decision}\n"

        if self.context.action_items:
            content += "\n## アクションアイテム\n"
            for item in self.context.action_items:
                content += f"- [{item['timestamp']}] {item['content']}\n"

        content += "\n## 文字起こし内容\n\n"

        for result in results:
            start_time = self._format_time(
                result["start_time"], self.current_audio_path
            )
            end_time = self._format_time(result["end_time"], self.current_audio_path)

            # チャンクのステータスを表示
            status_icon = (
                "✅"
                if result["status"] == "completed"
                else "⚠️" if result["status"] == "partial" else "❌"
            )
            content += f"### [{start_time} - {end_time}] {status_icon}\n\n"

            if result["status"] == "completed" and result["transcript"]:
                content += f"{result['transcript']}\n\n"
            elif result["status"] == "partial" and result["transcript"]:
                content += f"{result['transcript']}\n\n"
                content += (
                    "*（注：この部分は安全性フィルターまたは部分的な処理結果です）*\n\n"
                )
            elif result["status"] == "failed":
                error_info = result.get("error", "不明なエラー")
                content += f"❌ **処理エラー**: {error_info}\n\n"
                content += f"*リトライ回数: {result.get('attempts', 'N/A')}回*\n\n"
            else:
                content += "この音声チャンクには発話内容が含まれていません。\n\n"

        # 処理完了の注記
        if failed_chunks > 0:
            content += "## 注記\n\n"
            content += f"⚠️ {failed_chunks}個のチャンクで処理エラーが発生しました。\n"
            content += "チャンクサイズを小さくするか、音声品質を確認してください。\n\n"

        return content

    def _generate_lifelog_output(self, results: List[Dict]) -> str:
        """ライフログモードの出力生成（品質統計付き）"""
        # 処理統計の計算
        total_chunks = len(results)
        success_chunks = len([r for r in results if r["status"] == "completed"])
        partial_chunks = len([r for r in results if r["status"] == "partial"])
        failed_chunks = len([r for r in results if r["status"] == "failed"])
        success_rate = (success_chunks / total_chunks * 100) if total_chunks > 0 else 0

        content = f"""# ライフログ記録

## 処理情報
- 処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- モデル: {self.model_config.name}
- チャンク数: {total_chunks}
- 処理モード: ライフログ
- 処理品質: ✅成功 {success_chunks}, ⚠️部分 {partial_chunks}, ❌失敗 {failed_chunks} (成功率: {success_rate:.1f}%)

## 時系列ライフログ

"""

        for result in results:
            start_time = self._format_time(
                result["start_time"], self.current_audio_path
            )
            end_time = self._format_time(result["end_time"], self.current_audio_path)

            # チャンクのステータスを表示
            status_icon = (
                "✅"
                if result["status"] == "completed"
                else "⚠️" if result["status"] == "partial" else "❌"
            )
            content += f"### [{start_time} - {end_time}] {status_icon}\n\n"

            if result["status"] == "completed" and result["transcript"]:
                content += f"{result['transcript']}\n\n"
            elif result["status"] == "partial" and result["transcript"]:
                content += f"{result['transcript']}\n\n"
                content += (
                    "*（注：この部分は安全性フィルターまたは部分的な処理結果です）*\n\n"
                )
            elif result["status"] == "failed":
                error_info = result.get("error", "不明なエラー")
                content += f"❌ **処理エラー**: {error_info}\n\n"
                content += f"*リトライ回数: {result.get('attempts', 'N/A')}回*\n\n"
            else:
                content += "（無音区間）\n\n"

            content += "---\n\n"

        # 処理完了の注記
        if failed_chunks > 0:
            content += "## 注記\n\n"
            content += f"⚠️ {failed_chunks}個のチャンクで処理エラーが発生しました。\n"
            content += "音声品質やファイル形式を確認してください。\n\n"

        return content

    def _extract_start_time_from_filename(self, audio_path: str) -> Optional[datetime]:
        """
        音声ファイル名から開始時刻を抽出

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            datetime: 開始時刻（パターンに一致しない場合はNone）

        Examples:
            "2025-06-28_02_40_22.mp3" -> datetime(2025, 6, 28, 2, 40, 22)
        """
        import re

        filename = Path(audio_path).stem

        # パターン: YYYY-MM-DD_HH_MM_SS (プレフィックス/サフィックス対応)
        pattern = r"(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})"
        match = re.search(pattern, filename)

        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            try:
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                logger.warning(f"Invalid datetime in filename: {filename}")
                return None
        else:
            logger.debug(f"Filename pattern not matched: {filename}")
            return None

    def _format_time(self, seconds: float, audio_path: str = None) -> str:
        """
        時刻をフォーマット

        Args:
            seconds: 経過秒数
            audio_path: 音声ファイルのパス（ライフログモードで実時間計算に使用）

        Returns:
            str: フォーマットされた時刻
                transcriptモード: MM:SS形式
                lifelogモード: HH:MM:SS形式（ファイル名から開始時刻を取得できた場合）
        """
        # transcriptモードは従来通り相対時間
        if self.output_config.mode == "transcript":
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

        # lifelogモードで実時間計算を試行
        if self.output_config.mode == "lifelog" and audio_path:
            start_time = self._extract_start_time_from_filename(audio_path)
            if start_time:
                # 開始時刻に経過秒数を加算
                actual_time = start_time + timedelta(seconds=seconds)
                return actual_time.strftime("%H:%M:%S")

        # フォールバック: 相対時間表示
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    async def process_audio_file(self, audio_path: str, resume: bool = False) -> Dict:
        """
        音声ファイルの処理（公開API）

        Args:
            audio_path: 音声ファイルのパス
            resume: 中断したところから再開するか

        Returns:
            Dict: 処理結果
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Processing audio file: {audio_path}")
        start_time = datetime.now()

        try:
            result = await self.process_audio_pipeline(audio_path, resume)

            processing_time = datetime.now() - start_time
            result["processing_time"] = str(processing_time)
            result["status"] = "success"

            logger.info(f"Successfully processed {audio_path} in {processing_time}")
            return result

        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "processing_time": str(datetime.now() - start_time),
            }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Voice2Structured: 音声文字起こし専用パイプライン"
    )
    parser.add_argument("audio_file", nargs="?", help="処理する音声ファイルのパス")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルのパス")

    # モード選択
    parser.add_argument(
        "--mode",
        choices=["transcript", "lifelog"],
        default="transcript",
        help="処理モード: transcript (全文文字起こし) or lifelog (ライフログ)",
    )

    # 再開オプション
    parser.add_argument(
        "--resume",
        action="store_true",
        help="中断した処理を再開する",
    )

    # 設定チェックオプション
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="設定ファイルをチェックして問題を報告する",
    )

    args = parser.parse_args()

    try:
        # プロセッサーを初期化
        processor = Voice2Structured(args.config)

        # 設定チェックモード
        if args.check_config:
            print("🔍 設定チェックを実行中...")
            issues = processor.validate_configuration()

            print("\n📋 設定チェック結果:")
            for issue in issues:
                print(f"  {issue}")

            # エラーがある場合は終了
            error_count = sum(1 for issue in issues if issue.startswith("❌"))
            if error_count > 0:
                print(
                    f"\n❌ {error_count}個の問題が見つかりました。設定を確認してください。"
                )
                sys.exit(1)
            else:
                print("\n✅ 設定チェック完了！問題ありません。")
                return

        # 音声ファイルが必要な場合のチェック
        if not args.audio_file:
            print("❌ 音声ファイルのパスが指定されていません。")
            print("💡 ヒント: --check-config で設定をチェックできます。")
            sys.exit(1)

        # モードを設定
        processor.output_config.mode = args.mode

        # 非同期処理を実行
        result = asyncio.run(processor.process_audio_file(args.audio_file, args.resume))

        if result["status"] == "success":
            print("✅ 処理が完了しました!")
            print(f"📁 出力ディレクトリ: {processor.output_config.output_dir}")
            print(f"🎯 処理モード: {args.mode}")
            for output_type, file_path in result.items():
                if output_type not in ["status", "processing_time"]:
                    print(f"📄 {output_type}: {file_path}")
        else:
            print(f"❌ 処理に失敗しました: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  処理がユーザーによって中断されました")
        print("💡 ヒント: --resume オプションで続きから再開できます")
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
