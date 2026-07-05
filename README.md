# gemini-ensemble

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance LLM ensemble control library for Python based on the Google `google-genai` SDK.  
Googleの `google-genai` SDKをベースに構築された、Python向けの高性能LLMアンサンブル制御ライブラリ。

---

## English Documentation

### Overview
`gemini-ensemble` runs $N$ instances of a base model in parallel asynchronously, scattering the temperatures automatically between `0.0` and `1.0` to sample the LLM's probability space (Dynamic Temperature Scattering). The outputs are then consolidated using a specified reducer model and strategy.

### Key Features
1. **Dynamic Temperature Scattering**: Automatically generates a temperature gradient from `0.0` to `1.0` across the $N$ parallel instances.
2. **Flexibly Targeted Reducer**: Decouples the base models from the final reduction model. If not specified, the base model is reused for reduction.
3. **Dual API Style**: Supports both pure functional programming style and traditional class-based wrapping.

### Installation
It is highly recommended to use a virtual environment to avoid installing packages into your global system environment.

Create and activate a virtual environment, then install the package in editable mode:
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the package
pip install -e .
```

### API Key Setup
You need a Gemini API key. Set it as an environment variable or create a `.env` file in your project root directory:
```env
GEMINI_API_KEY=your_api_key_here
```
The library and CLI automatically load this key using `python-dotenv`.

### Reduction Strategies (Critic vs Voting)
The library provides two distinct strategy functions to consolidate parallel outputs:

| Strategy | `Critic` (Default) | `Voting` |
| :--- | :--- | :--- |
| **Primary Use Case** | Creative writing, reasoning, synthesis, summaries | Classification, entity extraction, structured JSON data |
| **Mechanism** | Critically reviews all candidates to eliminate hallucinations and contradictions, then **synthesizes** a single response. | Analyzes candidate answers to find the consensus, then **extracts** the majority vote according to the schema. |
| **Output Format** | Free-form text (Markdown) | Structured JSON data (Pydantic) |
| **How to Use** | Pass `reduce_critic` / `CriticReducer()` | Pass `reduce_voting` / `VotingReducer()` |

### Usage

#### 1. Pure Functional Style (Recommended)
This approach avoids class instantiation and uses pure functions.

```python
import asyncio
from google import genai
from pydantic import BaseModel
from gemini_ensemble import generate_ensemble, reduce_voting, reduce_critic

# Structured Output Schema
class SentimentReport(BaseModel):
    sentiment: str
    confidence: float
    reason: str

async def main():
    client = genai.Client()
    prompt = "Review: 'Great hardware but terrible battery life.' Determine the sentiment."

    # A. Text Synthesis (Critic)
    response = await generate_ensemble(
        client=client,
        prompt=prompt,
        model="gemini-3.1-flash-lite",
        n=3,
        strategy=reduce_critic
    )
    print("Critic Response:", response.text)

    # B. Structured Consensus (Voting)
    structured_response = await generate_ensemble(
        client=client,
        prompt=prompt,
        model="gemini-3.1-flash-lite",
        n=3,
        strategy=reduce_voting,
        response_schema=SentimentReport
    )
    print("Voting Response:", structured_response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Class-Based Style
You can also use class wrappers to achieve the same result.

```python
from gemini_ensemble import EnsembleClient, CriticReducer, VotingReducer

async def main():
    client = genai.Client()
    ensemble = EnsembleClient(client=client)
    
    response = await ensemble.generate(
        prompt="Explain quantum physics simply.",
        model="gemini-3.1-flash-lite",
        n=5,
        strategy=CriticReducer() # Can specify a reducer_model inside (e.g. CriticReducer(reducer_model="gemini-3.1-pro"))
    )
    print(response.text)
```

#### 3. Command Line Interface (CLI)
You can run the ensemble directly from your terminal by passing a text file containing the prompt.

```bash
# Run with a prompt file (uses gemini-3.1-flash-lite and CriticReducer by default)
gemini-ensemble prompt.txt

# Specify parameters (n=5, output language=Japanese, VotingReducer)
gemini-ensemble prompt.txt -n 5 -l Japanese -r voting
```

##### CLI Options:
*   `file` (positional, required): Path to the text file containing the prompt.
*   `-n` (optional, default: `3`): Number of parallel model instances.
*   `-m`, `--model` (optional, default: `gemini-3.1-flash-lite`): Base model to run in parallel.
*   `-r`, `--reducer` (optional, default: `critic`, choices: `critic`, `voting`): Reduction strategy.
*   `-l`, `--language` (optional): Target language for the final output (e.g. `Japanese`, `English`).
*   `--reducer-model` (optional): Specific model to use for reduction (defaults to the base model).

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 日本語ドキュメント

### 概要
`gemini-ensemble` は、Googleの `google-genai` SDK をベースとしたPython用LLMアンサンブル制御ライブラリです。

ベースモデルを並列かつ非同期で $N$ 台同時に実行し、各インスタンスへ `0.0`（確実性重視）から `1.0`（創造性/多様性重視）までの異なる温度（Temperature）をグラデーション状に自動で分散して割り当てます（Dynamic Temperature Scattering）。これにより、LLMの確率空間から多様で堅牢な出力をサンプリングしたのち、指定された統合モデル（Reducer）を用いて高品質な単一の回答へまとめ上げます。

### 主な特徴
1. **Dynamic Temperature Scattering (温度自動分散)**: $N$ 台の並列モデルに対し、`0.0` から `1.0` までの温度をグラデーション状に自動で散布して一斉駆動します。これにより、出力の偏りを抑えたサンプリングが可能になります。
2. **柔軟な統合モデル設計 (Flexibly Targeted Reducer)**: 並列推論を行う「ベースモデル」と、最終回答の検証・マージを行う「統合モデル（`reducer_model`）」を分離して指定できます。未指定の場合は、並列層と同じモデルが統合フェーズも兼任します。
3. **2つのAPIスタイルをサポート**: 状態を持たないシンプルな「関数型スタイル」と、オブジェクト指向に基づいた従来の「クラスベーススタイル」のどちらでも記述可能です。

### インストール方法
システムのグローバルなPython環境への干渉を避けるため、仮想環境（venv）の使用を強く推奨します。

仮想環境を作成・有効化（アクティベート）した上で、インストールを行ってください。
```bash
# 仮想環境を作成
python3 -m venv .venv

# 仮想環境を有効化（アクティベート）
source .venv/bin/activate  # Windowsの場合は `.venv\Scripts\activate`

# 開発者モードでインストールを実行
pip install -e .
```

### APIキーの設定
GeminiのAPIキーが必要です。環境変数として設定するか、プロジェクトのルートディレクトリに `.env` ファイルを作成して記述してください。
```env
GEMINI_API_KEY=your_api_key_here
```
ライブラリおよびCLIは `python-dotenv` を使ってこのキーを自動的にロードします。

### 統合戦略（Critic と Voting の違い）
並列で出力された結果を集約（Reduce）するために、2つの異なる戦略を提供しています。

| 項目 | 批判的統合 (`Critic`) [デフォルト] | 多数決合意形成 (`Voting`) |
| :--- | :--- | :--- |
| **主なユースケース** | 自由文生成、推論、要約、レポート作成 | 分類、真偽判定、構造化データ抽出（JSON） |
| **統合の仕組み** | 各回答を比較検証し、ハルシネーション（誤情報）や矛盾を排除して**単一の回答として再合成する**。 | 各回答の整合性を分析し、最も多数を占める結果（マジョリティ）を**決定・抽出する**。 |
| **出力形式** | 自由なテキスト（Markdown形式など） | 構造化データ（JSON形式、Pydanticスキーマ） |
| **使用方法** | `reduce_critic` または `CriticReducer()` を指定 | `reduce_voting` または `VotingReducer()` を指定 |

### 使い方

#### 1. 関数型スタイル（推奨）
クラスのインスタンス化を行わず、状態を持たない純粋関数のみを組み合わせて実行するシンプルなスタイルです。

```python
import asyncio
from google import genai
from pydantic import BaseModel
from gemini_ensemble import generate_ensemble, reduce_voting, reduce_critic

# 構造化出力用のスキーマ定義
class SentimentReport(BaseModel):
    sentiment: str
    confidence: float
    reason: str

async def main():
    client = genai.Client()
    prompt = "レビュー: 'ハードウェアは素晴らしいが、バッテリー持ちは最悪です。' この感情を判定してください。"

    # A. 批判的統合 (Critic)
    response = await generate_ensemble(
        client=client,
        prompt=prompt,
        model="gemini-3.1-flash-lite",
        n=3,
        strategy=reduce_critic
    )
    print("Critic 回答:", response.text)

    # B. 多数決合意形成 (Voting) + 構造化データ
    structured_response = await generate_ensemble(
        client=client,
        prompt=prompt,
        model="gemini-3.1-flash-lite",
        n=3,
        strategy=reduce_voting,
        response_schema=SentimentReport
    )
    print("Voting 回答 (JSON):", structured_response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. クラスベーススタイル
オブジェクト指向（クラスベース）による記述方法です。

```python
from gemini_ensemble import EnsembleClient, CriticReducer, VotingReducer

async def main():
    client = genai.Client()
    ensemble = EnsembleClient(client=client)
    
    response = await ensemble.generate(
        prompt="量子力学についてわかりやすく説明してください。",
        model="gemini-3.1-flash-lite",
        n=5,
        strategy=CriticReducer(reducer_model="gemini-3.1-pro") # 統合のみ上位モデルを指定可能
    )
    print(response.text)
```

#### 3. コマンドラインインターフェース (CLI)
プロンプトが記述されたテキストファイルを引数に渡し、ターミナルから直接アンサンブルを実行できます。

```bash
# プロンプトファイルを渡して実行 (デフォルトで gemini-3.1-flash-lite および CriticReducer が使用されます)
gemini-ensemble prompt.txt

# オプション指定 (並列数n=5、出力言語=日本語、VotingReducerを使用)
gemini-ensemble prompt.txt -n 5 -l Japanese -r voting
```

##### CLI オプション一覧:
*   `file` (位置引数、必須): プロンプトが記述されたテキストファイルのパス。
*   `-n` (任意、デフォルト: `3`): 並列駆動するベースモデルの数。
*   `-m`, `--model` (任意、デフォルト: `gemini-3.1-flash-lite`): 並列実行するベースモデル名。
*   `-r`, `--reducer` (任意、デフォルト: `critic`、選択肢: `critic`, `voting`): 出力結果の統合戦略。
*   `-l`, `--language` (任意): 最終回答の出力言語 (例: `Japanese`, `English`)。
*   `--reducer-model` (任意): 最終統合に用いるモデル名 (未指定の場合はベースモデルを使用)。

### ライセンス
このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。
