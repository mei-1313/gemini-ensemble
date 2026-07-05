# gemini-ensemble

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
Clone the repository and install it in editable mode:
```bash
pip install -e .
```

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
        model="gemini-2.5-flash",
        n=3,
        strategy=reduce_critic
    )
    print("Critic Response:", response.text)

    # B. Structured Consensus (Voting)
    structured_response = await generate_ensemble(
        client=client,
        prompt=prompt,
        model="gemini-2.5-flash",
        n=3,
        strategy=reduce_voting,
        response_schema=SentimentReport
    )
    print("Voting Response:", structured_response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Class-Based Style (Specification Compatible)
You can also use class wrappers to achieve the same result.

```python
from gemini_ensemble import EnsembleClient, CriticReducer, VotingReducer

async def main():
    client = genai.Client()
    ensemble = EnsembleClient(client=client)
    
    response = await ensemble.generate(
        prompt="Explain quantum physics simply.",
        model="gemini-2.5-flash",
        n=5,
        strategy=CriticReducer() # Can specify a reducer_model inside (e.g. CriticReducer(reducer_model="gemini-2.5-pro"))
    )
    print(response.text)
```

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
リポジトリをクローンし、開発者モード（editable mode）でインストールします。
```bash
pip install -e .
```

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
        model="gemini-2.5-flash",
        n=3,
        strategy=reduce_critic
    )
    print("Critic 回答:", response.text)

    # B. 多数決合意形成 (Voting) + 構造化データ
    structured_response = await generate_ensemble(
        client=client,
        prompt=prompt,
        model="gemini-2.5-flash",
        n=3,
        strategy=reduce_voting,
        response_schema=SentimentReport
    )
    print("Voting 回答 (JSON):", structured_response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. クラスベーススタイル（仕様書互換）
技術仕様書に準拠した、従来のオブジェクト指向（クラスベース）による記述方法です。

```python
from gemini_ensemble import EnsembleClient, CriticReducer, VotingReducer

async def main():
    client = genai.Client()
    ensemble = EnsembleClient(client=client)
    
    response = await ensemble.generate(
        prompt="量子力学についてわかりやすく説明してください。",
        model="gemini-2.5-flash",
        n=5,
        strategy=CriticReducer(reducer_model="gemini-2.5-pro") # 統合のみ上位モデルを指定可能
    )
    print(response.text)
```
