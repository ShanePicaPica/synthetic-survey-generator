"""
合成數據引擎 - 根據真實數據的分佈和邏輯生成合成數據
"""
import numpy as np
import pandas as pd
from scipy import stats
import json
import re
from openai import OpenAI


class SurveySynthesizer:
    def __init__(self, real_data, parsed_xml, openai_api_key=None):
        self.real_data = real_data.copy()
        self.parsed_xml = parsed_xml
        self.questions = {q['label']: q for q in parsed_xml.get('questions', [])}
        self.column_profiles = {}
        self.correlation_matrix = None
        self.openai_client = None
if openai_api_key:
    self.openai_client = OpenAI(
        api_key=openai_api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    def analyze_real_data(self):
        """分析真實數據的統計特徵"""
        for col in self.real_data.columns:
            profile = self._profile_column(col)
            if profile:
                self.column_profiles[col] = profile

        # 計算數值/類別欄位之間的相關性
        self._compute_correlations()

        return len(self.column_profiles)

    def _profile_column(self, col):
        """分析單個欄位的分佈"""
        series = self.real_data[col]
        non_null = series.dropna()

        if len(non_null) == 0:
            return {"type": "empty", "null_rate": 1.0}

        null_rate = series.isna().mean()

        # 判斷數據類型
        if self._is_numeric_column(non_null):
            values = pd.to_numeric(non_null, errors='coerce').dropna()
            if len(values) == 0:
                return {"type": "empty", "null_rate": null_rate}

            unique_ratio = values.nunique() / len(values) if len(values) > 0 else 0

            if unique_ratio < 0.05 or values.nunique() <= 20:
                # 類別型數值（如選項代碼 1,2,3...）
                freq = values.value_counts(normalize=True)
                return {
                    "type": "categorical_numeric",
                    "null_rate": null_rate,
                    "values": freq.index.tolist(),
                    "probs": freq.values.tolist()
                }
            else:
                # 連續數值
                return {
                    "type": "continuous",
                    "null_rate": null_rate,
                    "mean": float(values.mean()),
                    "std": float(values.std()) if values.std() > 0 else 1.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "values": values.tolist()
                }
        else:
            # 文字/類別欄位
            freq = non_null.astype(str).value_counts(normalize=True)

            if freq.iloc[0] if len(freq) > 0 else 0 > 0.95 and len(freq) <= 3:
                return {
                    "type": "constant",
                    "null_rate": null_rate,
                    "value": freq.index[0] if len(freq) > 0 else ""
                }

            unique_ratio = non_null.nunique() / len(non_null)

            if unique_ratio > 0.8:
                # 開放題 / 唯一值
                return {
                    "type": "open_text",
                    "null_rate": null_rate,
                    "samples": non_null.astype(str).tolist()[:50],
                    "avg_length": non_null.astype(str).str.len().mean()
                }
            else:
                return {
                    "type": "categorical",
                    "null_rate": null_rate,
                    "values": freq.index.tolist(),
                    "probs": freq.values.tolist()
                }

    def _is_numeric_column(self, series):
        """判斷是否為數值欄位"""
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except (ValueError, TypeError):
            try:
                converted = pd.to_numeric(series, errors='coerce')
                return converted.notna().mean() > 0.8
            except Exception:
                return False

    def _compute_correlations(self):
        """計算欄位間相關性矩陣"""
        # 選取可用於相關性計算的欄位
        numeric_cols = []
        for col, profile in self.column_profiles.items():
            if profile['type'] in ('categorical_numeric', 'continuous'):
                numeric_cols.append(col)

        if len(numeric_cols) < 2:
            self.correlation_matrix = None
            self.corr_columns = []
            return

        # 構建數值矩陣
        df_numeric = pd.DataFrame()
        for col in numeric_cols:
            df_numeric[col] = pd.to_numeric(
                self.real_data[col], errors='coerce'
            )

        # 用中位數填充缺失值
        df_numeric = df_numeric.fillna(df_numeric.median())

        # 計算 Spearman 相關係數（對非正態分佈更穩健）
        self.correlation_matrix = df_numeric.corr(method='spearman')
        self.corr_columns = numeric_cols

    def synthesize(self, n_samples, progress_callback=None):
        """生成合成數據"""
        if not self.column_profiles:
            self.analyze_real_data()

        synthetic = pd.DataFrame()
        columns_ordered = list(self.real_data.columns)

        # 第一步：識別關鍵分組/路由變數
        key_vars = self._identify_key_variables()

        # 第二步：先生成關鍵路由變數
        if progress_callback:
            progress_callback(0.1, "正在生成關鍵路由變數...")

        for col in key_vars:
            if col in self.column_profiles:
                synthetic[col] = self._generate_column(
                    col, self.column_profiles[col], n_samples
                )

        # 第三步：根據相關性生成其他數值/類別欄位
        if progress_callback:
            progress_callback(0.3, "正在生成數據並保持變數間相關性...")

        remaining_cols = [c for c in columns_ordered
                          if c not in key_vars and c in self.column_profiles]

        for i, col in enumerate(remaining_cols):
            profile = self.column_profiles[col]

            if profile['type'] == 'open_text':
                # 開放題最後用 AI 處理
                continue

            synthetic[col] = self._generate_column_with_correlation(
                col, profile, n_samples, synthetic
            )

            if progress_callback and i % 10 == 0:
                pct = 0.3 + 0.4 * (i / max(len(remaining_cols), 1))
                progress_callback(pct, f"處理欄位 {col}...")

        # 第四步：應用跳題邏輯（將不適用的欄位設為空白）
        if progress_callback:
            progress_callback(0.7, "正在應用跳題邏輯...")

        synthetic = self._apply_skip_logic(synthetic)

        # 第五步：生成開放題
        if progress_callback:
            progress_callback(0.8, "正在生成開放題文字...")

        open_text_cols = [
            c for c in columns_ordered
            if c in self.column_profiles
            and self.column_profiles[c]['type'] == 'open_text'
        ]

        for col in open_text_cols:
            profile = self.column_profiles[col]
            synthetic[col] = self._generate_open_text(
                col, profile, n_samples, synthetic
            )

        # 第六步：應用驗證規則
        if progress_callback:
            progress_callback(0.9, "正在驗證數據一致性...")

        synthetic = self._apply_validation_rules(synthetic)

        # 第七步：添加合成標記
        synthetic['is_synthetic'] = 1

        # 確保欄位順序一致
        final_cols = [c for c in columns_ordered if c in synthetic.columns]
        final_cols.append('is_synthetic')
        synthetic = synthetic[final_cols]

        if progress_callback:
            progress_callback(1.0, "合成完成！")

        return synthetic

    def _identify_key_variables(self):
        """識別關鍵路由/分組變數"""
        key_vars = []

        # 從 XML 中識別被引用為跳題條件的變數
        all_conds = set()
        for q in self.parsed_xml.get('questions', []):
            if q.get('cond'):
                # 提取條件中引用的變數名
                cond_vars = re.findall(r'([A-Za-z]\w*)\.[rc]\d+', q['cond'])
                all_conds.update(cond_vars)

        # 已知的關鍵路由變數
        known_keys = ['market', 'S6', 'hid_S6', 'hid_S9', 'sample',
                       'S2', 'S3', 'S5', 'S7', 'S18', 'S22', 'S12']

        for col in self.real_data.columns:
            col_base = col.split('_')[0] if '_' in col else col
            if col_base in all_conds or col_base in known_keys:
                if col in self.column_profiles:
                    key_vars.append(col)

        # 如果沒找到，就用所有在前20列中的欄位
        if not key_vars:
            key_vars = [c for c in list(self.real_data.columns)[:20]
                        if c in self.column_profiles]

        return key_vars

    def _generate_column(self, col, profile, n):
        """根據分佈生成單個欄位"""
        col_type = profile['type']

        if col_type in ('categorical', 'categorical_numeric'):
            values = profile['values']
            probs = profile['probs']
            # 確保概率和為1
            probs = np.array(probs, dtype=float)
            probs = probs / probs.sum()
            generated = np.random.choice(values, size=n, p=probs)
            # 隨機加入空值
            if profile.get('null_rate', 0) > 0.01:
                null_mask = np.random.random(n) < profile['null_rate']
                generated = pd.Series(generated)
                generated[null_mask] = np.nan
            return generated

        elif col_type == 'continuous':
            # 用核密度估計來生成更真實的分佈
            real_values = np.array(profile['values'])
            if len(real_values) > 5:
                try:
                    kde = stats.gaussian_kde(real_values)
                    generated = kde.resample(n).flatten()
                    # 裁剪到合理範圍
                    generated = np.clip(generated, profile['min'], profile['max'])
                except Exception:
                    generated = np.random.normal(
                        profile['mean'], profile['std'], n
                    )
                    generated = np.clip(generated, profile['min'], profile['max'])
            else:
                generated = np.random.choice(real_values, size=n)

            if profile.get('null_rate', 0) > 0.01:
                null_mask = np.random.random(n) < profile['null_rate']
                generated = pd.Series(generated)
                generated[null_mask] = np.nan
            return generated

        elif col_type == 'constant':
            generated = pd.Series([profile['value']] * n)
            if profile.get('null_rate', 0) > 0.01:
                null_mask = np.random.random(n) < profile['null_rate']
                generated[null_mask] = np.nan
            return generated

        elif col_type == 'empty':
            return pd.Series([np.nan] * n)

        else:
            return pd.Series([np.nan] * n)

    def _generate_column_with_correlation(self, col, profile, n, existing_df):
        """生成保持相關性的欄位"""
        if (self.correlation_matrix is None
                or col not in self.corr_columns
                or profile['type'] not in ('categorical_numeric', 'continuous')):
            return self._generate_column(col, profile, n)

        # 找到與此欄位相關性最強的已生成欄位
        best_corr_col = None
        best_corr_val = 0

        for existing_col in existing_df.columns:
            if (existing_col in self.corr_columns
                    and existing_col in self.correlation_matrix.columns):
                try:
                    corr_val = abs(
                        self.correlation_matrix.loc[col, existing_col]
                    )
                    if corr_val > best_corr_val and corr_val > 0.15:
                        best_corr_val = corr_val
                        best_corr_col = existing_col
                except KeyError:
                    continue

        if best_corr_col is None or best_corr_val < 0.15:
            return self._generate_column(col, profile, n)

        # 使用條件分佈
        try:
            ref_values = pd.to_numeric(
                existing_df[best_corr_col], errors='coerce'
            )
            real_ref = pd.to_numeric(
                self.real_data[best_corr_col], errors='coerce'
            )
            real_target = pd.to_numeric(
                self.real_data[col], errors='coerce'
            )

            result = pd.Series(index=range(n), dtype=float)

            # 按參考欄位的值分組，保持條件分佈
            ref_unique = ref_values.dropna().unique()

            for ref_val in ref_unique:
                mask = ref_values == ref_val
                count = mask.sum()
                if count == 0:
                    continue

                # 在真實數據中找相同參考值的目標分佈
                real_mask = real_ref == ref_val
                conditional_values = real_target[real_mask].dropna()

                if len(conditional_values) > 0:
                    if profile['type'] == 'categorical_numeric':
                        freq = conditional_values.value_counts(normalize=True)
                        probs = freq.values / freq.values.sum()
                        sampled = np.random.choice(
                            freq.index, size=count, p=probs
                        )
                    else:
                        sampled = np.random.choice(
                            conditional_values.values, size=count
                        )
                    result[mask] = sampled
                else:
                    result[mask] = self._generate_column(
                        col, profile, count
                    ).values[:count]

            # 處理剩餘的 NaN
            remaining_null = result.isna()
            if remaining_null.any():
                fill_values = self._generate_column(
                    col, profile, remaining_null.sum()
                )
                result[remaining_null] = fill_values.values[:remaining_null.sum()]

            return result

        except Exception:
            return self._generate_column(col, profile, n)

    def _apply_skip_logic(self, df):
        """根據跳題邏輯將不適用的欄位設為空值"""
        for q_label, q_info in self.questions.items():
            cond = q_info.get('cond', '')
            if not cond:
                continue

            # 找到所有屬於此問題的欄位
            q_columns = [c for c in df.columns if self._column_belongs_to_question(c, q_label)]

            if not q_columns:
                continue

            # 評估條件
            try:
                skip_mask = self._evaluate_condition(cond, df)
                if skip_mask is not None:
                    for col in q_columns:
                        df.loc[~skip_mask, col] = np.nan
            except Exception:
                continue

        return df

    def _column_belongs_to_question(self, col_name, q_label):
        """判斷欄位是否屬於某個問題"""
        if col_name == q_label:
            return True
        if col_name.startswith(q_label + '_'):
            return True
        if col_name.startswith(q_label + '.'):
            return True
        # 處理 grid 問題的欄位命名格式
        pattern = rf'^{re.escape(q_label)}[._]r\d+'
        if re.match(pattern, col_name):
            return True
        return False

    def _evaluate_condition(self, cond, df):
        """評估跳題條件，返回布爾遮罩"""
        try:
            # 簡單條件解析
            # 格式如: "S6.r1", "hid_S6.r1", "S12.r4 or S12.r5"

            # 替換常見模式
            eval_cond = cond

            # 處理 "not X.rN" 格式
            eval_cond = re.sub(
                r'not\s+(\w+)\.r(\d+)',
                lambda m: f'(df.get("{m.group(1)}", pd.Series([np.nan]*len(df))) != {m.group(2)})',
                eval_cond
            )

            # 處理 "X.rN" 格式 (單選題某選項被選中)
            eval_cond = re.sub(
                r'(\w+)\.r(\d+)',
                lambda m: f'(pd.to_numeric(df.get("{m.group(1)}", pd.Series([np.nan]*len(df))), errors="coerce") == {m.group(2)})',
                eval_cond
            )

            # 處理 "X.any" 格式
            eval_cond = re.sub(
                r'(\w+)\.any',
                lambda m: f'(df.get("{m.group(1)}", pd.Series([np.nan]*len(df))).notna())',
                eval_cond
            )

            # 替換邏輯運算符
            eval_cond = eval_cond.replace(' and ', ' & ')
            eval_cond = eval_cond.replace(' or ', ' | ')

            result = eval(eval_cond)
            if isinstance(result, pd.Series):
                return result.astype(bool)
            return None

        except Exception:
            return None

    def _generate_open_text(self, col, profile, n, existing_df):
        """生成開放題文字"""
        null_rate = profile.get('null_rate', 0)
        samples = profile.get('samples', [])

        # 先確定哪些行需要有值（根據跳題邏輯）
        needs_value = pd.Series([True] * n)
        q_label = col.split('_')[0] if '_' in col else col.split('.')[0]

        if q_label in self.questions:
            cond = self.questions[q_label].get('cond', '')
            if cond:
                try:
                    mask = self._evaluate_condition(cond, existing_df)
                    if mask is not None:
                        needs_value = mask
                except Exception:
                    pass

        result = pd.Series([np.nan] * n)

        # 需要生成文字的索引
        text_indices = needs_value[needs_value].index.tolist()

        if not text_indices or not samples:
            return result

        if self.openai_client and len(samples) >= 3:
            # 用 AI 生成
            try:
                generated_texts = self._ai_generate_texts(
                    col, samples, len(text_indices)
                )
                for i, idx in enumerate(text_indices):
                    if i < len(generated_texts):
                        result[idx] = generated_texts[i]
            except Exception:
                # AI 失敗，回退到隨機抽樣
                for idx in text_indices:
                    result[idx] = np.random.choice(samples)
        else:
            # 隨機抽樣現有回答
            for idx in text_indices:
                result[idx] = np.random.choice(samples)

        # 加入隨機空值
        if null_rate > 0.01:
            null_mask = np.random.random(n) < null_rate
            result[null_mask & needs_value] = np.nan

        return result

    def _ai_generate_texts(self, col_name, samples, n_needed):
        """使用 AI 生成開放題文字"""
        # 找到問題標題
        q_label = col_name.split('_')[0] if '_' in col_name else col_name
        q_title = ''
        if q_label in self.questions:
            q_title = self.questions[q_label].get('title', '')

        # 取樣本示例
        sample_texts = samples[:10]
        sample_str = '\n'.join([f'- {s}' for s in sample_texts])

        batch_size = min(n_needed, 20)
        all_texts = []

        while len(all_texts) < n_needed:
            remaining = n_needed - len(all_texts)
            current_batch = min(batch_size, remaining)

            prompt = f"""You are generating synthetic survey responses. 
Generate {current_batch} unique, realistic responses for the following survey question.

Question: {q_title if q_title else col_name}

Here are example real responses for reference (match the language, style, length, and tone):
{sample_str}

Rules:
1. Each response should be unique but similar in style and length to the examples
2. Use the same language as the examples
3. Keep responses realistic and varied
4. Return ONLY the responses, one per line, numbered 1-{current_batch}

Generate {current_batch} responses:"""

            try:
response = self.openai_client.chat.completions.create(
    model="google/gemma-3-27b-it:free",
    messages=[
        {"role": "system",
         "content": "You are a market research data synthesis assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
    max_tokens=2000,
    extra_headers={
        "HTTP-Referer": "https://synthetic-survey-generator.streamlit.app",
        "X-Title": "Synthetic Survey Generator"
    }
)

                text = response.choices[0].message.content
                lines = text.strip().split('\n')

                for line in lines:
                    # 去掉編號
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if cleaned and len(cleaned) > 1:
                        all_texts.append(cleaned)

            except Exception:
                # 回退到隨機抽樣
                for _ in range(current_batch):
                    all_texts.append(np.random.choice(samples))

        return all_texts[:n_needed]

    def _apply_validation_rules(self, df):
        """應用特定的驗證規則"""
        # 規則1: S17 不應大於 S16
        if 'S17' in df.columns and 'S16' in df.columns:
            s17 = pd.to_numeric(df['S17'], errors='coerce')
            s16 = pd.to_numeric(df['S16'], errors='coerce')
            violation = (s17 > s16) & s17.notna() & s16.notna()
            if violation.any():
                df.loc[violation, 'S17'] = df.loc[violation, 'S16']

        # 規則2: S21 各項加總應為100
        s21_cols = [c for c in df.columns if re.match(r'S21.*c1', c)]
        if s21_cols:
            for idx in df.index:
                vals = pd.to_numeric(
                    df.loc[idx, s21_cols], errors='coerce'
                ).dropna()
                if len(vals) > 0:
                    total = vals.sum()
                    if total != 0 and total != 100:
                        scale_factor = 100.0 / total
                        for col in s21_cols:
                            if pd.notna(df.loc[idx, col]):
                                df.loc[idx, col] = round(
                                    float(df.loc[idx, col]) * scale_factor
                                )

        # 規則3: S11 r1 + r2 = 7
        s11_cols = [c for c in df.columns if c.startswith('S11')]
        if len(s11_cols) >= 2:
            r1_col = [c for c in s11_cols if 'r1' in c]
            r2_col = [c for c in s11_cols if 'r2' in c]
            if r1_col and r2_col:
                for idx in df.index:
                    v1 = pd.to_numeric(
                        df.loc[idx, r1_col[0]], errors='coerce'
                    )
                    if pd.notna(v1):
                        df.loc[idx, r2_col[0]] = 7 - v1

        return df


def generate_quality_report(real_data, synthetic_data):
    """生成數據質量對比報告"""
    report = {
        "sample_sizes": {
            "real": len(real_data),
            "synthetic": len(synthetic_data)
        },
        "column_comparisons": []
    }

    common_cols = [
        c for c in real_data.columns
        if c in synthetic_data.columns and c != 'is_synthetic'
    ]

    for col in common_cols[:50]:  # 限制前50個欄位
        real_series = real_data[col].dropna()
        synth_series = synthetic_data[col].dropna()

        if len(real_series) == 0 or len(synth_series) == 0:
            continue

        comparison = {"column": col}

        try:
            real_numeric = pd.to_numeric(real_series, errors='coerce').dropna()
            synth_numeric = pd.to_numeric(
                synth_series, errors='coerce'
            ).dropna()

            if len(real_numeric) > 5 and len(synth_numeric) > 5:
                # 數值欄位比較
                unique_ratio = real_numeric.nunique() / len(real_numeric)

                if unique_ratio < 0.05 or real_numeric.nunique() <= 20:
                    # 類別型 - 用卡方檢驗
                    real_freq = real_numeric.value_counts(normalize=True)
                    synth_freq = synth_numeric.value_counts(normalize=True)

                    comparison["type"] = "categorical"
                    comparison["real_distribution"] = {
                        str(k): round(v, 4)
                        for k, v in real_freq.head(10).items()
                    }
                    comparison["synthetic_distribution"] = {
                        str(k): round(v, 4)
                        for k, v in synth_freq.head(10).items()
                    }

                    # 簡化的分佈相似度
                    all_vals = set(real_freq.index) | set(synth_freq.index)
                    diff = sum(
                        abs(real_freq.get(v, 0) - synth_freq.get(v, 0))
                        for v in all_vals
                    )
                    comparison["distribution_similarity"] = round(
                        max(0, 1 - diff / 2), 4
                    )
                else:
                    comparison["type"] = "continuous"
                    comparison["real_stats"] = {
                        "mean": round(real_numeric.mean(), 2),
                        "std": round(real_numeric.std(), 2)
                    }
                    comparison["synthetic_stats"] = {
                        "mean": round(synth_numeric.mean(), 2),
                        "std": round(synth_numeric.std(), 2)
                    }
            else:
                comparison["type"] = "text"
                comparison["real_unique"] = int(real_series.nunique())
                comparison["synthetic_unique"] = int(synth_series.nunique())
        except Exception:
            comparison["type"] = "error"

        report["column_comparisons"].append(comparison)

    # 計算整體品質分數
    similarities = [
        c.get("distribution_similarity", None)
        for c in report["column_comparisons"]
    ]
    valid_sims = [s for s in similarities if s is not None]
    report["overall_quality_score"] = (
        round(np.mean(valid_sims), 4) if valid_sims else None
    )

    return report
