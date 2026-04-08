"""
合成市場調研數據工具 - 網頁介面
支持 OpenAI 和 OpenRouter (免費模型)
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from xml_parser import parse_decipher_xml, get_question_summary
from synthesizer import SurveySynthesizer, generate_quality_report

# ===== 頁面設定 =====
st.set_page_config(
    page_title="市場調研數據合成工具",
    page_icon="📊",
    layout="wide"
)

st.title("📊 市場調研數據合成工具")
st.markdown("上傳真實數據和問卷邏輯 → 設定筆數 → 生成合成數據")

# ===== 側邊欄設定 =====
with st.sidebar:
    st.header("⚙️ 設定")

    api_provider = st.selectbox(
        "AI 服務提供商",
        ["OpenRouter (免費模型可用)", "OpenAI"],
        help="OpenRouter 提供免費的 Gemma 模型，適合測試"
    )

    if api_provider == "OpenRouter (免費模型可用)":
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="前往 https://openrouter.ai/keys 獲取。格式: sk-or-v1-xxxx..."
        )
        model_options = {
            "Gemma 3 27B (免費)": "google/gemma-3-27b-it:free",
            "Llama 3.1 8B (免費)": "meta-llama/llama-3.1-8b-instruct:free",
            "Qwen3 8B (免費)": "qwen/qwen3-8b:free",
            "Mistral 7B (免費)": "mistralai/mistral-7b-instruct:free",
            "DeepSeek V3 (免費)": "deepseek/deepseek-chat-v3-0324:free"
        }
        selected_model = st.selectbox(
            "選擇模型",
            list(model_options.keys()),
            help="免費模型有速率限制，建議生成量不超過500筆"
        )
        model_id = model_options[selected_model]

        st.info(f"📡 當前模型: {model_id}")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="格式: sk-xxxx..."
        )
        model_options_openai = {
            "GPT-4o Mini (推薦，便宜)": "gpt-4o-mini",
            "GPT-4o (更高質量)": "gpt-4o",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        selected_model = st.selectbox(
            "選擇模型",
            list(model_options_openai.keys())
        )
        model_id = model_options_openai[selected_model]

    st.divider()

    st.markdown("### 📖 使用說明")
    st.markdown("""
    **操作步驟：**
    1. 左側設定 AI API Key
    2. 上傳真實數據 (Excel)
    3. 上傳問卷 XML (Decipher)
    4. 設定生成筆數
    5. 點擊「開始生成」
    6. 下載合成數據

    **注意事項：**
    - 不提供 API Key 也可使用，但開放題將用隨機抽樣替代
    - 合成數據以 `is_synthetic=1` 標記
    - 建議生成量不超過真實數據的 5 倍
    """)

    st.divider()
    st.markdown("### 🔗 相關連結")
    st.markdown("- [OpenRouter](https://openrouter.ai/keys) - 免費 API Key")
    st.markdown("- [OpenAI](https://platform.openai.com/api-keys) - 付費 API Key")

# ===== 主介面 =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 步驟一：上傳真實數據")
    data_file = st.file_uploader(
        "上傳 Excel 文件 (.xlsx)",
        type=['xlsx', 'xls'],
        help="包含真實調研數據的 Excel 文件"
    )

with col2:
    st.subheader("📄 步驟二：上傳問卷邏輯")
    xml_file = st.file_uploader(
        "上傳 Decipher XML 文件 (.xml 或 .txt)",
        type=['xml', 'txt'],
        help="從 Decipher 導出的問卷 XML 代碼"
    )

# ===== 數據預覽 =====
real_data = None
parsed_xml = None

if data_file:
    try:
        xls = pd.ExcelFile(data_file)
        sheet_names = xls.sheet_names

        if len(sheet_names) > 1:
            selected_sheet = st.selectbox(
                "選擇數據所在的 Sheet",
                sheet_names
            )
        else:
            selected_sheet = sheet_names[0]

        real_data = pd.read_excel(data_file, sheet_name=selected_sheet)

        with st.expander("📊 真實數據預覽", expanded=False):
            st.write(
                f"**行數:** {len(real_data)} | "
                f"**欄位數:** {len(real_data.columns)}"
            )
            st.dataframe(real_data.head(10), use_container_width=True)

            col_info = pd.DataFrame({
                '欄位': real_data.columns,
                '非空值數': real_data.notna().sum().values,
                '空值率': (
                    real_data.isna().mean() * 100
                ).round(1).astype(str) + '%',
                '唯一值數': real_data.nunique().values
            })
            st.dataframe(col_info, use_container_width=True, height=300)

    except Exception as e:
        st.error(f"讀取 Excel 失敗: {str(e)}")

if xml_file:
    try:
        xml_content = xml_file.read().decode('utf-8')
        parsed_xml = parse_decipher_xml(xml_content)

        if 'error' in parsed_xml:
            st.warning(f"XML 解析警告: {parsed_xml.get('error')}")

        with st.expander("📋 問卷結構預覽", expanded=False):
            summary = get_question_summary(parsed_xml)
            st.write(f"**問卷名稱:** {summary['survey_name']}")
            st.write(f"**解析出的問題數:** {summary['total_questions']}")

            st.write("**題目類型分佈:**")
            type_df = pd.DataFrame([
                {"類型": k, "數量": v}
                for k, v in summary['type_breakdown'].items()
            ])
            st.dataframe(type_df, use_container_width=True)

            st.write(
                f"**含跳題邏輯的問題:** "
                f"{summary['conditional_questions']} 題"
            )

            st.write("**問題列表:**")
            q_df = pd.DataFrame(summary['questions_list'])
            st.dataframe(q_df, use_container_width=True, height=400)

    except Exception as e:
        st.error(f"解析 XML 失敗: {str(e)}")

# ===== 生成設定 =====
st.divider()
st.subheader("🔧 步驟三：設定生成參數")

col_a, col_b, col_c = st.columns(3)

with col_a:
    n_samples = st.number_input(
        "生成筆數",
        min_value=10,
        max_value=10000,
        value=200,
        step=50,
        help="建議不超過真實數據的 5 倍"
    )

with col_b:
    if real_data is not None:
        st.metric("真實數據筆數", len(real_data))

with col_c:
    if real_data is not None:
        st.metric("合成後總計", len(real_data) + n_samples)

# ===== 進階選項 =====
with st.expander("🔧 進階選項", expanded=False):
    preserve_correlation = st.checkbox(
        "保持變數間相關性",
        value=True,
        help="啟用後會分析並保持變數之間的統計關聯"
    )
    apply_skip_logic = st.checkbox(
        "應用跳題邏輯",
        value=True,
        help="根據 XML 中的條件邏輯，將不適用的欄位設為空白"
    )
    generate_open_text = st.checkbox(
        "使用 AI 生成開放題",
        value=True if api_key else False,
        help="需要 API Key。關閉後開放題將使用隨機抽樣"
    )

# ===== 生成按鈕 =====
st.divider()

if real_data is not None and parsed_xml is not None:
    if st.button(
        "🚀 開始生成合成數據",
        type="primary",
        use_container_width=True
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(min(pct, 1.0))
            status_text.text(msg)

        try:
            # 初始化合成器
            synthesizer = SurveySynthesizer(
                real_data=real_data,
                parsed_xml=parsed_xml,
                openai_api_key=api_key if (api_key and generate_open_text) else None,
                api_provider=api_provider,
                model_id=model_id
            )

            # 分析真實數據
            update_progress(0.05, "正在分析真實數據分佈...")
            n_profiled = synthesizer.analyze_real_data()
            update_progress(
                0.1,
                f"已分析 {n_profiled} 個欄位的分佈特徵"
            )

            # 生成合成數據
            synthetic_data = synthesizer.synthesize(
                n_samples=n_samples,
                progress_callback=update_progress
            )

            progress_bar.progress(1.0)
            status_text.text("✅ 合成完成！")

            # 顯示結果
            st.success(
                f"成功生成 {len(synthetic_data)} 筆合成數據！"
            )

            # 質量報告
            with st.expander("📈 數據質量報告", expanded=True):
                report = generate_quality_report(real_data, synthetic_data)

                if report.get('overall_quality_score'):
                    score = report['overall_quality_score']
                    if score > 0.85:
                        score_color = "green"
                        score_label = "優秀"
                    elif score > 0.7:
                        score_color = "orange"
                        score_label = "良好"
                    else:
                        score_color = "red"
                        score_label = "需改進"

                    st.markdown(
                        f"### 整體分佈相似度: "
                        f":{score_color}[{score:.1%} ({score_label})]"
                    )

                comparisons = report.get('column_comparisons', [])
                cat_comparisons = [
                    c for c in comparisons
                    if c.get('type') == 'categorical'
                    and c.get('distribution_similarity') is not None
                ]

                if cat_comparisons:
                    st.write("**各欄位分佈相似度：**")
                    comp_df = pd.DataFrame([
                        {
                            "欄位": c['column'],
                            "相似度": f"{c['distribution_similarity']:.1%}",
                            "相似度值": c['distribution_similarity']
                        }
                        for c in sorted(
                            cat_comparisons,
                            key=lambda x: x['distribution_similarity']
                        )
                    ])

                    # 用顏色標記
                    st.dataframe(
                        comp_df[['欄位', '相似度']],
                        use_container_width=True
                    )

                cont_comparisons = [
                    c for c in comparisons
                    if c.get('type') == 'continuous'
                ]
                if cont_comparisons:
                    st.write("**連續型變數統計比較：**")
                    cont_df = pd.DataFrame([
                        {
                            "欄位": c['column'],
                            "真實均值": c['real_stats']['mean'],
                            "合成均值": c['synthetic_stats']['mean'],
                            "真實標準差": c['real_stats']['std'],
                            "合成標準差": c['synthetic_stats']['std']
                        }
                        for c in cont_comparisons
                    ])
                    st.dataframe(cont_df, use_container_width=True)

            # 合成數據預覽
            with st.expander("🔍 合成數據預覽"):
                st.dataframe(
                    synthetic_data.head(20),
                    use_container_width=True
                )

            # 下載按鈕
            st.divider()
            st.subheader("📥 下載結果")

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                buffer1 = io.BytesIO()
                synthetic_data.to_excel(
                    buffer1, index=False, engine='openpyxl'
                )
                buffer1.seek(0)

                st.download_button(
                    label="📥 下載合成數據 (Excel)",
                    data=buffer1,
                    file_name="synthetic_data.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )

            with col_dl2:
                real_data_marked = real_data.copy()
                real_data_marked['is_synthetic'] = 0

                all_cols = list(real_data_marked.columns)
                for col in synthetic_data.columns:
                    if col not in all_cols:
                        all_cols.append(col)

                for col in all_cols:
                    if col not in real_data_marked.columns:
                        real_data_marked[col] = np.nan
                    if col not in synthetic_data.columns:
                        synthetic_data[col] = np.nan

                combined = pd.concat(
                    [real_data_marked[all_cols], synthetic_data[all_cols]],
                    ignore_index=True
                )

                buffer2 = io.BytesIO()
                combined.to_excel(
                    buffer2, index=False, engine='openpyxl'
                )
                buffer2.seek(0)

                st.download_button(
                    label="📥 下載合併數據（真實+合成）",
                    data=buffer2,
                    file_name="combined_data.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )

            # 下載質量報告
            report_json = json.dumps(
                report, indent=2, ensure_ascii=False
            )
            st.download_button(
                label="📥 下載質量報告 (JSON)",
                data=report_json,
                file_name="quality_report.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"生成過程出錯: {str(e)}")
            st.exception(e)

elif real_data is None and xml_file is None:
    st.info("👆 請先上傳真實數據和問卷 XML 文件")
elif real_data is None:
    st.warning("⚠️ 請上傳真實數據 Excel 文件")
elif parsed_xml is None:
    st.warning("⚠️ 請上傳問卷 XML 文件")

# ===== 頁腳 =====
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Synthetic Survey Data Generator v1.0 | 
    合成數據已標記 is_synthetic=1 以區分真實數據 |
    支持 OpenAI & OpenRouter API
    </div>
    """,
    unsafe_allow_html=True
)
