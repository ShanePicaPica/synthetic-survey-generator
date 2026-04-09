"""
Synthetic Survey Data Generator
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from xml_parser import parse_decipher_xml, get_question_summary
from synthesizer import SurveySynthesizer, generate_quality_report

st.set_page_config(
    page_title="Survey Data Synthesizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Survey Data Synthesizer")
st.markdown("Upload real data + questionnaire XML → Generate synthetic data")

# ===== Session State =====
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None
if "real_data_for_download" not in st.session_state:
    st.session_state.real_data_for_download = None
if "generation_done" not in st.session_state:
    st.session_state.generation_done = False

# ===== Sidebar =====
with st.sidebar:
    st.header("Settings")

    api_provider = st.selectbox(
        "AI Provider",
        ["OpenRouter (Free models)", "OpenAI"]
    )

    if api_provider == "OpenRouter (Free models)":
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get from https://openrouter.ai/keys"
        )
        model_options = {
            "Gemma 3 27B (Free)": "google/gemma-3-27b-it:free",
            "Llama 3.1 8B (Free)": "meta-llama/llama-3.1-8b-instruct:free",
            "Qwen3 8B (Free)": "qwen/qwen3-8b:free",
            "Mistral 7B (Free)": "mistralai/mistral-7b-instruct:free",
            "DeepSeek V3 (Free)": "deepseek/deepseek-chat-v3-0324:free"
        }
        selected_model = st.selectbox(
            "Model",
            list(model_options.keys())
        )
        model_id = model_options[selected_model]
        st.info("Model: " + model_id)
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Format: sk-xxxx..."
        )
        model_options_openai = {
            "GPT-4o Mini": "gpt-4o-mini",
            "GPT-4o": "gpt-4o",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        selected_model = st.selectbox(
            "Model",
            list(model_options_openai.keys())
        )
        model_id = model_options_openai[selected_model]

    if api_key:
        st.success("API Key set")
    else:
        st.warning("No API Key - open text will use random sampling")

    st.divider()
    st.markdown("### How to use")
    st.markdown(
        "1. Set API Key (optional)\n"
        "2. Upload real data (Excel)\n"
        "3. Upload questionnaire XML\n"
        "4. Set sample size\n"
        "5. Click Generate\n"
        "6. Download results"
    )
    st.divider()
    st.markdown("[OpenRouter Keys](https://openrouter.ai/keys)")
    st.markdown("[OpenAI Keys](https://platform.openai.com/api-keys)")

# ===== Main =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Upload Real Data")
    data_file = st.file_uploader(
        "Upload Excel (.xlsx)",
        type=["xlsx", "xls"]
    )

with col2:
    st.subheader("Step 2: Upload Questionnaire")
    xml_file = st.file_uploader(
        "Upload Decipher XML (.xml or .txt)",
        type=["xml", "txt"]
    )

real_data = None
parsed_xml = None

if data_file:
    try:
        xls = pd.ExcelFile(data_file)
        sheet_names = xls.sheet_names

        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Select Sheet", sheet_names)
        else:
            selected_sheet = sheet_names[0]

        real_data = pd.read_excel(data_file, sheet_name=selected_sheet)

        with st.expander("Real Data Preview", expanded=False):
            row_count = len(real_data)
            col_count = len(real_data.columns)
            st.write("Rows: " + str(row_count) + " | Columns: " + str(col_count))
            st.dataframe(real_data.head(10), use_container_width=True)

            col_info = pd.DataFrame({
                "Column": real_data.columns,
                "Non-null": real_data.notna().sum().values,
                "Null%": (real_data.isna().mean() * 100).round(1).astype(str) + "%",
                "Unique": real_data.nunique().values
            })
            st.dataframe(col_info, use_container_width=True, height=300)

    except Exception as e:
        st.error("Failed to read Excel: " + str(e))

if xml_file:
    try:
        xml_content = xml_file.read().decode("utf-8")
        parsed_xml = parse_decipher_xml(xml_content)

        if "error" in parsed_xml:
            st.warning("XML parse warning: " + str(parsed_xml.get("error")))

        with st.expander("Questionnaire Preview", expanded=False):
            summary = get_question_summary(parsed_xml)
            st.write("Survey: " + str(summary["survey_name"]))
            st.write("Questions: " + str(summary["total_questions"]))

            st.write("Question types:")
            type_df = pd.DataFrame([
                {"Type": k, "Count": v}
                for k, v in summary["type_breakdown"].items()
            ])
            st.dataframe(type_df, use_container_width=True)

            cond_count = summary["conditional_questions"]
            st.write("Questions with skip logic: " + str(cond_count))

            st.write("Question list:")
            q_df = pd.DataFrame(summary["questions_list"])
            st.dataframe(q_df, use_container_width=True, height=400)

    except Exception as e:
        st.error("Failed to parse XML: " + str(e))

# ===== Generation Settings =====
st.divider()
st.subheader("Step 3: Generation Settings")

col_a, col_b, col_c = st.columns(3)

with col_a:
    n_samples = st.number_input(
        "Number of samples",
        min_value=10,
        max_value=10000,
        value=200,
        step=50
    )

with col_b:
    if real_data is not None:
        st.metric("Real data rows", len(real_data))

with col_c:
    if real_data is not None:
        st.metric("Total after merge", len(real_data) + n_samples)

with st.expander("Advanced Options", expanded=False):
    generate_open_text = st.checkbox(
        "Use AI for open-ended questions",
        value=True if api_key else False,
        help="Requires API Key"
    )

# ===== Generate Button =====
st.divider()

if real_data is not None and parsed_xml is not None:

    if st.button(
        "Generate Synthetic Data",
        type="primary",
        use_container_width=True
    ):
        st.session_state.synthetic_data = None
        st.session_state.quality_report = None
        st.session_state.generation_done = False

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(min(pct, 1.0))
            status_text.text(msg)

        try:
            use_api_key = api_key if (api_key and generate_open_text) else None

            synthesizer = SurveySynthesizer(
                real_data=real_data,
                parsed_xml=parsed_xml,
                openai_api_key=use_api_key,
                api_provider=api_provider,
                model_id=model_id
            )

            update_progress(0.05, "Analyzing real data distributions...")
            n_profiled = synthesizer.analyze_real_data()
            update_progress(0.1, "Profiled " + str(n_profiled) + " columns")

            synthetic_data = synthesizer.synthesize(
                n_samples=n_samples,
                progress_callback=update_progress
            )

            progress_bar.progress(1.0)
            status_text.text("Done!")

            st.session_state.synthetic_data = synthetic_data
            st.session_state.real_data_for_download = real_data.copy()
            st.session_state.quality_report = generate_quality_report(
                real_data, synthetic_data
            )
            st.session_state.generation_done = True

        except Exception as e:
            st.error("Generation error: " + str(e))
            st.exception(e)

    # ===== Display Results =====
    if st.session_state.generation_done:
        synthetic_data = st.session_state.synthetic_data
        report = st.session_state.quality_report
        real_data_saved = st.session_state.real_data_for_download

        if synthetic_data is not None:
            st.success(
                "Generated " + str(len(synthetic_data)) + " synthetic rows!"
            )

            # Quality Report
            with st.expander("Quality Report", expanded=True):
                if report and report.get("overall_quality_score"):
                    score = report["overall_quality_score"]
                    if score > 0.85:
                        label = "Excellent"
                    elif score > 0.7:
                        label = "Good"
                    else:
                        label = "Needs Improvement"

                    score_pct = str(round(score * 100, 1)) + "%"
                    st.markdown(
                        "### Distribution Similarity: "
                        + score_pct + " (" + label + ")"
                    )

                comparisons = report.get("column_comparisons", [])
                cat_comps = [
                    c for c in comparisons
                    if c.get("type") == "categorical"
                    and c.get("distribution_similarity") is not None
                ]

                if cat_comps:
                    st.write("Column-level similarity:")
                    comp_df = pd.DataFrame([
                        {
                            "Column": c["column"],
                            "Similarity": str(
                                round(c["distribution_similarity"] * 100, 1)
                            ) + "%"
                        }
                        for c in sorted(
                            cat_comps,
                            key=lambda x: x["distribution_similarity"]
                        )
                    ])
                    st.dataframe(comp_df, use_container_width=True)

                cont_comps = [
                    c for c in comparisons
                    if c.get("type") == "continuous"
                ]
                if cont_comps:
                    st.write("Continuous variable comparison:")
                    cont_df = pd.DataFrame([
                        {
                            "Column": c["column"],
                            "Real Mean": c["real_stats"]["mean"],
                            "Synth Mean": c["synthetic_stats"]["mean"],
                            "Real Std": c["real_stats"]["std"],
                            "Synth Std": c["synthetic_stats"]["std"]
                        }
                        for c in cont_comps
                    ])
                    st.dataframe(cont_df, use_container_width=True)

            # Preview
            with st.expander("Synthetic Data Preview"):
                st.dataframe(
                    synthetic_data.head(20), use_container_width=True
                )

            # ===== Downloads =====
            st.divider()
            st.subheader("Download Results")

            buf1 = io.BytesIO()
            synthetic_data.to_excel(buf1, index=False, engine="openpyxl")
            buf1.seek(0)
            synth_bytes = buf1.getvalue()

            real_marked = real_data_saved.copy()
            real_marked["is_synthetic"] = 0

            all_cols = list(real_marked.columns)
            for c in synthetic_data.columns:
                if c not in all_cols:
                    all_cols.append(c)

            for c in all_cols:
                if c not in real_marked.columns:
                    real_marked[c] = np.nan
            synth_copy = synthetic_data.copy()
            for c in all_cols:
                if c not in synth_copy.columns:
                    synth_copy[c] = np.nan

            combined = pd.concat(
                [real_marked[all_cols], synth_copy[all_cols]],
                ignore_index=True
            )

            buf2 = io.BytesIO()
            combined.to_excel(buf2, index=False, engine="openpyxl")
            buf2.seek(0)
            combined_bytes = buf2.getvalue()

            report_json = json.dumps(report, indent=2, ensure_ascii=False)

            dl1, dl2, dl3 = st.columns(3)

            with dl1:
                st.download_button(
                    label="Synthetic Data (Excel)",
                    data=synth_bytes,
                    file_name="synthetic_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with dl2:
                st.download_button(
                    label="Combined Data (Excel)",
                    data=combined_bytes,
                    file_name="combined_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with dl3:
                st.download_button(
                    label="Quality Report (JSON)",
                    data=report_json,
                    file_name="quality_report.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.info(
                "Download buttons persist. "
                "Click Generate again to create new data."
            )

elif real_data is None and xml_file is None:
    st.info("Please upload real data and questionnaire XML")
elif real_data is None:
    st.warning("Please upload real data Excel file")
elif parsed_xml is None:
    st.warning("Please upload questionnaire XML file")

st.divider()
st.caption("Synthetic Survey Generator v1.0 | Synthetic rows marked is_synthetic=1")
