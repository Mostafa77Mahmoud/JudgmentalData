import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from src.knowledge_base import AAOIFIKnowledgeBase
from src.dataset_generator import DatasetGenerator
from src.data_processor import DataProcessor

# Page configuration
st.set_page_config(
    page_title="AAOIFI Islamic Finance Knowledge System",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

def initialize_system():
    """Initialize the AAOIFI knowledge system"""
    try:
        # Check for required files
        required_files = [
            "inputs/arabic_cleaned.txt",
            "inputs/english_cleaned.txt", 
            "inputs/arabic_chunks.json",
            "inputs/english_chunks.json",
            "inputs/arabic_qa_pairs.json",
            "inputs/english_qa_pairs.json"
        ]

        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            st.error(f"Missing required files: {missing_files}")
            return False

        # Check for API keys from config/keys.json
        keys_file = Path("config/keys.json")
        api_keys = []
        if keys_file.exists():
            with open(keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
                # Extract keys from the "keys" array in the config structure
                if isinstance(keys_data, dict) and "keys" in keys_data:
                    api_keys.extend(keys_data["keys"])
                elif isinstance(keys_data, list):
                    api_keys.extend(keys_data)
                elif isinstance(keys_data, dict):
                    api_keys.extend(keys_data.values())
                else:
                    st.warning("Unexpected format in keys.json. API keys might not be loaded correctly.")

        # Fallback to environment variables if config file is missing or empty
        if not api_keys:
            api_keys = [
                os.getenv("GEMINI_API_KEY"),
                os.getenv("GEMINI_KEY_1"),
                os.getenv("GEMINI_KEY_2"), 
                os.getenv("GEMINI_KEY_3"),
                os.getenv("GEMINI_KEY_4")
            ]

        valid_keys = [key for key in api_keys if key]
        if not valid_keys:
            st.error("No valid Gemini API keys found. Please set GEMINI_API_KEY or GEMINI_KEY_1-4 environment variables, or provide keys in config/keys.json.")
            return False

        # Initialize components
        st.session_state.processor = DataProcessor()
        st.session_state.knowledge_base = AAOIFIKnowledgeBase()
        # Pass the config file path to DatasetGenerator
        st.session_state.generator = DatasetGenerator("config/keys.json")

        # Load data
        st.session_state.processor.load_data()
        st.session_state.knowledge_base.load_data(st.session_state.processor)

        return True

    except FileNotFoundError:
        st.error("Configuration file 'config/keys.json' not found. Please ensure it exists or set environment variables.")
        return False
    except json.JSONDecodeError:
        st.error("Error decoding 'config/keys.json'. Please ensure it's valid JSON.")
        return False
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def main():
    st.title("🕌 AAOIFI Islamic Finance Knowledge System")
    st.markdown("*An AI-powered system for processing AAOIFI Shari'ah standards with verified reference validation*")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Knowledge Base Query", "Dataset Generation", "System Status", "Data Explorer"]
        )

        st.header("System Info")
        if st.button("Initialize System"):
            with st.spinner("Initializing AAOIFI system..."):
                if initialize_system():
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ System initialization failed!")

    # Main content based on selected page
    if page == "Knowledge Base Query":
        knowledge_base_page()
    elif page == "Dataset Generation":
        dataset_generation_page()
    elif page == "System Status":
        system_status_page()
    elif page == "Data Explorer":
        data_explorer_page()

def knowledge_base_page():
    st.header("📚 Knowledge Base Query")

    if st.session_state.knowledge_base is None:
        st.warning("Please initialize the system first using the sidebar.")
        return

    # Language selection
    language = st.selectbox("Select Language", ["English", "Arabic"])
    lang_code = "en" if language == "English" else "ar"

    # Query input
    query = st.text_area(
        "Enter your Islamic finance question:",
        placeholder="e.g., What are the rules for currency trading in Islamic finance?",
        height=100
    )

    if st.button("Search Knowledge Base"):
        if query:
            with st.spinner("Searching AAOIFI standards..."):
                results = st.session_state.knowledge_base.search(query, lang_code)

                if results:
                    st.success(f"Found {len(results)} relevant results")

                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - Confidence: {result['confidence']:.2f}"):
                            st.markdown(f"**Standard:** {result.get('standard', 'N/A')}")
                            st.markdown(f"**Chunk ID:** {result['chunk_id']}")
                            st.markdown(f"**Content:**")
                            st.text_area("", result['content'], height=150, disabled=True, key=f"result_{i}")
                            if result.get('reference'):
                                st.markdown(f"**Reference:** {result['reference']}")
                else:
                    st.info("No relevant results found in the AAOIFI standards.")
        else:
            st.warning("Please enter a query.")

def dataset_generation_page():
    st.header("🔧 Dataset Generation")

    if st.session_state.generator is None:
        st.warning("Please initialize the system first using the sidebar.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Smoke Test")
        st.markdown("Generate a small sample to validate the system")

        # Language selection for smoke test
        smoke_language = st.selectbox("Select Language for Smoke Test", ["Arabic", "English"], key="smoke_lang")
        smoke_target = st.number_input("Number of examples for smoke test", min_value=10, max_value=50, value=20, key="smoke_target")

        if st.button("🧪 Run Smoke Test"):
            with st.spinner("Running smoke test..."):
                try:
                    # Convert language name to language code
                    lang_code = "ar" if smoke_language == "Arabic" else "en"
                    results = st.session_state.generator.run_smoke_test(
                        language=lang_code,
                        target_count=smoke_target
                    )

                    if results['success']:
                        st.success("✅ Smoke test completed successfully!")
                        st.json(results['stats'])

                        # Show sample results
                        if results.get('samples'):
                            st.subheader("Sample Results")
                            for i, sample in enumerate(results['samples'][:3], 1):
                                with st.expander(f"Sample {i}"):
                                    st.json(sample)
                    else:
                        st.error(f"❌ Smoke test failed: {results.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Error during smoke test: {str(e)}")

    with col2:
        st.subheader("Full Generation")
        st.markdown("Generate the complete dataset after smoke test passes")

        target_count = st.number_input("Target examples per language", min_value=100, max_value=5000, value=2000)

        # Add strict mode toggle
        st.subheader("Generation Mode")
        use_strict_mode = st.checkbox("Use Strict Local-First Mode", 
                                    value=True, 
                                    help="Reduces API calls and prevents hallucination by using local verification first")

        if st.button("Generate Full Dataset"):
            # Check if smoke test output exists
            ar_files = list(Path("data/generation_stage_B/ar").glob("smoke_test_ar_*.jsonl"))
            en_files = list(Path("data/generation_stage_B/en").glob("smoke_test_en_*.jsonl"))

            if not ar_files and not en_files:
                st.warning("Please run and pass the smoke test first.")
                return

            with st.spinner("Generating full dataset... This may take a while."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Arabic generation
                    status_text.text("Generating Arabic dataset...")
                    ar_results = st.session_state.generator.generate_full_dataset("ar", target_count, progress_bar, strict_mode=use_strict_mode)

                    if ar_results['success']:
                        progress_bar.progress(0.5)
                        status_text.text("Generating English dataset...")

                        # English generation
                        en_results = st.session_state.generator.generate_full_dataset("en", target_count, progress_bar, strict_mode=use_strict_mode)

                        if en_results['success']:
                            progress_bar.progress(1.0)
                            status_text.text("Generation completed!")

                            st.success("✅ Full dataset generation completed!")

                            # Show final stats
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Arabic Examples", ar_results['stats']['total'])
                                st.metric("Arabic True/False", f"{ar_results['stats']['true']}/{ar_results['stats']['false']}")
                            with col2:
                                st.metric("English Examples", en_results['stats']['total'])
                                st.metric("English True/False", f"{en_results['stats']['true']}/{en_results['stats']['false']}")
                        else:
                            st.error(f"English generation failed: {en_results.get('error')}")
                    else:
                        st.error(f"Arabic generation failed: {ar_results.get('error')}")

                except Exception as e:
                    st.error(f"Error during full generation: {str(e)}")

def system_status_page():
    st.header("📊 System Status")

    # File status
    st.subheader("Required Files Status")
    required_files = [
        "inputs/arabic_cleaned.txt",
        "inputs/english_cleaned.txt",
        "inputs/arabic_chunks.json", 
        "inputs/english_chunks.json",
        "inputs/arabic_qa_pairs.json",
        "inputs/english_qa_pairs.json"
    ]

    file_status = []
    for file_path in required_files:
        exists = Path(file_path).exists()
        size = Path(file_path).stat().st_size if exists else 0
        file_status.append({
            "File": file_path,
            "Status": "✅ Found" if exists else "❌ Missing",
            "Size (MB)": f"{size / (1024*1024):.2f}" if exists else "N/A"
        })

    st.dataframe(pd.DataFrame(file_status))

    # API Keys status
    st.subheader("API Keys Status")
    # Add strict mode toggle
    st.subheader("Generation Mode")
    use_strict_mode = st.checkbox("Use Strict Local-First Mode", 
                                value=True, 
                                help="Reduces API calls and prevents hallucination by using local verification first")

    # Check keys from config file first, then environment variables
    keys_from_config = []
    keys_file = Path("config/keys.json")
    if keys_file.exists():
        try:
            with open(keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
                # Extract keys from the "keys" array in the config structure
                if isinstance(keys_data, dict) and "keys" in keys_data:
                    keys_from_config.extend(keys_data["keys"])
                elif isinstance(keys_data, list):
                    keys_from_config.extend(keys_data)
                elif isinstance(keys_data, dict):
                    keys_from_config.extend(keys_data.values())
        except (FileNotFoundError, json.JSONDecodeError):
            pass # Ignore errors here, will be reported by env vars

    env_keys = [
        ("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
        ("GEMINI_KEY_1", os.getenv("GEMINI_KEY_1")),
        ("GEMINI_KEY_2", os.getenv("GEMINI_KEY_2")),
        ("GEMINI_KEY_3", os.getenv("GEMINI_KEY_3")),
        ("GEMINI_KEY_4", os.getenv("GEMINI_KEY_4"))
    ]

    all_keys_info = {}
    for key_name, key_value in env_keys:
        all_keys_info[key_name] = key_value

    for key in keys_from_config:
        # Add keys from config if they are not already present from env vars,
        # or update if env var is missing but config has it.
        # This logic might need refinement based on how you want to prioritize.
        # For now, we'll just list all found keys.
        # A simple approach is to just check if any key exists.
        pass # Keys are already captured in all_keys_info if they were env vars

    key_status = []
    # Display keys found in config file
    for i, key in enumerate(keys_from_config):
        key_status.append({
            "Key Source": "config/keys.json",
            "Key Name": f"Key {i+1}",
            "Status": "✅ Set",
            "Value": f"{str(key)[:10]}..." if key else "Not set"
        })

    # Display environment variables
    for key_name, key_value in env_keys:
        status = "✅ Set" if key_value else "❌ Missing"
        masked_key = f"{str(key_value)[:10]}..." if key_value else "Not set"
        # Avoid duplicating if key from config is the same as env var
        if key_name not in all_keys_info or all_keys_info[key_name] != key_value:
             key_status.append({
                "Key Source": "Environment Variable",
                "Key Name": key_name,
                "Status": status,
                "Value": masked_key
            })

    st.dataframe(pd.DataFrame(key_status))


    # Progress status
    st.subheader("Generation Progress")
    progress_file = Path("progress/state.json")
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            st.json(progress_data)
        except Exception as e:
            st.error(f"Error reading progress file: {str(e)}")
    else:
        st.info("No generation progress found.")

def data_explorer_page():
    st.header("🔍 Data Explorer")

    if st.session_state.processor is None:
        st.warning("Please initialize the system first using the sidebar.")
        return

    # Data overview
    st.subheader("Data Overview")

    try:
        # Arabic data
        ar_chunks = st.session_state.processor.arabic_chunks
        ar_qa = st.session_state.processor.arabic_qa_pairs

        # English data  
        en_chunks = st.session_state.processor.english_chunks
        en_qa = st.session_state.processor.english_qa_pairs

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Arabic Chunks", len(ar_chunks))
            st.metric("Arabic Q&A Pairs", len(ar_qa))

        with col2:
            st.metric("English Chunks", len(en_chunks))
            st.metric("English Q&A Pairs", len(en_qa))

        # Sample data viewer
        st.subheader("Sample Data")

        data_type = st.selectbox("Select data type to explore", 
                                ["Arabic Chunks", "English Chunks", "Arabic Q&A", "English Q&A"])

        if data_type == "Arabic Chunks" and ar_chunks:
            chunk_id = st.selectbox("Select chunk ID", range(len(ar_chunks)))
            chunk = ar_chunks[chunk_id]
            st.json({
                "id": chunk.get("id"),
                "word_count": chunk.get("word_count"),
                "language": chunk.get("language"),
                "text_preview": chunk.get("text", "")[:500] + "..." if len(chunk.get("text", "")) > 500 else chunk.get("text", "")
            })

        elif data_type == "English Chunks" and en_chunks:
            chunk_id = st.selectbox("Select chunk ID", range(len(en_chunks)))
            chunk = en_chunks[chunk_id]
            st.json({
                "id": chunk.get("id"),
                "word_count": chunk.get("word_count"), 
                "language": chunk.get("language"),
                "text_preview": chunk.get("text", "")[:500] + "..." if len(chunk.get("text", "")) > 500 else chunk.get("text", "")
            })

        elif data_type == "Arabic Q&A" and ar_qa:
            qa_id = st.selectbox("Select Q&A ID", range(len(ar_qa)))
            qa_pair = ar_qa[qa_id]
            st.json(qa_pair)

        elif data_type == "English Q&A" and en_qa:
            qa_id = st.selectbox("Select Q&A ID", range(len(en_qa)))
            qa_pair = en_qa[qa_id]
            st.json(qa_pair)
        else:
            st.info("No data available for the selected type.")

    except Exception as e:
        st.error(f"Error exploring data: {str(e)}")

if __name__ == "__main__":
    main()