import streamlit as st
import os
from typing import Dict, List
from datetime import datetime
from src.registry import data_source_registry
from src.core.file_utils import get_context
from src.core.logger import logger
from src.core.file_utils import load_data
from src.core.integrations.assistants_api_manger import AssistantManager
import asyncio

# Model provider configurations
MODEL_PROVIDERS = load_data('Q:\SANDBOX\PredictEstateShowcase_dev\config\llm_provider_config.json', file_type="json", load_as="json")["providers"]

def render_context_source(index: int, data_sources: List[str]) -> Dict:
    """Render a single context source selector."""
    st.subheader(f"Context Source {index + 1}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        source_type = st.selectbox(
            "Source Type",
            data_sources,
            key=f"source_type_{index}"
        )
    
    source_config = {"type": source_type}
    
    if source_type == "csv":
        # Get available CSV files from DataSourceRegistry
        available_csvs = data_source_registry.get_files_by_type("csv")
        if available_csvs:
            source_config["file"] = st.selectbox(
                "Select CSV file",
                available_csvs,
                key=f"csv_{index}"
            )
            if st.checkbox("Preview data", key=f"preview_csv_{index}"):
                df = data_source_registry.get_table(source_config["file"])
                st.dataframe(df.head())
    
    elif source_type == "pdf":
        available_pdfs = data_source_registry.get_files_by_type("pdf")
        if available_pdfs:
            source_config["file"] = st.selectbox(
                "Select PDF file",
                available_pdfs,
                key=f"pdf_{index}"
            )
            
    elif source_type == "wikipedia":
        source_config["query"] = st.text_input(
            "Wikipedia Article Title",
            key=f"wiki_{index}"
        )
        
    elif source_type == "text":
        source_config["text"] = st.text_area(
            "Enter Text",
            key=f"text_{index}"
        )
    
    return source_config

def render_model_config(provider: str, model_config: Dict) -> Dict:
    """Render model configuration options."""
    provider_config = MODEL_PROVIDERS[provider]
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "Model",
            [m["id"] for m in provider_config["models"]],
            key="model_select"
        )
    
    st.write("Model Parameters:")
    parameters = {}
    
    for param_name, param_config in provider_config["parameters"].items():
        parameters[param_name] = st.slider(
            param_name,
            min_value=param_config["min"],
            max_value=param_config["max"],
            value=param_config["default"],
            step=param_config["step"],
            key=f"param_{param_name}"
        )
    
    return {
        "id": selected_model,
        "parameters": parameters
    }

async def process_query(query: str, context_sources: List[Dict], provider: str, model_config: Dict) -> Dict:
    """Process the query using selected model and context sources."""
    if provider == "openai-assistant":
        try:
            # Initialize AssistantManager if not exists in session state
            if "assistant_manager" not in st.session_state:
                st.session_state.assistant_manager = AssistantManager()
                
                # Create new assistant with custom instructions
                instructions = """You are a helpful assistant that analyzes data and provides insights. 
                You can work with various data formats including CSV, PDF, and text documents.
                When analyzing data, always:
                1. Understand the context and data structure
                2. Use appropriate analytical tools
                3. Provide clear explanations
                4. Include relevant code or calculations when needed"""
                
                st.session_state.assistant_manager.create_assistant(
                    name="DataAnalysisAssistant",
                    instructions=instructions
                )
                
                # Create new thread
                st.session_state.assistant_manager.create_thread()
            
            # Add context files
            file_ids = await st.session_state.assistant_manager.add_context_files(context_sources)
            
            # Process message
            result = await st.session_state.assistant_manager.process_message(query)
            
            # Save conversation history
            conversation_path = st.session_state.assistant_manager.save_conversation()
            
            return {
                "text": result["response"],
                "sources": [s.get("file", s.get("query", "text")) for s in context_sources],
                "conversation_path": conversation_path
            }
            
        except Exception as e:
            logger.error(f"Error processing with Assistant API: {e}")
            raise
    """Process the query using selected model and context sources."""
    try:
        # Collect context from all sources
        context_texts = []
        for source in context_sources:
            if source["type"] == "csv":
                df = data_source_registry.get_table(source["file"])
                context_texts.append(f"CSV Data from {source['file']}:\n{df.head().to_string()}")
            elif source["type"] == "pdf":
                text = data_source_registry.get_file_content(source["file"])
                context_texts.append(f"PDF Content from {source['file']}:\n{text[:1000]}...")
            elif source["type"] == "wikipedia":
                wiki_text = get_context("wiki", term=source["query"])
                context_texts.append(f"Wikipedia content for '{source['query']}':\n{wiki_text[:1000]}...")
            elif source["type"] == "text":
                context_texts.append(source["text"])
        
        combined_context = "\n\n".join(context_texts)
        
        # Process with selected model
        if provider == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model=model_config["id"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing data and providing insights."},
                    {"role": "user", "content": f"Context:\n{combined_context}\n\nQuery: {query}"}
                ],
                **model_config["parameters"]
            )
            return {
                "text": response.choices[0].message.content,
                "token_usage": response.usage.to_dict(),
                "sources": [s.get("file", s.get("query", "text")) for s in context_sources]
            }
        
        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model_config["id"],
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{combined_context}\n\nQuery: {query}"
                    }
                ],
                max_tokens=4000,
                temperature=model_config["parameters"]["temperature"]
            )
            return {
                "text": response.content[0].text,
                "sources": [s.get("file", s.get("query", "text")) for s in context_sources]
            }
            
        elif provider == "mistral":
            from mistralai.client import MistralClient
            client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
            response = client.chat(
                model=model_config["id"],
                messages=[
                    {"role": "user", "content": f"Context:\n{combined_context}\n\nQuery: {query}"}
                ],
                **model_config["parameters"]
            )
            return {
                "text": response.messages[0].content,
                "sources": [s.get("file", s.get("query", "text")) for s in context_sources]
            }
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

def render():
    """Main rendering function for the Enhanced Universal Task Processor."""
    st.title("Enhanced Universal Task Processor")
    # Conversation history management
    st.sidebar.header("Conversation History")
    
    # Load previous conversation
    if st.sidebar.checkbox("Load Previous Conversation"):
        conversation_dir = "data/conversations"
        if os.path.exists(conversation_dir):
            conversation_files = os.listdir(conversation_dir)
            if conversation_files:
                selected_file = st.sidebar.selectbox(
                    "Select Conversation",
                    conversation_files,
                    format_func=lambda x: x.replace(".json", "")
                )
                
                if st.sidebar.button("Load Conversation"):
                    file_path = os.path.join(conversation_dir, selected_file)
                    if "assistant_manager" not in st.session_state:
                        st.session_state.assistant_manager = AssistantManager()
                    conversation = st.session_state.assistant_manager.load_conversation(file_path)
                    
                    # Display conversation history
                    st.sidebar.subheader("Conversation History")
                    for msg in conversation["messages"]:
                        with st.sidebar.chat_message(msg["role"]):
                            st.write(msg["content"])
    # User query
    query = st.text_area("Enter your query", height=100)
    
    # Context sources
    st.header("Context Sources")
    available_source_types = ["csv", "pdf", "wikipedia", "text"]
    
    # Initialize session state for context sources if not exists
    if "context_sources" not in st.session_state:
        st.session_state.context_sources = []
    
    # Add new context source button
    if st.button("Add Context Source") and len(st.session_state.context_sources) < 5:
        st.session_state.context_sources.append({})
    
    # Render existing context sources
    context_configs = []
    for i, _ in enumerate(st.session_state.context_sources):
        with st.expander(f"Context Source {i + 1}", expanded=True):
            source_config = render_context_source(i, available_source_types)
            context_configs.append(source_config)
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.context_sources.pop(i)
                st.rerun()
    
    # Model selection
    st.header("Model Selection")
    
    provider = st.selectbox(
        "Select Provider",
        list(MODEL_PROVIDERS.keys()),
        format_func=lambda x: MODEL_PROVIDERS[x]["name"]
    )
    
    model_config = render_model_config(provider, MODEL_PROVIDERS[provider])
    
    # Process button
    if st.button("Process Query", disabled=not query):
        try:
            with st.spinner("Processing..."):
                result = process_query(query, context_configs, provider, model_config)
            
            st.header("Result")
            st.write(result["text"])
            
            if "token_usage" in result:
                st.write("Token Usage:", result["token_usage"])
            
            if result["sources"]:
                st.write("Sources used:", ", ".join(result["sources"]))
                
            # Save result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = f"data/results/query_result_{timestamp}.json"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            
            with open(result_path, "w") as f:
                import json
                json.dump({
                    "query": query,
                    "context_sources": context_configs,
                    "model_config": model_config,
                    "result": result,
                    "timestamp": timestamp
                }, f, indent=2)
            
            st.download_button(
                "Download Result",
                json.dumps(result, indent=2),
                file_name=f"query_result_{timestamp}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.exception("Error in query processing")