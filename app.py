import streamlit as st
from main import execute_research_query, ResearchResponse

# Page configuration
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="",
    layout="wide"
)

# Title and description
st.title(" AI Research Agent")
st.markdown("Ask me anything and I'll research it for you using AI-powered tools!")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for conversation history
with st.sidebar:
    st.header("Conversation History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    
    if st.session_state.history:
        for i, (query, response) in enumerate(st.session_state.history):
            with st.expander(f"Query {i+1}: {query[:50]}..."):
                st.write(f"**Topic:** {response.topic}")
                st.write(f"**Summary:** {response.summary[:200]}...")

# Main input area
query = st.chat_input("What would you like me to research?")

if query:
    # Add user query to chat
    with st.chat_message("user"):
        st.write(query)
    
    # Execute research with loading spinner
    with st.spinner("🔍 Researching... This may take a moment."):
        response, error = execute_research_query(query)
    
    if error:
        # Display error message
        with st.chat_message("assistant"):
            st.error(" An error occurred while processing your query.")
            st.error(f"Error: {error['error']}")
            with st.expander("View raw response"):
                st.json(error.get("raw_response", {}))
    else:
        # Display research results
        with st.chat_message("assistant"):
            # Topic
            st.markdown(f"##  {response.topic}")
            st.divider()
            
            # Summary
            st.markdown("###  Summary")
            st.markdown(response.summary)
            st.divider()
            
            # Sources
            with st.expander(f" Sources ({len(response.sources)})"):
                for i, source in enumerate(response.sources, 1):
                    st.markdown(f"{i}. {source}")
            
            # Tools Used
            with st.expander(f" Tools Used ({len(response.tools_used)})"):
                for tool in response.tools_used:
                    st.markdown(f"• {tool}")
        
        # Add to conversation history
        st.session_state.history.append((query, response))

# Display welcome message if no queries yet
if not st.session_state.history and not query:
    st.info(" Enter a research question above to get started!")
    st.markdown("""
    ### Example queries:
    - "What are the latest developments in quantum computing?"
    - "Explain the impact of climate change on ocean currents"
    - "Research the history of artificial intelligence"
    """)

# Footer with creator credit
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Created by <strong>Ryan Orozco</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

