import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Pipeline Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Pipeline - FastAPI Documentation")
st.caption("Production RAG system with hybrid search, citation verification, and confidence scoring")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ask Questions", "Documents", "Eval Results"])

if page == "Ask Questions":
    st.header("Ask a Question")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="How do I add authentication to FastAPI?"
        )
    with col2:
        top_k = st.slider("Results", 3, 10, 5)
        use_reranker = st.checkbox("Use reranker", value=True)

    if st.button("Ask", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/v1/ask",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "use_reranker": use_reranker
                    },
                    timeout=60
                )
                data = response.json()

                # Answer
                st.subheader("Answer")
                st.markdown(data["answer"])

                # Confidence scores
                st.subheader("Confidence Scores")
                conf = data["confidence"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Composite", f"{conf['composite_score']:.1%}")
                c2.metric("Retrieval", f"{conf['retrieval_confidence']:.1%}")
                c3.metric("Citations", f"{conf['citation_coverage']:.1%}")
                c4.metric("Completeness", f"{conf['completeness']:.1%}")

                # Citations
                if data["citations"]:
                    st.subheader("Citation Verification")
                    for idx, citation in data["citations"].items():
                        status = "✅" if citation.get("supported") else "❌"
                        confidence = citation.get("confidence", 0)
                        source = citation.get("source", "unknown")
                        reason = citation.get("reason", "")
                        with st.expander(f"{status} [{idx}] {source} (confidence: {confidence:.0%})"):
                            st.write(reason)

                # Sources
                st.subheader("Sources Used")
                for source in data["sources"]:
                    st.markdown(f"📄 `{source}`")

            except Exception as e:
                st.error(f"Error: {e}")

elif page == "Documents":
    st.header("Indexed Documents")
    
    try:
        response = requests.get(f"{API_URL}/v1/documents")
        data = response.json()
        
        col1, col2 = st.columns(2)
        col1.metric("Total Chunks", data["total_chunks"])
        col2.metric("Total Documents", data["total_documents"])
        
        st.subheader("Document List")
        for doc in data["documents"]:
            st.markdown(f"📄 `{doc}`")
            
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "Eval Results":
    st.header("Evaluation Results")
    
    try:
        response = requests.get(f"{API_URL}/v1/eval/results")
        data = response.json()
        
        # Summary metrics
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correctness", f"{data['avg_correctness']:.1%}")
        c2.metric("Source Retrieval", f"{data['avg_source_retrieval']:.1%}")
        c3.metric("Confidence", f"{data['avg_confidence']:.1%}")
        c4.metric("Citation Accuracy", f"{data['citation_accuracy']:.1%}")
        
        # Per question results
        st.subheader("Per Question Results")
        for result in data["results"]:
            difficulty_color = {
                "easy": "🟢",
                "medium": "🟡", 
                "hard": "🔴",
                "unanswerable": "⚪"
            }.get(result["difficulty"], "⚪")
            
            correctness = result["correctness_scores"]["overall"]
            
            with st.expander(
                f"{difficulty_color} [{result['id']}] {result['question'][:70]}... "
                f"| Correctness: {correctness:.0%}"
            ):
                st.markdown(f"**Difficulty:** {result['difficulty']}")
                st.markdown(f"**Answer:** {result['answer'][:500]}...")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Correctness", f"{correctness:.0%}")
                col2.metric(
                    "Source Retrieval",
                    f"{result['source_retrieval_score']:.0%}"
                )
                col3.metric(
                    "Confidence",
                    f"{result['confidence']['composite_score']:.0%}"
                )
                
    except Exception as e:
        st.error(f"Error connecting to API. Make sure the API is running.")
        st.code("uvicorn main:app --reload")