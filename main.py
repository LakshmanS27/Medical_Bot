import streamlit as st
from chain_module import get_qa_chain, create_vector_db

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺")
st.title("🩺 Medical Q&A Chatbot")
st.markdown("Ask medical questions based on the MedQuAD dataset. Click below to (re)build the knowledge base if needed.")

if st.button("Create Knowledge Base"):
    with st.spinner("Processing MedQuAD and building vector DB..."):
        try:
            create_vector_db()
            st.success("✅ Knowledge base created!")
        except Exception as e:
            st.error(f"❌ Failed to create knowledge base: {e}")

question = st.text_input("Ask a medical question:")

if question:
    with st.spinner("Searching for answer..."):
        try:
            chain = get_qa_chain()
            response = chain.invoke({"query": question})

            st.header("Answer")
            st.write(response["result"])

            if response.get("source_documents"):
                with st.expander("Context & Sources"):
                    for doc in response["source_documents"]:
                        st.markdown(f"**Document Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(doc.page_content)
        except Exception as e:
            st.error(f"❌ Error answering question: {e}")
