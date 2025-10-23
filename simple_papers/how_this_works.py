import streamlit as st


def display_how_this_works():
    """
    Display a concise explanation of how Simple Papers works using Streamlit components.
    """


    st.markdown("### 🎯 What is Simple Papers?")
    st.markdown("Makes scary academic papers less scary. Click things, get explanations. Revolutionary stuff!")
    
    st.video("https://youtu.be/NZFg6RpLY6M")

    st.markdown("### 🌐 Streamlit Cloud Version")
    st.warning("**This is pretty much just a demo version!** So you only see pre-parsed papers here. If you want to upload your own, you'll need to run this locally (see below).")
    
    st.markdown("### 🚀 How to Use This (It's Not Rocket Science)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 1. Pick a Paper**")
        st.markdown("• Select from the dropdown")
        st.markdown("• Marvel at the PDF viewer")
        
        st.markdown("**🎧 3. Listen (Optional)**")
        st.markdown("• Toggle audio in sidebar")
        st.markdown("• Let AI read to you")
        st.markdown("• Change voices when available")

        
    
    with col2:
        st.markdown("**👆2. Click Stuff**")
        st.markdown("• Click colored rectangles")
        st.markdown("• Read simplified explanations")
        
        st.markdown("🔍**4. Dig further**")
        st.markdown("• Click on highlighted keywords")
        st.markdown("• Bask in the glory of understanding")
        st.markdown("• Get lost in a million open browser tabs")
    
    st.markdown("---")
    
    st.markdown("### 🏠 Want to Run This Locally?")
    st.info("**Good news**: 😃 This will let you parse your own papers!  **Bad news**: You need a ton of API keys. 🫠")
    
    api_col1, api_col2 = st.columns(2)
    
    with api_col1:
        st.markdown("**Required APIs:**")
        st.markdown("• Agentic Document Extraction (for parsing)")
        st.markdown("• AWS (for AI models)")
    
    with api_col2:
        st.markdown("**Optional APIs:**")
        st.markdown("• ElevenLabs (fancy voices)")
        st.markdown("• OpenAI (basic voices)")
    
    st.markdown("Check `.streamlit/secrets.toml.example` for setup instructions.")
    
    st.markdown("---")
    
    st.markdown("### 🚧 This is just a POC")
    st.markdown("**Translation**: I just built this as a fun learning experience. If people actually use this thing, we'll make it less janky and easier to set up. So if you like it, spread the word! ")
    st.success("**TL;DR**: Click on rectangles → Read simplified explanations → Feel smarter (results may vary)")
    
