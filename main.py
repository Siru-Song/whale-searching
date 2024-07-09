import streamlit as st
import openai
import requests
from langchain import LangChain
from langchain.react import ReAct

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]

# Initialize LangChain
react = ReAct()

def serper_search(query):
    url = "https://api.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key
    }
    params = {
        "q": query
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def openai_query(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def main():
    st.title("Searching Whale")
    st.image("Whale.png", width=300)
    
    # Input from user
    user_input = st.text_input("Enter your query:")
    
    if st.button("Search"):
        # Display the thinking chain
        st.write("### Thinking Chain")
        
        # Perform Serper search
        st.write("**Step 1: Performing Google Search...**")
        search_results = serper_search(user_input)
        
        # Display search results
        st.write("### Google Search Results")
        for result in search_results['organic']:
            st.write(f"**{result['title']}**")
            st.write(result['link'])
            st.write(result['snippet'])
        
        # Use ReAct to process the input
        st.write("**Step 2: Processing Input with ReAct...**")
        react_result = react.process(user_input)
        st.write(f"ReAct Output: {react_result}")
        
        # Query OpenAI API with ReAct result
        st.write("**Step 3: Querying OpenAI API...**")
        openai_result = openai_query(react_result)
        
        # Display OpenAI result
        st.write("### OpenAI Response")
        st.write(openai_result)

if __name__ == "__main__":
    main()
