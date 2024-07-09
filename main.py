import streamlit as st
import openai
import requests
from langchain import LangChain
from langchain.react import ReAct

# Set API keys
openai.api_key = 'YOUR_OPENAI_API_KEY'
serper_api_key = 'YOUR_SERPER_API_KEY'

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

def openai_query(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    st.title("Streamlit App with OpenAI and Serper API")
    
    # Input from user
    user_input = st.text_input("Enter your query:")
    
    if st.button("Search"):
        # Perform Serper search
        search_results = serper_search(user_input)
        
        # Display search results
        st.write("### Google Search Results")
        for result in search_results['organic']:
            st.write(f"**{result['title']}**")
            st.write(result['link'])
            st.write(result['snippet'])
        
        # Use ReAct to process the input
        react_result = react.process(user_input)
        
        # Query OpenAI API with ReAct result
        openai_result = openai_query(react_result)
        
        # Display OpenAI result
        st.write("### OpenAI Response")
        st.write(openai_result)

if __name__ == "__main__":
    main()
