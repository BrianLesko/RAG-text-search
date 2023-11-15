##################################################
# About the author: Brian Lesko is a robotics engineer and recent graduate
import streamlit as st
def about():
    with st.sidebar:
        col1, col2, = st.columns([1,5], gap="medium")

        with col1:
            st.image('docs/bl.png')

        with col2:
            st.write(""" 
            Hey it's Brian,
                     
            In this code you can ask a question about any .txt file upload and have chatGPT answer.

            """)

        col1, col2, col3, col4, col5, col6 = st.columns([1.1,1,1,1,1,1.5], gap="medium")
        with col2:
            # TWITTER
            st.write("[![X](https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-335095-blue/x-logo-blue.svg)](https://twitter.com/BrianJosephLeko)")
        with col3:
            # GITHUB
            st.write("[![Github](https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-335095-blue/github-mark-blue.svg)](https://github.com/BrianLesko)")
        with col4:
            # LINKEDIN
            st.write("[![LinkedIn](https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-335095-blue/linkedin-icon-blue.svg)](https://www.linkedin.com/in/brianlesko/)")
        with col5:
            # YOUTUBE
            "."
            #st.write("[![LinkedIn](https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-335095-blue/yt-logo-blue.svg)](https://www.linkedin.com/in/brianlesko/)")
        with col6:
            # BLOG Visual Study Code
            "."
            #"[![VSC]()](https://www.visualstudycode.com/)"

def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)