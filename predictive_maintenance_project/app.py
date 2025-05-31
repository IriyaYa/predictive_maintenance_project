import streamlit as st

st.sidebar.title("Меню")
page = st.sidebar.radio("Выберите страницу:",
                        ["Анализ данных", "Презентация проекта"])

if page == "Анализ данных":
    import analysis_and_model
    analysis_and_model.main()
else:
    import presentation
    presentation.main()