import streamlit as st
import pandas as pd
import plotly.express as px
import json
from fpdf import FPDF
import os

        
# Загрузка конфигурационного файла
def load_config(config_path="Q:\SANDBOX\PredictEstateShowcase_dev\data\current_data_config.json"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        return {}

# Функция генерации PDF
def generate_pdf(data, file_metadata, output_path="eda_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Заголовок
    pdf.set_font("Arial", "B", size=14)
    pdf.cell(200, 10, txt=f"EDA Report for {file_metadata['name']}", ln=True, align='C')
    pdf.ln(10)
    
    # Метаданные
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Timestamp: {file_metadata['timestamp']}", ln=True)
    pdf.cell(200, 10, txt=f"Columns: {file_metadata['columns']}", ln=True)
    pdf.cell(200, 10, txt=f"Rows: {file_metadata['rows']}", ln=True)
    pdf.ln(10)
    
    # Добавление базового анализа
    pdf.cell(200, 10, txt="Basic Statistics:", ln=True)
    pdf.ln(5)
    stats = data.describe().transpose()
    for col in stats.index:
        pdf.cell(200, 10, txt=f"{col}: {stats.loc[col].to_dict()}", ln=True)
        pdf.ln(3)
    
    # Сохранение
    pdf.output(output_path)
    return output_path

# Streamlit UI
def main():
    st.title("EDA Module for Uploaded Files")
    
    # Загрузка конфигурации
    config = load_config()
    
    if not config:
        st.error("No files found in configuration.")
        return
    
    # Выбор файла
    file_options = list(config.keys())
    selected_file = st.selectbox("Select a file for analysis", file_options)
    
    if selected_file:
        # Загрузка данных
        data = pd.read_csv(selected_file)
        metadata = config[selected_file]
        
        # Отображение метаданных
        st.subheader("File Metadata")
        st.write(metadata)
        
        # Выбор графиков
        st.subheader("Visualization")
        column_options = data.columns.tolist()
        selected_column = st.selectbox("Select a column for visualization", column_options)
        
        if selected_column:
            # Гистограмма
            st.plotly_chart(px.histogram(data, x=selected_column, title=f"Histogram for {selected_column}"))
            
            # Боксплот
            st.plotly_chart(px.box(data, y=selected_column, title=f"Boxplot for {selected_column}"))
        
        # Генерация отчета
        st.subheader("Generate Report")
        if st.button("Generate PDF Report"):
            pdf_path = generate_pdf(data, metadata)
            st.success(f"Report generated: {pdf_path}")
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Report",
                    data=pdf_file,
                    file_name="eda_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
