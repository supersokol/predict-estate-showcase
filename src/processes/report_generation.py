from fpdf import FPDF
import pandas as pd
from src.registry.process_registry import register_process

@register_process("Generate PDF Report", metadata={"description": "Creates a PDF report for a dataset."})
def generate_pdf_report(data: pd.DataFrame, file_path: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="EDA Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"File: {file_path}", ln=True)
    pdf.cell(200, 10, txt=f"Rows: {len(data)}, Columns: {len(data.columns)}", ln=True)
    pdf.output("eda_report.pdf")
    return "eda_report.pdf"
