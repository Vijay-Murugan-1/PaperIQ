import pymupdf as pymnu
def extract_text_from_pdf(uploaded_file):
    doc = pymnu.open(stream = uploaded_file,filetype="pdf")
    text=" "
    for page in doc:
        text +=page.get_text() + "\n"
    return text
